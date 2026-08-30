"""
Twilio adapter — READY TO USE. No code changes needed: point a Twilio
phone number's "A call comes in" webhook at this service's /twilio/twiml
endpoint, and calls will flow through automatically.

Uses Twilio's Media Streams protocol (bidirectional, real-time audio over a
WebSocket) — good call quality, low latency, no third-party STT needed
since we do our own ASR.

Optional but recommended for production: set TWILIO_AUTH_TOKEN in .env to
enable request-signature validation on the /twiml webhook, so only requests
that genuinely came from Twilio are accepted. Implemented here with plain
HMAC-SHA1 (Twilio's documented scheme) — no twilio SDK dependency needed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import time
import wave
from importlib import resources
from typing import Optional
from urllib.parse import urlencode

import numpy as np
from fastapi import APIRouter, Form, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional runtime dependency
    ort = None

try:
    import webrtcvad
except ImportError:  # pragma: no cover - optional runtime dependency
    webrtcvad = None

try:
    from silero_vad import load_silero_vad
except ImportError:  # pragma: no cover - optional runtime dependency
    load_silero_vad = None

from maya_voice_os.asr_service.engine import ASREngine
from maya_voice_os.orchestration_service.pipeline import VoiceCommandRouter
from maya_voice_os.telephony_service.orchestration_client import get_client
from maya_voice_os.telephony_service.session_manager import session_manager
from maya_voice_os.shared.audio_utils import float32_to_mulaw_bytes, float32_to_pcm16_bytes, mulaw_bytes_to_float32, resample_audio
from maya_voice_os.tts_service.engine import get_provider as get_tts_provider

logger = logging.getLogger("twilio-adapter")
router = APIRouter(prefix="/twilio", tags=["twilio"])

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN") or None
PUBLIC_BASE_URL = os.getenv("TELEPHONY_PUBLIC_BASE_URL", "http://localhost:8100")
BARGE_IN_DEMO_MODE = os.getenv("BARGE_IN_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

TELEPHONY_SR = 8000
ASR_SR = 16000
VAD_THRESHOLD = 0.3
VOICEMAIL_CHECK_SECONDS = 5.0
VOICEMAIL_BEEP_FRAMES = 25  # ~500ms of sustained tone before silence
SILENCE_FRAMES_TO_END_TURN = 20  # 20 * 20ms = 400ms of low speech probability ends a turn
ESCALATION_KEYWORDS = (
    "let me speak to a human",
    "manager",
    "angry",
    "frustrated",
)
ESCALATION_PHONE_NUMBER = os.getenv("ESCALATION_PHONE_NUMBER", "+1234567890")

SILERO_VAD_MODEL = None
WEBRTC_VAD = None if webrtcvad is None else webrtcvad.Vad(3)
LOCAL_ASR_ENGINE = ASREngine()
VOICE_COMMAND_ROUTER = VoiceCommandRouter()
SAFE_MODE = os.getenv("SAFE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

if ort is not None and load_silero_vad is not None:
    try:
        # Silero VAD ships with an ONNX checkpoint; keep the session loaded once
        # and reuse it across incoming chunks for lower latency.
        SILERO_VAD_MODEL = load_silero_vad(onnx=True, force_onnx_cpu=True)
        if hasattr(SILERO_VAD_MODEL, "session"):
            SILERO_VAD_MODEL.session.set_providers(["CPUExecutionProvider"])
    except Exception:
        logger.warning("Silero VAD failed to initialize; falling back to webrtcvad.", exc_info=True)
        SILERO_VAD_MODEL = None


def _normalize_pcm16_for_vad(audio_16k: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def _webrtc_vad_probability(audio_16k: np.ndarray) -> float:
    if WEBRTC_VAD is None or audio_16k.size == 0:
        return 0.0
    frame = _normalize_pcm16_for_vad(audio_16k)
    if frame.size < 160:
        frame = np.pad(frame, (0, 160 - frame.size), mode="constant")
    pcm16 = np.clip(frame, -1.0, 1.0) * 32767.0
    pcm16 = pcm16.astype(np.int16)
    return 1.0 if WEBRTC_VAD.is_speech(pcm16.tobytes(), ASR_SR) else 0.0


def _silero_vad_probability(audio_16k: np.ndarray) -> float:
    if SILERO_VAD_MODEL is None or audio_16k.size == 0:
        return 0.0
    frame = _normalize_pcm16_for_vad(audio_16k)
    if frame.size < 256:
        frame = np.pad(frame, (0, 256 - frame.size), mode="constant")
    frame = frame[:256]

    try:
        if hasattr(SILERO_VAD_MODEL, "predict_proba"):
            return float(SILERO_VAD_MODEL.predict_proba(frame.reshape(1, -1))[0, 1])
        if hasattr(SILERO_VAD_MODEL, "__call__"):
            result = SILERO_VAD_MODEL(frame.reshape(1, -1))
            if isinstance(result, (tuple, list)):
                result = result[0]
            return float(np.asarray(result).reshape(-1)[0])
    except Exception:
        logger.warning("Silero VAD inference failed; falling back to webrtcvad.", exc_info=True)
        return _webrtc_vad_probability(audio_16k)
    return 0.0


def _vad_probability_from_mulaw_chunk(mulaw_bytes: bytes) -> float:
    if not mulaw_bytes:
        return 0.0

    # Twilio delivers 8kHz mu-law frames; convert to float32 PCM at 16kHz so both
    # the Silero model and the webrtcvad fallback see the expected sample rate.
    chunk_8k = mulaw_bytes_to_float32(mulaw_bytes, sample_rate=TELEPHONY_SR)
    if chunk_8k.size == 0:
        return 0.0

    chunk_16k = resample_audio(chunk_8k, TELEPHONY_SR, ASR_SR)
    if SILERO_VAD_MODEL is not None:
        return _silero_vad_probability(chunk_16k)
    return _webrtc_vad_probability(chunk_16k)


def _read_wav_float32(path: str | os.PathLike[str]) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Unsupported WAV format in {path}: sample width {sample_width} bytes is not PCM16.")
        frames = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _load_voicemail_audio() -> np.ndarray:
    wav_bytes = resources.files("maya_voice_os.telephony_service").joinpath("voicemail_message.wav").read_bytes()
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def build_demo_gap_plan(duration_ms: int, gap_every_ms: int = 1400, gap_length_ms: int = 300) -> list[dict[str, int]]:
    """Build a small set of intentional silent pauses that make interruption easy to demo."""
    if duration_ms <= 0:
        return []
    gaps: list[dict[str, int]] = []
    at_ms = gap_every_ms
    while at_ms < duration_ms:
        gaps.append({"at_ms": at_ms, "gap_ms": min(gap_length_ms, max(150, min(duration_ms - at_ms, gap_length_ms)))})
        at_ms += gap_every_ms
    return gaps


def _apply_demo_gaps(audio_16k: np.ndarray) -> np.ndarray:
    if not BARGE_IN_DEMO_MODE or audio_16k.size == 0:
        return audio_16k

    duration_ms = int((audio_16k.size / ASR_SR) * 1000.0)
    gaps = build_demo_gap_plan(duration_ms)
    if not gaps:
        return audio_16k

    segments: list[np.ndarray] = []
    current_index = 0
    for gap in gaps:
        trigger_sample = int((gap["at_ms"] / 1000.0) * ASR_SR)
        left = audio_16k[current_index:trigger_sample]
        if left.size:
            segments.append(left)
        gap_samples = int((gap["gap_ms"] / 1000.0) * ASR_SR)
        segments.append(np.zeros(gap_samples, dtype=np.float32))
        current_index = trigger_sample
    tail = audio_16k[current_index:]
    if tail.size:
        segments.append(tail)
    if not segments:
        return audio_16k
    return np.concatenate(segments)


def _should_escalate(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in ESCALATION_KEYWORDS)


async def _send_escalation(websocket: WebSocket) -> None:
    escalation_number = os.getenv("ESCALATION_PHONE_NUMBER", "+1234567890")
    logger.warning("Escalation trigger fired; forwarding call to human broker at %s", escalation_number)
    await websocket.send_text(
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Dial>{escalation_number}</Dial></Response>"
    )


async def _play_text_response(websocket: WebSocket, stream_sid: str, text: str) -> None:
    try:
        tts = get_tts_provider()
        audio = await tts.synthesize_pcm16(text, voice=os.getenv("TTS_VOICE", "en-US-JennyNeural"), sample_rate=ASR_SR)
        await _send_audio(websocket, stream_sid, audio)
    except Exception:
        logger.exception("Failed to synthesize canned confirmation response.")


async def _send_audio(websocket: WebSocket, stream_sid: str, audio_16k: np.ndarray) -> None:
    if BARGE_IN_DEMO_MODE:
        audio_16k = _apply_demo_gaps(audio_16k)

    audio_8k = resample_audio(audio_16k, ASR_SR, TELEPHONY_SR)
    mulaw_bytes = float32_to_mulaw_bytes(audio_8k)
    payload_b64 = base64.b64encode(mulaw_bytes).decode("ascii")
    await websocket.send_text(json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": payload_b64},
    }))


async def _fire_crm_action_webhook(conversation_id: str, transcript: str, action: str) -> None:
    crm_url = os.getenv("CRM_WEBHOOK_URL")
    if not crm_url:
        return
    payload = {"conversation_id": conversation_id, "transcript": transcript, "action": action}
    try:
        async with __import__("httpx").AsyncClient(timeout=10.0) as client:
            response = await client.post(crm_url, json=payload)
            logger.info("CRM action webhook sent: %s %s", response.status_code, response.text[:200])
    except Exception:
        logger.exception("CRM webhook action delivery failed.")


def _validate_twilio_signature(url: str, form_params: dict, signature: Optional[str]) -> bool:
    """Twilio's documented request-validation scheme: HMAC-SHA1 over the
    full URL + sorted form params, base64-encoded, compared to the
    X-Twilio-Signature header. Returns True if validation is disabled
    (no auth token configured) or the signature matches."""
    if not TWILIO_AUTH_TOKEN:
        return True
    if not signature:
        return False
    data = url + "".join(f"{k}{v}" for k, v in sorted(form_params.items()))
    computed = base64.b64encode(
        hmac.new(TWILIO_AUTH_TOKEN.encode("utf-8"), data.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")
    return hmac.compare_digest(computed, signature)


@router.post("/twiml")
async def twiml(request: Request, x_twilio_signature: Optional[str] = Header(default=None)):
    form = await request.form()
    form_dict = {k: v for k, v in form.items()}

    if not _validate_twilio_signature(str(request.url), form_dict, x_twilio_signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    host = request.headers.get("host", "localhost")
    stream_url = f"wss://{host}/twilio/stream"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""
    return PlainTextResponse(content=xml, media_type="text/xml")


@router.websocket("/stream")
async def twilio_media_stream(websocket: WebSocket):
    await websocket.accept()
    client = get_client()
    stream_sid = None
    conversation_id = None
    buffer: list[np.ndarray] = []
    silence_run = 0
    session_state = None
    response_started_at: Optional[float] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid", stream_sid)
                session_state = await session_manager.create_session(
                    from_number=msg["start"].get("customParameters", {}).get("from_number"),
                    to_number=None,
                    call_id=call_sid,
                )
                conversation_id = session_state.conversation_id
                call_started_at = time.monotonic()
                voicemail_check_started = False
                speech_detected_in_first_5s = False
                tone_frames = 0
                silence_after_tone_frames = 0
                is_voicemail = False
                logger.info(f"Twilio call started: {stream_sid}")

                greeting = await client.get_greeting_audio()
                if greeting and greeting.get("audio_base64"):
                    greeting_pcm16 = np.frombuffer(
                        base64.b64decode(greeting["audio_base64"]), dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    await _send_audio(websocket, stream_sid, greeting_pcm16)
                else:
                    logger.warning("Could not fetch greeting audio from orchestration-service; call proceeds silently until caller speaks.")

            elif event == "media":
                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                chunk = mulaw_bytes_to_float32(mulaw_bytes, sample_rate=TELEPHONY_SR)
                buffer.append(chunk)

                if not is_voicemail and time.monotonic() - call_started_at <= VOICEMAIL_CHECK_SECONDS:
                    voicemail_check_started = True
                    vad_probability = _vad_probability_from_mulaw_chunk(mulaw_bytes)
                    if vad_probability >= VAD_THRESHOLD:
                        speech_detected_in_first_5s = True
                    chunk_rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
                    if not speech_detected_in_first_5s:
                        if chunk_rms > 0.03:
                            tone_frames += 1
                            silence_after_tone_frames = 0
                        else:
                            if tone_frames >= VOICEMAIL_BEEP_FRAMES:
                                silence_after_tone_frames += 1
                            else:
                                silence_after_tone_frames = 0
                            tone_frames = 0
                        if tone_frames >= VOICEMAIL_BEEP_FRAMES and silence_after_tone_frames >= 10:
                            is_voicemail = True
                    if time.monotonic() - call_started_at >= VOICEMAIL_CHECK_SECONDS and not speech_detected_in_first_5s:
                        is_voicemail = True

                if is_voicemail:
                    logger.info("Voicemail detected, left pre-recorded message.")
                    voicemail_path = resources.files("maya_voice_os.telephony_service").joinpath("voicemail_message.wav")
                    try:
                        voicemail_audio = _load_voicemail_audio()
                        await _send_audio(websocket, stream_sid, voicemail_audio)
                    except Exception:
                        logger.exception("Failed to play bundled voicemail message.")
                    await websocket.close()
                    break

                # Replace RMS silence detection with VAD probability. A low-probability
                # streak indicates the user has stopped speaking and the turn is over.
                vad_probability = _vad_probability_from_mulaw_chunk(mulaw_bytes)
                silence_run = silence_run + 1 if vad_probability < VAD_THRESHOLD else 0

                if (
                    session_state is not None
                    and session_state.current_processing_task is not None
                    and not session_state.current_processing_task.done()
                    and vad_probability >= VAD_THRESHOLD
                ):
                    interrupted_at = time.monotonic() - (response_started_at or time.monotonic())
                    if BARGE_IN_DEMO_MODE:
                        logger.warning("DEMO MODE: Turn interrupted at %.1fs, cancelling TTS", interrupted_at)
                    else:
                        logger.info("Cancelling in-flight response because new user audio arrived.")
                    session_state.current_processing_task.cancel()
                    await websocket.send_text("<Response><Clear/></Response>")
                    buffer = []
                    silence_run = 0
                    continue

                total_samples = sum(c.size for c in buffer)
                has_enough_audio = total_samples > TELEPHONY_SR * 0.5

                if has_enough_audio and silence_run >= SILENCE_FRAMES_TO_END_TURN and conversation_id:
                    audio_8k = np.concatenate(buffer)
                    buffer = []
                    silence_run = 0
                    audio_16k = resample_audio(audio_8k, TELEPHONY_SR, ASR_SR)
                    pcm16_bytes = float32_to_pcm16_bytes(audio_16k)

                    if session_state is not None and session_state.awaiting_confirmation and not SAFE_MODE:
                        async with LOCAL_ASR_ENGINE.transcription_lock:
                            transcript_result = await asyncio.to_thread(LOCAL_ASR_ENGINE.transcribe, audio_16k, sample_rate=ASR_SR)
                        command = VOICE_COMMAND_ROUTER.classify_command(transcript_result.text)
                        logger.info("Confirmation command classified as '%s' from transcript: %r", command, transcript_result.text)

                        if command == "confirm":
                            await _play_text_response(websocket, stream_sid, "Thanks for confirming. I have noted that.")
                            await _fire_crm_action_webhook(session_state.conversation_id, transcript_result.text, "confirm")
                            session_state.awaiting_confirmation = False
                            session_state.confirmation_retries = 0
                            continue
                        if command == "deny":
                            await _play_text_response(websocket, stream_sid, "No problem. Thank you for your time.")
                            session_state.awaiting_confirmation = False
                            session_state.confirmation_retries = 0
                            continue
                        if command == "escalate":
                            await _send_escalation(websocket)
                            session_state.awaiting_confirmation = False
                            session_state.confirmation_retries = 0
                            break
                        if command == "unclear":
                            session_state.confirmation_retries += 1
                            if session_state.confirmation_retries <= 2:
                                await _play_text_response(
                                    websocket,
                                    stream_sid,
                                    "I didn't catch that. Please say yes, no, or tell me more.",
                                )
                                session_state.awaiting_confirmation = True
                                continue
                            session_state.awaiting_confirmation = False
                            session_state.confirmation_retries = 0
                    elif session_state is not None and session_state.awaiting_confirmation and SAFE_MODE:
                        logger.info("SAFE MODE: skipping complex confirmation routing; asking for a simple, explicit response.")
                        await _play_text_response(websocket, stream_sid, "I’m not fully sure, so I’d rather be cautious. Please say yes or no clearly.")
                        session_state.awaiting_confirmation = True
                        continue

                    if session_state is not None:
                        response_started_at = time.monotonic()
                        task = asyncio.create_task(client.process_audio(pcm16_bytes, conversation_id=conversation_id))
                        session_state.current_processing_task = task
                        try:
                            result = await task
                        except asyncio.CancelledError:
                            logger.info("In-flight response processing cancelled by new user audio.")
                            session_state.current_processing_task = None
                            continue
                        finally:
                            if session_state.current_processing_task is task:
                                session_state.current_processing_task = None

                        reply_text = result.get("llm_response")
                        transcript_text = result.get("transcript")
                        audio_b64 = result.get("audio_base64")

                        if session_state is not None and reply_text and (
                            "would you like to" in reply_text.lower() or "shall i" in reply_text.lower()
                        ):
                            session_state.awaiting_confirmation = True
                            session_state.confirmation_retries = 0
                            logger.info("Awaiting confirmation response from caller.")

                        combined_text = " ".join(part for part in [transcript_text, reply_text] if part)
                        if _should_escalate(combined_text):
                            await _send_escalation(websocket)
                            break

                        session_state.record_exchange(
                            transcript=transcript_text, response=reply_text, audio_url=None
                        )
                        await session_manager.upsert(session_state)

                        if audio_b64 and reply_text:
                            reply_pcm16 = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16).astype(np.float32) / 32768.0
                            await _send_audio(websocket, stream_sid, reply_pcm16)

            elif event == "stop":
                logger.info(f"Twilio call ended: {stream_sid}")
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected.")


