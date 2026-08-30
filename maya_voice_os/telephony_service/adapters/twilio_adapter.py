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

import base64
import hashlib
import hmac
import json
import logging
import os
from typing import Optional
from urllib.parse import urlencode

import numpy as np
from fastapi import APIRouter, Form, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from maya_voice_os.telephony_service.orchestration_client import get_client
from maya_voice_os.telephony_service.session_manager import session_manager
from maya_voice_os.shared.audio_utils import float32_to_mulaw_bytes, float32_to_pcm16_bytes, mulaw_bytes_to_float32, resample_audio

logger = logging.getLogger("twilio-adapter")
router = APIRouter(prefix="/twilio", tags=["twilio"])

TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN") or None
PUBLIC_BASE_URL = os.getenv("TELEPHONY_PUBLIC_BASE_URL", "http://localhost:8100")

TELEPHONY_SR = 8000
ASR_SR = 16000
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_FRAMES_TO_END_TURN = 20  # ~20 * 20ms frames ≈ 400ms of quiet ends a turn


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

                rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
                silence_run = silence_run + 1 if rms < SILENCE_RMS_THRESHOLD else 0

                total_samples = sum(c.size for c in buffer)
                has_enough_audio = total_samples > TELEPHONY_SR * 0.5

                if has_enough_audio and silence_run >= SILENCE_FRAMES_TO_END_TURN and conversation_id:
                    audio_8k = np.concatenate(buffer)
                    buffer = []
                    silence_run = 0
                    audio_16k = resample_audio(audio_8k, TELEPHONY_SR, ASR_SR)
                    pcm16_bytes = float32_to_pcm16_bytes(audio_16k)

                    result = await client.process_audio(pcm16_bytes, conversation_id=conversation_id)
                    reply_text = result.get("llm_response")
                    audio_b64 = result.get("audio_base64")

                    if session_state:
                        session_state.record_exchange(
                            transcript=result.get("transcript"), response=reply_text, audio_url=None
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


async def _send_audio(websocket: WebSocket, stream_sid: str, audio_16k: np.ndarray) -> None:
    audio_8k = resample_audio(audio_16k, ASR_SR, TELEPHONY_SR)
    mulaw_bytes = float32_to_mulaw_bytes(audio_8k)
    payload_b64 = base64.b64encode(mulaw_bytes).decode("ascii")
    await websocket.send_text(json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": payload_b64},
    }))
