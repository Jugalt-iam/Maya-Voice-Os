"""
Orchestration-service HTTP API.

Exposes:
    POST /process   - the single endpoint telephony-service (or anything
                        else) calls: send audio or text in, get a spoken
                        reply out. Kept API-shape-compatible with the
                        original project's /process contract
                        (transcript / llm_response / audio_url / ... fields)
                        so existing integrations don't need to change their
                        parsing — but this service is fully standalone and
                        makes NO calls to any other project or external
                        infrastructure.
    GET  /health    - liveness check, no auth required.

Security:
    - If ORCHESTRATION_API_TOKEN is set in .env, every /process call must
      include it as `Authorization: Bearer <token>`. Unset = open (fine for
      local/dev use, NOT recommended if this port is exposed to the internet).
    - Base64 audio payloads are size-capped to avoid trivial memory-exhaustion
      abuse.
    - No eval/exec/pickle anywhere; YAML is loaded with yaml.safe_load only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from maya_voice_os.orchestration_service.pipeline import ConversationState, VoicePipeline
from maya_voice_os.shared.audio_utils import pcm16_bytes_to_float32, float32_to_pcm16_bytes
from maya_voice_os.orchestration_service.streaming_llm import TokenBuffer, stream_chat_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("orchestration-server")

API_TOKEN = os.getenv("ORCHESTRATION_API_TOKEN") or None
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB decoded cap, ample for a single utterance

app = FastAPI(title="orchestration-service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ORCHESTRATION_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[VoicePipeline] = None
_conversations: Dict[str, ConversationState] = {}
_lock = asyncio.Lock()


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    logger.info("Loading voice pipeline (ASR model, playbooks, TTS provider)...")
    pipeline = VoicePipeline()
    await pipeline.async_init()  # wires up Redis for smart_memory, context_manager
    logger.info(f"Ready. Identity: {pipeline.identity.name}")


def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    if API_TOKEN is None:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing orchestration token")


async def _get_state(conversation_id: str) -> ConversationState:
    async with _lock:
        if conversation_id not in _conversations:
            _conversations[conversation_id] = ConversationState(conversation_id=conversation_id)
        return _conversations[conversation_id]


class ProcessRequest(BaseModel):
    conversation_id: str
    user_id: Optional[str] = "caller"
    language: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    audio_data: Optional[str] = None   # base64 PCM16 mono 16kHz, OR...
    text_input: Optional[str] = None   # ...plain text, skips ASR entirely


class ProcessResponse(BaseModel):
    conversation_id: str
    transcript: Optional[str] = None
    llm_response: Optional[str] = None
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None  # PCM16 mono 16kHz, base64-encoded
    processing_stages: Dict[str, Any] = {}
    total_processing_time: float = 0.0
    expert_used: Optional[str] = None
    confidence: Optional[float] = None
    context_updates: Dict[str, Any] = {}
    reasoning_chain: list = []


class IdentityResponse(BaseModel):
    name: str
    greeting: str
    voice: str


class GreetingResponse(BaseModel):
    greeting_text: str
    audio_base64: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int
    quality_rating: Optional[int] = None
    relevance_rating: Optional[int] = None
    satisfaction: Optional[int] = None
    suggestions: Optional[str] = None


@app.post("/feedback")
async def feedback(request: FeedbackRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, str]:
    _require_token(authorization)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading, try again shortly")
    route = pipeline.last_route_by_conversation.get(request.conversation_id, "unknown")
    pipeline.continual_learning.record_feedback(request.conversation_id, {
        "rating": request.rating,
        "route_used": route,
        "quality_rating": request.quality_rating or 0,
        "relevance_rating": request.relevance_rating or 0,
        "satisfaction": request.satisfaction or 0,
        "suggestions": request.suggestions or "",
    })
    return {"status": "recorded", "route_used": route}


@app.get("/stats")
async def stats(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_token(authorization)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading, try again shortly")
    return {
        "routing": pipeline.stats.to_dict(),
        "learning": pipeline.continual_learning.get_learning_statistics(),
    }


@app.get("/identity", response_model=IdentityResponse)
async def identity() -> IdentityResponse:
    """Lets telephony adapters (or anything else) know the bot's name,
    greeting text, and TTS voice, without duplicating identity/*.yaml
    content into telephony-service."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading, try again shortly")
    return IdentityResponse(
        name=pipeline.identity.name,
        greeting=pipeline.identity.greeting,
        voice=pipeline.identity.voice,
    )


@app.post("/greeting", response_model=GreetingResponse)
async def greeting(authorization: Optional[str] = Header(default=None)) -> GreetingResponse:
    """Synthesizes the identity's configured greeting directly — bypasses
    fast-path/LLM entirely, so the greeting is always exactly what's in
    identity/*.yaml, not something an LLM improvised in response to it."""
    _require_token(authorization)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading, try again shortly")

    audio = await pipeline.tts.synthesize_pcm16(
        pipeline.identity.greeting, voice=pipeline.identity.voice, sample_rate=16000
    )
    audio_b64 = base64.b64encode(float32_to_pcm16_bytes(audio)).decode("ascii")
    return GreetingResponse(greeting_text=pipeline.identity.greeting, audio_base64=audio_b64)


@app.get("/health")
async def health() -> Dict[str, str]:
    if pipeline is None:
        return {"status": "loading"}
    return {"status": "ok", "identity": pipeline.identity.name}


@app.post("/process", response_model=ProcessResponse, dependencies=[])
async def process(request: ProcessRequest, authorization: Optional[str] = Header(default=None)) -> ProcessResponse:
    _require_token(authorization)

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading, try again shortly")

    start = time.time()
    state = await _get_state(request.conversation_id)

    if request.text_input:
        t0 = time.time()
        reply_text = await pipeline.respond(request.text_input, state)
        stage_process_ms = (time.time() - t0) * 1000

        t0 = time.time()
        reply_audio = await pipeline.tts.synthesize_pcm16(reply_text, voice=pipeline.identity.voice, sample_rate=16000)
        stage_tts_ms = (time.time() - t0) * 1000

        audio_b64 = base64.b64encode(float32_to_pcm16_bytes(reply_audio)).decode("ascii")
        return ProcessResponse(
            conversation_id=request.conversation_id,
            transcript=request.text_input,
            llm_response=reply_text,
            audio_base64=audio_b64,
            processing_stages={"process": {"duration_ms": stage_process_ms}, "act": {"duration_ms": stage_tts_ms}},
            total_processing_time=time.time() - start,
        )

    if not request.audio_data:
        raise HTTPException(status_code=400, detail="Provide either audio_data or text_input")

    try:
        raw = base64.b64decode(request.audio_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio_data") from exc

    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio payload too large")

    audio_np = pcm16_bytes_to_float32(raw)

    t0 = time.time()
    reply_text, transcript, reply_audio, stage_timings = await pipeline.handle_utterance(audio_np, 16000, state)
    total_ms = (time.time() - t0) * 1000

    audio_b64 = base64.b64encode(float32_to_pcm16_bytes(reply_audio)).decode("ascii")

    return ProcessResponse(
        conversation_id=request.conversation_id,
        transcript=transcript,
        llm_response=reply_text,
        audio_base64=audio_b64,
        processing_stages={
            "asr": {"duration_ms": stage_timings["asr_ms"]},
            "respond": {"duration_ms": stage_timings["respond_ms"]},
            "tts": {"duration_ms": stage_timings["tts_ms"]},
            "total": {"duration_ms": round(total_ms, 1)},
        },
        total_processing_time=time.time() - start,
    )


@app.websocket("/process/stream")
async def process_stream(websocket: WebSocket):
    """
    Real token-level streaming: text in, reply streamed back as it's
    generated (text chunks + synthesized audio chunks), rather than
    waiting for the full reply like /process does.

    Protocol (JSON messages both ways):
        client -> {"conversation_id": "...", "text_input": "..."}
        server -> {"type": "text_chunk", "text": "..."}         (repeated)
        server -> {"type": "audio_chunk", "audio_base64": "..."} (repeated,
                    PCM16 mono 16kHz, one per flushed text chunk)
        server -> {"type": "done", "full_text": "..."}
        server -> {"type": "error", "message": "..."}

    Bypasses homemath for the LLM call (see streaming_llm.py's docstring
    for why); falls back to the normal blocking llm_router.chat() for this
    turn if the direct streaming call fails, so a caller still gets a
    reply even when streaming itself doesn't work.

    Audio-in / ASR is intentionally out of scope here — this endpoint is
    text-in only. Streaming benefits the LLM+TTS latency, not ASR, and
    keeping this endpoint text-only keeps the bypass-homemath surface
    area small and easy to reason about.
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        conversation_id = msg.get("conversation_id", "stream-default")
        text_input = (msg.get("text_input") or "").strip()

        if pipeline is None:
            await websocket.send_text(json.dumps({"type": "error", "message": "Pipeline still loading"}))
            await websocket.close()
            return
        if not text_input:
            await websocket.send_text(json.dumps({"type": "error", "message": "text_input required"}))
            await websocket.close()
            return

        state = await _get_state(conversation_id)

        # Fast-path playbooks: a cached one-liner gets no benefit from
        # streaming, so just answer directly if it matches.
        fast = pipeline.fast_router.match(text_input)
        if fast is not None:
            state.add("user", text_input)
            state.add("assistant", fast.response_text)
            await websocket.send_text(json.dumps({"type": "text_chunk", "text": fast.response_text}))
            audio = await pipeline.tts.synthesize_pcm16(fast.response_text, voice=pipeline.identity.voice, sample_rate=16000)
            audio_b64 = base64.b64encode(float32_to_pcm16_bytes(audio)).decode("ascii")
            await websocket.send_text(json.dumps({"type": "audio_chunk", "audio_base64": audio_b64}))
            await websocket.send_text(json.dumps({"type": "done", "full_text": fast.response_text}))
            return

        state.add("user", text_input)
        messages = [{"role": "system", "content": pipeline.identity.system_prompt}, *state.history]

        full_text = ""
        streamed_ok = False
        try:
            provider = pipeline.llm_router.providers[0] if pipeline.llm_router.providers else None
            if provider is None:
                raise RuntimeError("No LLM providers configured")

            buffer = TokenBuffer()
            async for token in stream_chat_completion(provider.host, provider.model, provider.api_key, messages):
                full_text += token
                flushed = buffer.add(token)
                if flushed:
                    await websocket.send_text(json.dumps({"type": "text_chunk", "text": flushed}))
                    audio = await pipeline.tts.synthesize_pcm16(flushed, voice=pipeline.identity.voice, sample_rate=16000)
                    audio_b64 = base64.b64encode(float32_to_pcm16_bytes(audio)).decode("ascii")
                    await websocket.send_text(json.dumps({"type": "audio_chunk", "audio_base64": audio_b64}))

            remaining = buffer.drain()
            if remaining:
                full_text_piece = remaining
                await websocket.send_text(json.dumps({"type": "text_chunk", "text": full_text_piece}))
                audio = await pipeline.tts.synthesize_pcm16(full_text_piece, voice=pipeline.identity.voice, sample_rate=16000)
                audio_b64 = base64.b64encode(float32_to_pcm16_bytes(audio)).decode("ascii")
                await websocket.send_text(json.dumps({"type": "audio_chunk", "audio_base64": audio_b64}))

            streamed_ok = bool(full_text.strip())
        except Exception as e:
            logger.warning(f"Direct streaming failed ({e}), falling back to non-streaming llm_router.chat()")

        if not streamed_ok:
            full_text = pipeline.llm_router.chat(state.history, system_prompt=pipeline.identity.system_prompt)
            await websocket.send_text(json.dumps({"type": "text_chunk", "text": full_text}))
            audio = await pipeline.tts.synthesize_pcm16(full_text, voice=pipeline.identity.voice, sample_rate=16000)
            audio_b64 = base64.b64encode(float32_to_pcm16_bytes(audio)).decode("ascii")
            await websocket.send_text(json.dumps({"type": "audio_chunk", "audio_base64": audio_b64}))

        state.add("assistant", full_text)
        await websocket.send_text(json.dumps({"type": "done", "full_text": full_text}))

    except WebSocketDisconnect:
        logger.info("process/stream client disconnected")
    except Exception as e:
        logger.error(f"process/stream error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# Mounted last, so it never shadows the explicit API routes above --
# StaticFiles only serves paths that don't match an already-registered
# route. This is what makes `python run_orchestrator.py` + opening
# http://localhost:8004 in a browser work with no separate server: the
# same FastAPI app serves both the API and the local test UI.
_UI_DIR = Path(__file__).resolve().parent / "ui"
if _UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("ORCHESTRATION_HOST", "0.0.0.0"),
        port=int(os.getenv("ORCHESTRATION_PORT", "8004")),
    )
