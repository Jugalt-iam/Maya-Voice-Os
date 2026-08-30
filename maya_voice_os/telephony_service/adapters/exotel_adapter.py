"""
Exotel-style adapter — a working TEMPLATE for turn-based ("record an
utterance, POST it, get a reply") telephony providers, as opposed to
Twilio's real-time bidirectional streaming.

This is the pattern most non-Twilio telephony/IVR APIs use: the provider
calls a webhook per turn with the caller's recorded audio (or a transcript,
depending on the provider), you return the reply, and the provider plays it.
Adjust `_extract_audio_from_request` / the XML response shape for whatever
provider you're actually integrating — the request/response bodies vary
provider to provider, but the call into orchestration-service (via
orchestration_client) stays the same for all of them.

Inbound provider webhooks can't easily send a custom Authorization header,
so this endpoint is protected instead by an unguessable path segment
(EXOTEL_WEBHOOK_SECRET) rather than a bearer token — put that secret in the
webhook URL you give the provider, and keep it out of version control.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from maya_voice_os.telephony_service.orchestration_client import get_client
from maya_voice_os.telephony_service.session_manager import session_manager

logger = logging.getLogger("exotel-adapter")
router = APIRouter(prefix="/exotel", tags=["exotel"])

WEBHOOK_SECRET = os.getenv("EXOTEL_WEBHOOK_SECRET") or None
MGMT_TOKEN = os.getenv("TELEPHONY_GATEWAY_TOKEN") or None
PUBLIC_BASE_URL = os.getenv("TELEPHONY_PUBLIC_BASE_URL", "http://localhost:8100")


def _require_webhook_secret(secret: str) -> None:
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")


def _require_mgmt_token(x_telephony_token: Optional[str] = Header(default=None)) -> None:
    if MGMT_TOKEN and x_telephony_token != MGMT_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid gateway token")


class ExotelIncomingPayload(BaseModel):
    CallSid: Optional[str] = None
    From: Optional[str] = None
    To: Optional[str] = None


@router.post("/incoming/{secret}")
async def exotel_incoming(secret: str, payload: ExotelIncomingPayload) -> Response:
    """Called when a call comes in. Returns XML telling the provider what
    to do next (here: greet, then gather speech and POST it to /turn)."""
    _require_webhook_secret(secret)

    state = await session_manager.create_session(
        from_number=payload.From, to_number=payload.To, call_id=payload.CallSid
    )
    turn_url = f"{PUBLIC_BASE_URL.rstrip('/')}/exotel/turn/{secret}/{state.session_id}"

    identity = await get_client().get_identity()
    greeting_text = identity.get("greeting", "Hello!")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather action="{turn_url}" method="POST" input="speech" timeout="5" language="en-IN">
    <Say voice="woman" language="en-IN">{greeting_text}</Say>
  </Gather>
</Response>""".strip()
    return Response(content=xml, media_type="application/xml")


class ExotelTurnPayload(BaseModel):
    # Adjust these field names to match your actual provider's callback
    # shape. Some providers send a transcribed SpeechResult; others send a
    # recording URL you must fetch; a few send raw audio inline.
    SpeechResult: Optional[str] = None
    RecordingUrl: Optional[str] = None


@router.post("/turn/{secret}/{session_id}")
async def exotel_turn(secret: str, session_id: str, payload: ExotelTurnPayload) -> Response:
    _require_webhook_secret(secret)

    state = await session_manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown session")

    client = get_client()

    if payload.SpeechResult:
        result = await client.process_text(payload.SpeechResult, conversation_id=state.conversation_id)
    elif payload.RecordingUrl:
        # TEMPLATE: fetch the recording and decode to PCM16 here, then call
        # client.process_audio(pcm16_bytes, conversation_id=state.conversation_id).
        # Left unimplemented since the audio format varies by provider.
        raise HTTPException(status_code=501, detail="RecordingUrl handling not implemented for this provider — see comment in exotel_adapter.py")
    else:
        raise HTTPException(status_code=400, detail="No SpeechResult or RecordingUrl in payload")

    reply_text = result.get("llm_response") or "Sorry, I didn't catch that."
    state.record_exchange(transcript=result.get("transcript"), response=reply_text, audio_url=None)
    await session_manager.upsert(state)

    turn_url = f"{PUBLIC_BASE_URL.rstrip('/')}/exotel/turn/{secret}/{session_id}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather action="{turn_url}" method="POST" input="speech" timeout="5" language="en-IN">
    <Say voice="woman" language="en-IN">{reply_text}</Say>
  </Gather>
</Response>""".strip()
    return Response(content=xml, media_type="application/xml")


@router.post("/callback/{secret}/{session_id}")
async def exotel_callback(secret: str, session_id: str) -> dict:
    _require_webhook_secret(secret)
    state = await session_manager.get(session_id)
    if state:
        state.status = "completed"
        await session_manager.upsert(state)
    return {"status": "ack"}


@router.get("/sessions/{session_id}", dependencies=[])
async def get_session(session_id: str, x_telephony_token: Optional[str] = Header(default=None)) -> dict:
    _require_mgmt_token(x_telephony_token)
    state = await session_manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown session")
    return {
        "session_id": state.session_id,
        "conversation_id": state.conversation_id,
        "status": state.status,
        "exchanges": [e.__dict__ for e in state.exchanges],
    }
