"""
Client telephony-service uses to reach THIS repo's own orchestration-service
over HTTP. Points only at ORCHESTRATION_URL (defaults to localhost) — there
is no reference anywhere to any external/private project's infrastructure.

Failure handling: a BOUNDED number of retries (2, short backoff) for
transient failures (connection blips, 502/503/504/429) via shared/retry.py
— not an indefinite retry loop. If still failing after that, returns a
clear "service unavailable" result rather than fabricating a fake
successful transcript/response.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from maya_voice_os.shared.retry import RetryConfig, RetryExhausted, retry_with_backoff

logger = logging.getLogger("orchestration-client")

_RETRY_CONFIG = RetryConfig(max_retries=2, initial_delay=0.5, max_delay=4.0)


@dataclass
class OrchestrationClient:
    base_url: str
    api_token: Optional[str] = None
    timeout_seconds: float = 30.0

    async def process_audio(
        self,
        audio_pcm16_bytes: bytes,
        *,
        conversation_id: str,
        user_id: str = "caller",
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "language": language,
            "context": context or {},
            "audio_data": base64.b64encode(audio_pcm16_bytes).decode("ascii"),
        }
        return await self._post("/process", payload)

    async def process_text(
        self,
        text: str,
        *,
        conversation_id: str,
        user_id: str = "caller",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "context": context or {},
            "text_input": text,
        }
        return await self._post("/process", payload)

    async def get_identity(self) -> Dict[str, Any]:
        """Fetch {name, greeting, voice} so telephony adapters stay in sync
        with identity/*.yaml without duplicating it."""
        url = f"{self.base_url.rstrip('/')}/identity"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=10.0)) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.error(f"orchestration-service /identity call failed ({exc}).")
            return {"name": "Assistant", "greeting": "Hello!", "voice": ""}

    async def get_greeting_audio(self) -> Optional[Dict[str, Any]]:
        """Fetch the identity's configured greeting already synthesized to
        audio, for providers (like Twilio Media Streams) that need actual
        audio bytes rather than text a provider-side <Say> can read.
        Returns None on failure so callers can decide how to degrade."""
        url = f"{self.base_url.rstrip('/')}/greeting"
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=10.0)) as client:
                response = await client.post(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            logger.error(f"orchestration-service /greeting call failed ({exc}); call will proceed without a spoken greeting.")
            return None

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}

        async def attempt():
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=10.0)) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()  # retry_with_backoff classifies retryable vs non-retryable statuses
                return response.json()

        try:
            return await retry_with_backoff(attempt, _RETRY_CONFIG)
        except (RetryExhausted, httpx.HTTPError) as exc:
            logger.error(f"orchestration-service call failed ({exc}); returning unavailable result.")
            return {
                "conversation_id": payload.get("conversation_id"),
                "transcript": None,
                "llm_response": "Sorry, the assistant is temporarily unavailable. Please try again shortly.",
                "audio_base64": None,
                "error": "orchestration_unavailable",
            }


_client: Optional[OrchestrationClient] = None


def get_client() -> OrchestrationClient:
    global _client
    if _client is None:
        base_url = os.getenv("ORCHESTRATION_URL", "http://127.0.0.1:8004")
        api_token = os.getenv("ORCHESTRATION_API_TOKEN") or None
        _client = OrchestrationClient(base_url=base_url, api_token=api_token)
        logger.info(f"Orchestration client targeting {base_url}")
    return _client
