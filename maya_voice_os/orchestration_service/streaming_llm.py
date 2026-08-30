"""
Real token-level streaming for the LLM reply.

homemath (used everywhere else via llm-service/llm_router.py) is
synchronous and blocking by design — it streams internally from the
provider but only returns one complete string when done, never partial
tokens to the caller. That's fine for /process, but genuinely defeats the
purpose of "streaming" for lower perceived latency.

So this module bypasses homemath for this one path only: it calls the
provider's OpenAI-compatible streaming chat-completions endpoint directly
(Groq, OpenRouter, and Ollama all speak the same SSE format with
`stream: true`), and yields tokens as they arrive.

Trade-off, stated plainly: this path does NOT get homemath's task
classification, <think>-stripping, or friendly-failure-message behavior.
On any failure it raises, and the caller (server.py's /process/stream) is
expected to fall back to the normal blocking llm_router.chat() for that
turn, so a caller never gets silence — they just lose the streaming
benefit for that one turn.

TokenBuffer (ported near-verbatim from realtime_streamer.py) decides when
enough tokens have accumulated to flush as a chunk worth synthesizing to
speech, rather than speaking one word at a time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from maya_voice_os.shared.retry import RETRYABLE_EXCEPTIONS, RETRYABLE_STATUS_CODES, RetryConfig

logger = logging.getLogger("streaming-llm")

SENTENCE_ENDERS = re.compile(r'[.!?]')
CLAUSE_ENDERS = re.compile(r'[,;:\-]')
MIN_FLUSH_WORDS = 4
MAX_FLUSH_WORDS = 12


class TokenBuffer:
    """Accumulates LLM tokens and flushes at natural speech boundaries,
    so TTS synthesizes phrase-sized chunks rather than single words."""

    def __init__(self):
        self.buffer = ""
        self.word_count = 0

    def add(self, token: str) -> Optional[str]:
        self.buffer += token
        self.word_count = len(self.buffer.split())

        if SENTENCE_ENDERS.search(token) and self.word_count >= MIN_FLUSH_WORDS:
            return self._flush()
        if CLAUSE_ENDERS.search(token) and self.word_count >= MAX_FLUSH_WORDS // 2:
            return self._flush()
        if self.word_count >= MAX_FLUSH_WORDS:
            return self._flush()
        return None

    def _flush(self) -> str:
        text = self.buffer.strip()
        self.buffer = ""
        self.word_count = 0
        return text

    def drain(self) -> Optional[str]:
        if self.buffer.strip():
            return self._flush()
        return None


async def stream_chat_completion(
    host: str,
    model: str,
    api_key: Optional[str],
    messages: List[Dict[str, str]],
    timeout: float = 30.0,
    retry_config: Optional[RetryConfig] = None,
) -> AsyncGenerator[str, None]:
    """
    Streams token deltas from an OpenAI-compatible /v1/chat/completions
    endpoint. `host` should be the bare provider host WITHOUT /v1 (same
    convention as llm_router.py's ProviderConfig.host) — this function
    appends /v1/chat/completions itself.

    Retries transient connection failures (timeouts, connection resets) a
    few times with backoff — but ONLY before any token has been yielded.
    Once tokens have started reaching the caller, a failure is raised
    immediately rather than retried, since restarting the request at that
    point would re-send already-delivered tokens and produce a duplicated/
    garbled reply. The caller (server.py's /process/stream) already has
    its own fallback to the non-streaming path for a failure at any point.
    """
    url = f"{host.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "messages": messages, "stream": True}
    retry_config = retry_config or RetryConfig(max_retries=2, initial_delay=0.5, max_delay=4.0)

    attempt = 0
    while True:
        any_token_yielded = False
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code in RETRYABLE_STATUS_CODES and attempt < retry_config.max_retries:
                        delay = retry_config.get_delay(attempt)
                        logger.warning(
                            f"Streaming call got retryable HTTP {response.status_code} "
                            f"(attempt {attempt + 1}/{retry_config.max_retries + 1}), retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    if response.status_code != 200:
                        body = await response.aread()
                        raise RuntimeError(f"Streaming call to {url} failed: {response.status_code} {body[:200]!r}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        token = delta.get("content")
                        if token:
                            any_token_yielded = True
                            yield token
            return  # stream completed without a [DONE] sentinel; treat as done

        except RETRYABLE_EXCEPTIONS as e:
            if any_token_yielded or attempt >= retry_config.max_retries:
                raise
            delay = retry_config.get_delay(attempt)
            logger.warning(
                f"Streaming connect attempt {attempt + 1}/{retry_config.max_retries + 1} "
                f"failed before any token arrived ({type(e).__name__}: {e}), retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)
            attempt += 1
            continue
