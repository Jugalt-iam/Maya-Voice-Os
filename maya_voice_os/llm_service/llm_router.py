"""
LLM router — the "full homemath logic, multiple free models, plus Ollama"
layer.

homemath itself (https://github.com/Jugalt-iam/homemath) is a smart client
for a SINGLE OpenAI-compatible endpoint: it streams, classifies the task,
strips <think> blocks, and fails over between exactly two configured slots
(LLM_HOST / LLM_HOST_FALLBACK). To get routing across an arbitrary list of
FREE providers (Groq -> OpenRouter free models -> local Ollama), this module
wraps homemath in an outer loop: for each provider in priority order it
points homemath at that provider (via env vars) and calls it; on any
failure, timeout, or empty answer it moves to the next provider. Local
Ollama is always last, so the bot never hard-blocks even with zero API keys
configured, as long as Ollama is running.

Every provider used here has a free tier / is fully local — no paid API is
ever required.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger("llm_router")


@dataclass
class ProviderConfig:
    name: str
    host: str            # base URL, must expose /v1/chat/completions
    api_key: Optional[str]
    model: str


def _build_provider_configs() -> List[ProviderConfig]:
    """Reads .env-configured providers and returns only the ones that are
    actually usable (have a key, or are the always-available local Ollama)."""
    order = [p.strip() for p in os.getenv("LLM_PROVIDER_ORDER", "groq,cerebras,mistral,openrouter,ollama").split(",") if p.strip()]

    all_providers = {
        "groq": ProviderConfig(
            name="groq",
            # NOTE: no trailing /v1 here — homemath appends /v1/chat/completions
            # to LLM_HOST itself. Including /v1 in both places produced a
            # doubled .../v1/v1/chat/completions path that 404'd every call.
            host="https://api.groq.com/openai",
            api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        ),
        "cerebras": ProviderConfig(
            name="cerebras",
            host="https://api.cerebras.ai",
            api_key=os.getenv("CEREBRAS_API_KEY") or None,
            model=os.getenv("CEREBRAS_MODEL", "llama3.1-8b"),
        ),
        "mistral": ProviderConfig(
            name="mistral",
            host="https://api.mistral.ai",
            api_key=os.getenv("MISTRAL_API_KEY") or None,
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        ),
        "openrouter": ProviderConfig(
            name="openrouter",
            host="https://openrouter.ai/api",
            api_key=os.getenv("OPENROUTER_API_KEY") or None,
            model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        ),
        "ollama": ProviderConfig(
            name="ollama",
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            api_key=None,  # local, no key needed
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ),
    }

    configs = []
    for name in order:
        cfg = all_providers.get(name)
        if cfg is None:
            logger.warning(f"Unknown provider '{name}' in LLM_PROVIDER_ORDER, skipping.")
            continue
        # Cloud providers need a key; Ollama is always attempted (it fails
        # fast and harmlessly if not running).
        if cfg.name != "ollama" and not cfg.api_key:
            logger.info(f"Provider '{name}' has no API key set — skipping.")
            continue
        configs.append(cfg)

    if not configs:
        logger.warning(
            "No LLM providers configured/available (no API keys set, and "
            "Ollama not in LLM_PROVIDER_ORDER). The bot will fall back to "
            "playbook-only responses."
        )
    return configs


# Text homemath itself falls back to when every provider IT tried
# internally failed. homemath_chat() returns this as a normal non-empty
# string rather than raising or returning empty, so without this check our
# loop below would mistake homemath's own "sorry, I'm unreachable" text for
# a real successful model reply and stop trying further providers.
#
# NOTE: this is a best-effort heuristic keyed on the specific phrasing
# observed from homemath's fallback, not a documented/guaranteed API
# contract. If homemath's fallback wording changes, or a real model
# response ever happens to contain this exact phrase, this check can be
# wrong in either direction — kept deliberately narrow/specific (rather
# than a generic phrase like "having trouble") to minimize false positives
# against genuine replies.
_HOMEMATH_FAILURE_MARKERS = (
    "can't reach that part of my mind",
)


def _looks_like_homemath_failure_text(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _HOMEMATH_FAILURE_MARKERS)


class LLMRouter:
    def __init__(self):
        self.providers = _build_provider_configs()
        self.timeout = float(os.getenv("LLM_TIMEOUT", "25"))
        self._circuit_breaker: dict[str, dict] = {}
        # Set by chat() to whichever provider actually answered the most
        # recent call (or None if every provider failed). Used for feedback
        # attribution (continual_learning) and stats (route tracking) —
        # read this right after calling chat(), not concurrently from
        # another call, since it's plain instance state, not per-call.
        self.last_provider_used: Optional[str] = None
        self.last_route_trace: Optional[dict[str, Any]] = None
        self.route_history: list[dict[str, Any]] = []
        self.safe_mode = os.getenv("SAFE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    def _record_route_trace(self, attempts: list[dict[str, Any]], selected_provider: Optional[str], fallback_reason: str) -> None:
        primary_provider = attempts[0]["provider"] if attempts else None
        trace = {
            "primary_provider": primary_provider,
            "selected_provider": selected_provider,
            "success": selected_provider is not None,
            "fallback_count": max(0, len(attempts) - 1),
            "reason": fallback_reason,
            "attempts": attempts,
        }
        self.last_route_trace = trace
        self.route_history.append(trace)
        if len(self.route_history) > 50:
            self.route_history = self.route_history[-50:]

        if not attempts:
            logger.info("LLM routing: no providers available")
            return

        first_attempt = attempts[0]
        if selected_provider and selected_provider != primary_provider:
            logger.info(
                "LLM routing: tried %s → %s (%0.1fs), fallback to %s → success (%0.1fs)",
                first_attempt["provider"],
                first_attempt["reason"],
                first_attempt["elapsed_ms"] / 1000.0,
                selected_provider,
                attempts[-1]["elapsed_ms"] / 1000.0,
            )
        elif selected_provider == primary_provider:
            logger.info(
                "LLM routing: tried %s → success (%0.1fs)",
                primary_provider,
                attempts[-1]["elapsed_ms"] / 1000.0,
            )
        else:
            final_attempt = attempts[-1]
            logger.warning(
                "LLM routing: tried %s → %s (%0.1fs), all providers failed",
                primary_provider,
                final_attempt["reason"],
                final_attempt["elapsed_ms"] / 1000.0,
            )

    def chat(self, messages: list[dict], system_prompt: Optional[str] = None) -> str:
        """
        Tries each configured free provider in order using homemath's
        low-level ollama_chat_stream_dual(url, payload, headers) — this is
        the CORRECT way to use homemath across more than 2 providers.

        homemath's high-level engine (homemath_chat / HomemathEngine) is
        hard-limited to exactly 2 fixed slots by design ("Provider
        selection is two fixed slots, primary and fallback... There is no
        pool" — homemath's own README) and also shares a single
        LLM_API_KEY across both slots, which breaks the moment two
        providers need two different keys — both of which make it
        unsuitable for our 4-5 provider free-tier list.

        ollama_chat_stream_dual sidesteps both: it's a stateless per-call
        function that takes url/payload/headers as explicit arguments, not
        cached env-based config — so this loop can call it once per
        provider, each with its own correct URL, model, and Authorization
        header, while still getting homemath's real request/response
        handling (schema-tolerant answer extraction across the different
        shapes various providers use, <think>-block handling).
        """
        import homemath

        self.last_provider_used = None
        if self.safe_mode:
            logger.info("SAFE MODE: using conservative-only LLM handling.")
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}, *messages]
            return "I’m not sure, so I’d rather be cautious. Could you rephrase it more simply?"
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        if not self.providers:
            self._record_route_trace([], None, "no_provider_configured")
            return "I'm having trouble reaching my language model right now, but I'm still here — could you rephrase that?"

        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "120"))
        attempts: list[dict[str, Any]] = []
        for provider in self.providers:
            breaker_state = self._circuit_breaker.setdefault(provider.name, {"failures": 0, "last_failure_time": 0.0})
            failures = breaker_state.get("failures", 0)
            last_failure_time = breaker_state.get("last_failure_time", 0.0)
            if failures >= 3 and time.time() - last_failure_time < 300:
                attempts.append({
                    "provider": provider.name,
                    "status": "skipped",
                    "reason": "circuit_open",
                    "elapsed_ms": 0.0,
                })
                logger.warning(f"Circuit breaker open for {provider.name}, skipping")
                continue

            url = f"{provider.host.rstrip('/')}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
            payload = {"model": provider.model, "messages": messages}
            if max_tokens > 0:
                payload["max_tokens"] = max_tokens
            start = time.time()
            try:
                result = homemath.ollama_chat_stream_dual(
                    url,
                    payload,
                    timeout=int(self.timeout),
                    headers=headers,
                )
                answer = (result.get("content") or "").strip()
                elapsed_ms = (time.time() - start) * 1000
                if answer:
                    if result.get("source") == "choices[0].delta.reasoning_content":
                        logger.warning(
                            f"Provider '{provider.name}' only returned reasoning, no finished "
                            "answer — using it anyway since it's better than silence."
                        )
                    logger.info(f"LLM answer served by provider '{provider.name}' (source={result.get('source')}).")
                    attempts.append({
                        "provider": provider.name,
                        "status": "success",
                        "reason": "success",
                        "elapsed_ms": round(elapsed_ms, 2),
                    })
                    self._circuit_breaker[provider.name] = {"failures": 0, "last_failure_time": 0.0}
                    self.last_provider_used = provider.name
                    self._record_route_trace(attempts, provider.name, "success")
                    return answer
                logger.warning(f"Provider '{provider.name}' returned an empty answer, trying next.")
                attempts.append({
                    "provider": provider.name,
                    "status": "failed",
                    "reason": "empty_response",
                    "elapsed_ms": round(elapsed_ms, 2),
                })
                self._circuit_breaker[provider.name] = {
                    "failures": breaker_state.get("failures", 0) + 1,
                    "last_failure_time": time.time(),
                }
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                reason = "timeout" if "timed out" in str(e).lower() or "timeout" in str(e).lower() else "error"
                attempts.append({
                    "provider": provider.name,
                    "status": "failed",
                    "reason": reason,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": str(e),
                })
                logger.warning(f"Provider '{provider.name}' failed ({e}), trying next.")
                self._circuit_breaker[provider.name] = {
                    "failures": breaker_state.get("failures", 0) + 1,
                    "last_failure_time": time.time(),
                }
                continue

        self._record_route_trace(attempts, None, "all_providers_failed")
        logger.error("All configured LLM providers failed.")
        return "I'm having a bit of trouble thinking right now — can you say that again?"
