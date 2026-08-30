"""
Smart AI stats tracking — adapted from smart_ai_integration.py.

DIFFERENCE FROM THE ORIGINAL: the original was a top-level orchestrator
class that wired together audio_cache.py (a pregenerated-audio-clip cache)
and context_engine.py (RAG + CRM integration) — neither of those files was
ever uploaded, and the pieces they *would* coordinate (smart_understanding,
smart_memory, smart_response_selector, context_manager) are already wired
directly into orchestration-service/pipeline.py instead of through a
separate orchestrator class.

What's left with real, portable value from the original — the stats
tracking (cache hit rate, LLM fallback ratio) — is what this module
provides, adapted to this repo's actual three-stage routing (fast_path /
response_selector / llm fallback) instead of the original's audio-cache
hit/miss concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RouteStats:
    total_requests: int = 0
    fast_path_hits: int = 0
    response_selector_hits: int = 0
    llm_fallbacks: int = 0
    llm_provider_counts: Dict[str, int] = field(default_factory=dict)

    def record(self, route: str) -> None:
        self.total_requests += 1
        if route == "fast_path":
            self.fast_path_hits += 1
        elif route == "response_selector":
            self.response_selector_hits += 1
        elif route.startswith("llm:"):
            self.llm_fallbacks += 1
            provider = route.split(":", 1)[1]
            self.llm_provider_counts[provider] = self.llm_provider_counts.get(provider, 0) + 1

    def to_dict(self) -> dict:
        non_llm_hits = self.fast_path_hits + self.response_selector_hits
        hit_rate = (non_llm_hits / self.total_requests * 100) if self.total_requests else 0.0
        return {
            "total_requests": self.total_requests,
            "fast_path_hits": self.fast_path_hits,
            "response_selector_hits": self.response_selector_hits,
            "llm_fallbacks": self.llm_fallbacks,
            "non_llm_hit_rate_percent": round(hit_rate, 1),
            "llm_provider_counts": dict(self.llm_provider_counts),
        }
