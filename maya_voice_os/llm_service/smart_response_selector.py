"""
Smart Response Selector — adapted from the original Aayna module.

DIFFERENCE FROM THE ORIGINAL: the original selected pre-generated AUDIO
files by cache key (e.g. "aayna_pricing__pricing_value") from a large
pregenerated-audio-clip library that isn't part of this repo. We don't ship
that library — Maya synthesizes replies live via TTS every turn — so this
version selects TEXT responses instead, from a small built-in bank plus
whatever multi-response playbook entries already exist. The selection
intelligence (subtext/emotion-aware picking, epsilon-greedy performance
tracking) is preserved as-is; only the data source changed.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from maya_voice_os.llm_service.smart_understanding import (
    Understanding,
    EmotionalState,
    RecommendedAction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TEXT RESPONSE BANK (replaces the original's pregenerated-audio-key bank)
# ============================================================================
# Organized the same way the original was (intent -> subtext -> options),
# but each option is response TEXT, synthesized live via TTS, not a cache key.

TEXT_RESPONSE_BANK: Dict[str, Dict[str, List[str]]] = {
    "greeting": {
        "default": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
        ],
    },
    "confirmation": {
        "default": [
            "Got it.",
            "Understood.",
            "Okay, noted.",
        ],
    },
    "closing": {
        "default": [
            "Thanks for calling — take care!",
            "Thank you, have a great day!",
        ],
    },
    "frustration": {
        "default": [
            "I'm sorry about that — let me help you sort this out.",
            "I understand the frustration, let's get this fixed.",
        ],
    },
}


@dataclass
class SelectedResponse:
    """Selected response with metadata."""
    text: Optional[str] = None
    is_selected: bool = False
    fallback_to_llm: bool = False
    selection_reason: str = ""
    confidence: float = 0.0
    bank_key: Optional[str] = None  # for recording outcomes later


# Performance tracking: maps a bank key ("intent/subtext/option_index") to
# (conversion_rate, usage_count). In-memory only, per-process — this is a
# lightweight heuristic, not persistent learning; wire it to real storage
# if you want it to survive restarts.
_PERFORMANCE: Dict[str, Tuple[float, int]] = {}


class SmartResponseSelector:
    """
    Picks a fast, appropriate text response for simple/known intents based
    on SmartUnderstanding's output, so those turns can skip the LLM
    entirely. Falls back to the LLM for anything not covered.
    """

    def __init__(self, extra_bank: Optional[Dict[str, Dict[str, List[str]]]] = None):
        """extra_bank lets you merge in additional intent->responses entries
        (e.g. built from playbook YAML multi-response lists) without
        editing this file."""
        self.bank: Dict[str, Dict[str, List[str]]] = {**TEXT_RESPONSE_BANK}
        if extra_bank:
            for intent, subtext_map in extra_bank.items():
                self.bank.setdefault(intent, {}).update(subtext_map)

    async def select_response(
        self,
        understanding: Understanding,
        context: Optional[Dict[str, Any]] = None,
    ) -> SelectedResponse:
        context = context or {}
        intent = understanding.intent

        # Only use this fast path for the small set of simple, safe
        # intents — anything else (objections, product questions, etc.)
        # goes to the LLM, same as the original's design intent.
        simple_intents = ("greeting", "confirmation", "closing", "frustration")
        if intent not in simple_intents or intent not in self.bank:
            return SelectedResponse(
                fallback_to_llm=True,
                selection_reason=f"No fast-path bank for intent={intent}",
                confidence=understanding.confidence,
            )

        subtext_key = self._get_subtext_key(understanding)
        bank_entry = self.bank[intent]
        options = bank_entry.get(subtext_key) or bank_entry.get("default", [])

        if not options:
            return SelectedResponse(
                fallback_to_llm=True,
                selection_reason=f"Bank for {intent} has no options",
            )

        idx, text = self._select_best_performer(intent, subtext_key, options)
        bank_key = f"{intent}/{subtext_key}/{idx}"

        logger.info(f"SmartResponseSelector: intent={intent} subtext={subtext_key} -> {bank_key}")
        return SelectedResponse(
            text=text,
            is_selected=True,
            selection_reason=f"Bank match: {intent}/{subtext_key}",
            confidence=0.9,
            bank_key=bank_key,
        )

    def _get_subtext_key(self, understanding: Understanding) -> str:
        subtext = (understanding.subtext or "").lower()
        if understanding.emotional_state == EmotionalState.FRUSTRATED:
            return "default"
        if understanding.detected_language == "hi":
            return "hindi" if "hindi" in subtext else "default"
        return "default"

    def _select_best_performer(self, intent: str, subtext_key: str, options: List[str]) -> Tuple[int, str]:
        """Epsilon-greedy: 90% pick the best-tracked performer, 10% explore
        a random option, same policy as the original."""
        if random.random() < 0.1 or not options:
            idx = random.randrange(len(options))
            return idx, options[idx]

        best_idx, best_score = 0, -1.0
        for i, _ in enumerate(options):
            key = f"{intent}/{subtext_key}/{i}"
            if key in _PERFORMANCE:
                rate, count = _PERFORMANCE[key]
                score = rate - (1.0 / max(count, 1))
            else:
                score = 0.5
            if score > best_score:
                best_score, best_idx = score, i
        return best_idx, options[best_idx]

    def record_outcome(self, bank_key: str, converted: bool) -> None:
        """Record whether a selected response led to a good outcome, for
        the epsilon-greedy selection to improve over time."""
        if bank_key not in _PERFORMANCE:
            _PERFORMANCE[bank_key] = (0.5, 0)
        current_rate, count = _PERFORMANCE[bank_key]
        weight = 0.1
        new_rate = current_rate * (1 - weight) + (1.0 if converted else 0.0) * weight
        _PERFORMANCE[bank_key] = (new_rate, count + 1)


_response_selector: Optional[SmartResponseSelector] = None


def get_response_selector(extra_bank: Optional[Dict[str, Dict[str, List[str]]]] = None) -> SmartResponseSelector:
    global _response_selector
    if _response_selector is None:
        _response_selector = SmartResponseSelector(extra_bank=extra_bank)
    return _response_selector
