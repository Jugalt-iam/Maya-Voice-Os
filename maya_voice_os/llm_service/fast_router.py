"""
Fast-path router — instant, free, no-LLM-call responses for known patterns
(greetings, FAQs, objections), matched against YAML playbooks.

Ported from the original orchestration-service/fast_path_router.py, stripped
of Redis / audio-cache-key / multi-agent coupling. Any input that doesn't hit
a fast-path pattern falls through to the LLM router (see llm_router.py).

Every trigger is matched with word boundaries (\\btrigger\\b), not plain
substring search — a short trigger like "hi" was matching inside unrelated
words ("phir") before this. Because of that, each stored pattern entry keeps
the ORIGINAL plain trigger text alongside the compiled regex — word-count
and startswith() checks below use that plain text, not pattern.pattern
(which now has \\b/escaping baked in and is no longer a plain substring of
real input — an actual regression from adding the boundary matching that
was caught by testing against the original file's word-count guards).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("fast_router")


def _compile_trigger(trigger: str) -> re.Pattern:
    """
    Compiles a trigger phrase with word boundaries, so a short trigger like
    "hi" matches the word "hi" and not the substring inside "phir", "this",
    "history", etc. Plain re.escape(trigger) alone does unanchored substring
    matching, which caused exactly that collision during testing (Hindi
    "phir main" was matching the English "hi" greeting trigger).
    \\b works correctly here since triggers are ASCII/romanized text, not
    Devanagari script (which \\b doesn't reliably bound in Python's re).
    """
    return re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)


@dataclass
class FastPathResult:
    matched: bool = False
    response_text: str = ""
    playbook_id: str = ""
    intent: str = ""
    confidence: float = 0.0
    next_action: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)


class FastPathRouter:
    def __init__(self, playbook_dir: str = "playbooks", only: Optional[List[str]] = None):
        """
        playbook_dir: folder of *.yaml playbooks.
        only: optional list of playbook ids/filenames to load (identity files
              can restrict which playbooks apply). None/[] = load all.
        """
        self.playbook_dir = Path(playbook_dir)
        self.only = set(only) if only else None

        self.playbooks: Dict[str, Dict] = {}
        # Each entry: (compiled_pattern, plain_trigger_text, response, playbook_id, confidence)
        self.faq_patterns: List[Tuple[re.Pattern, str, str, str, float]] = []
        # Each entry: (compiled_pattern, plain_trigger_text, response, playbook_id)
        self.greeting_patterns: List[Tuple[re.Pattern, str, str, str]] = []
        self.objection_patterns: List[Tuple[re.Pattern, str, Optional[str], str]] = []

        self._load_all()

    def _load_all(self) -> int:
        if not self.playbook_dir.exists():
            logger.warning(f"Playbook directory not found: {self.playbook_dir}")
            return 0

        loaded = 0
        for yaml_file in sorted(self.playbook_dir.glob("*.yaml")):
            playbook_id_guess = yaml_file.stem
            if self.only and playbook_id_guess not in self.only:
                continue
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    playbook = yaml.safe_load(f)
                if not playbook:
                    continue
                playbook_id = playbook.get("id", playbook_id_guess)
                self.playbooks[playbook_id] = playbook
                self._extract_patterns(playbook)
                loaded += 1
                logger.info(f"Loaded playbook: {playbook_id}")
            except Exception as e:
                logger.error(f"Failed to load playbook {yaml_file}: {e}")

        logger.info(f"Loaded {loaded} playbooks, {len(self.faq_patterns)} FAQ patterns.")
        return loaded

    def _extract_patterns(self, playbook: Dict) -> None:
        playbook_id = playbook.get("id", "unknown")

        for entry in playbook.get("faq_entries", {}).values():
            confidence = entry.get("confidence", 0.85)

            # Bilingual entries (optional, backward-compatible): if an entry
            # defines triggers_en/triggers_hi + response_en/response_hi, the
            # matched trigger's own language decides the response language —
            # e.g. "who are you" (English trigger) gets response_en, "aap
            # kaun ho" (Hindi trigger) gets response_hi. This classifies the
            # short, curated TRIGGER phrase, never the free-form response
            # text, which is what made an earlier attempt at this misfire
            # (common words like "main" are ambiguous in response prose, but
            # not in a handful of triggers we write ourselves).
            if "triggers_en" in entry or "triggers_hi" in entry:
                response_en = entry.get("response_en", entry.get("response", ""))
                response_hi = entry.get("response_hi", entry.get("response", ""))
                for trigger in entry.get("triggers_en", []):
                    pattern = _compile_trigger(trigger)
                    self.faq_patterns.append((pattern, trigger, response_en, playbook_id, confidence))
                for trigger in entry.get("triggers_hi", []):
                    pattern = _compile_trigger(trigger)
                    self.faq_patterns.append((pattern, trigger, response_hi, playbook_id, confidence))
                continue

            response = entry.get("response", "")
            for trigger in entry.get("triggers", []):
                pattern = _compile_trigger(trigger)
                self.faq_patterns.append((pattern, trigger, response, playbook_id, confidence))

        for greeting_data in playbook.get("greetings", {}).values():
            # Same bilingual pattern as FAQ entries above: triggers_en /
            # triggers_hi + responses_en / responses_hi, so which trigger
            # matched decides the response language, instead of randomly
            # picking once at load time from a mixed-language list (the
            # old behavior — meant caller language had zero say in it).
            if "triggers_en" in greeting_data or "triggers_hi" in greeting_data:
                responses_en = greeting_data.get("responses_en") or greeting_data.get("responses", [])
                responses_hi = greeting_data.get("responses_hi") or greeting_data.get("responses", [])
                if responses_en:
                    chosen_en = random.choice(responses_en)
                    for trigger in greeting_data.get("triggers_en", []):
                        pattern = _compile_trigger(trigger)
                        self.greeting_patterns.append((pattern, trigger, chosen_en, playbook_id))
                if responses_hi:
                    chosen_hi = random.choice(responses_hi)
                    for trigger in greeting_data.get("triggers_hi", []):
                        pattern = _compile_trigger(trigger)
                        self.greeting_patterns.append((pattern, trigger, chosen_hi, playbook_id))
                continue

            responses = greeting_data.get("responses", [])
            if not responses:
                continue
            chosen = random.choice(responses)
            for trigger in greeting_data.get("triggers", []):
                pattern = _compile_trigger(trigger)
                self.greeting_patterns.append((pattern, trigger, chosen, playbook_id))

        for obj_data in playbook.get("objections", {}).values():
            response = obj_data.get("response", "")
            next_action = obj_data.get("next_action")
            for trigger in obj_data.get("triggers", []):
                pattern = _compile_trigger(trigger)
                self.objection_patterns.append((pattern, response, next_action, playbook_id))

    def match_greeting(self, user_input: str) -> Optional[FastPathResult]:
        """
        Restores the original file's stricter matching (which the first
        port missed): for 4-5 word inputs, the trigger must be at the
        START of the input, AND whatever follows the trigger must be 2
        words or fewer. Without this, "hello can you help me with pricing"
        would match the bare greeting and never reach the LLM/smart layers
        that could actually answer the pricing question.
        """
        input_lower = user_input.lower().strip()
        words = input_lower.split()
        if len(words) > 5:
            return None
        for pattern, trigger, response, playbook_id in self.greeting_patterns:
            if len(words) > 3:
                if not input_lower.startswith(trigger.lower()):
                    continue
                remainder = input_lower[len(trigger):].strip()
                remainder_words = len(remainder.split()) if remainder else 0
                if remainder_words > 2:
                    continue
            if pattern.search(input_lower):
                return FastPathResult(
                    matched=True, response_text=response,
                    playbook_id=playbook_id, intent="greeting", confidence=0.95,
                )
        return None

    def match_faq(self, user_input: str) -> Optional[FastPathResult]:
        input_lower = user_input.lower().strip()
        word_count = len(input_lower.split())
        if word_count > 8:
            return None
        for pattern, trigger, response, playbook_id, confidence in self.faq_patterns:
            if not pattern.search(input_lower):
                continue
            trigger_word_count = len(trigger.split())
            if trigger_word_count == 1 and word_count >= 6:
                continue
            return FastPathResult(
                matched=True, response_text=response,
                playbook_id=playbook_id, intent="faq", confidence=confidence,
            )
        return None

    def match_objection(self, user_input: str) -> Optional[FastPathResult]:
        input_lower = user_input.lower().strip()
        for pattern, response, next_action, playbook_id in self.objection_patterns:
            if pattern.search(input_lower):
                return FastPathResult(
                    matched=True, response_text=response, next_action=next_action,
                    playbook_id=playbook_id, intent="objection", confidence=0.8,
                )
        return None

    def match(self, user_input: str) -> Optional[FastPathResult]:
        """Try greeting -> FAQ -> objection, in that order. First hit wins."""
        return (
            self.match_greeting(user_input)
            or self.match_faq(user_input)
            or self.match_objection(user_input)
        )
