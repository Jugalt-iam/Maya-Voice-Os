"""
Orchestration pipeline. This is the piece that "runs everything": it loads
the ASR, LLM (fast-path + homemath routing), and TTS service modules and
wires them into one turn-taking pipeline.

Per-turn flow (each stage only runs if the one before it didn't handle the
turn), fastest/cheapest first:
    1. fast_router     — regex playbook matching, no LLM, no network
    2. smart_response_selector — bank of quick text replies for simple
       intents (greeting/confirmation/closing/frustration), chosen using
       smart_understanding's intent/emotion read of the message
    3. llm_router       — homemath-routed free-tier LLM call (Groq ->
       OpenRouter -> Ollama), for anything the above two didn't cover

Alongside that, every turn also updates:
    - smart_memory (Redis-backed if configured, in-process fallback
      otherwise) — sales-conversation state: objections, buying signals,
      emotional trend, stage
    - context_manager — turn history / working memory, independent of
      ConversationState's raw message list, used for future summarization

Loaded in-process for simplicity (no extra servers/ports to run for a
single-machine, plug-and-play setup). If you later want to split
asr/llm/tts back out into their own deployable services, each folder is
already self-contained and only needs an HTTP wrapper added.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from maya_voice_os.shared.redis_client import get_redis_client

logger = logging.getLogger("orchestration")

ROOT = Path(__file__).resolve().parent.parent

from maya_voice_os.asr_service.engine import ASREngine
from maya_voice_os.llm_service.fast_router import FastPathRouter
from maya_voice_os.llm_service.llm_router import LLMRouter
from maya_voice_os.llm_service.identity_loader import Identity, load_identity
from maya_voice_os.tts_service.engine import get_provider as get_tts_provider
from maya_voice_os.llm_service.smart_understanding import SmartUnderstanding
from maya_voice_os.llm_service.smart_memory import SmartMemoryManager
from maya_voice_os.llm_service.smart_response_selector import SmartResponseSelector
from maya_voice_os.llm_service.context_manager import ContextManager, ContextConfig
from maya_voice_os.llm_service.continual_learning_integration import ContinualLearningManager
from maya_voice_os.llm_service.smart_ai_integration import RouteStats
@dataclass
class ConversationState:
    history: List[dict] = field(default_factory=list)
    # Stable id used as the key into smart_memory / context_manager, which
    # are keyed by string id rather than by this object's identity.
    conversation_id: str = "default"

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]


class VoicePipeline:
    def __init__(self, identity_path: Optional[str] = None):
        self.identity: "Identity" = load_identity(identity_path)
        self.asr = ASREngine(language=self.identity.language)
        self.fast_router = FastPathRouter(
            playbook_dir=str(ROOT / "playbooks"),
            only=self.identity.playbooks or None,
        )
        self.llm_router = LLMRouter()
        self.tts = get_tts_provider()

        self.smart_understanding = SmartUnderstanding()
        self.response_selector = SmartResponseSelector()
        self.smart_memory = SmartMemoryManager(redis_client=None)  # wired in async_init()
        self.context_manager = ContextManager(ContextConfig(), llm_engine=self.llm_router)
        self.continual_learning = ContinualLearningManager()
        self.stats = RouteStats()
        # conversation_id -> last route that answered ("fast_path",
        # "response_selector", "llm:<provider>") — lets POST /feedback
        # attribute a rating without the caller needing to know internals.
        self.last_route_by_conversation: dict = {}
        self._async_initialized = False

    async def async_init(self) -> None:
        """Call once after construction (from the server's startup event)
        to wire up async-only dependencies — currently just Redis, which
        needs an async ping to confirm it's actually reachable."""
        if self._async_initialized:
            return
        redis_client = await get_redis_client()
        self.smart_memory.set_redis_client(redis_client)
        await self.context_manager.initialize()
        self._async_initialized = True

    async def handle_utterance(
        self, audio: np.ndarray, sample_rate: int, state: ConversationState
    ) -> Tuple[str, Optional[str], np.ndarray, dict]:
        """
        Runs one full turn. Returns (reply_text, transcript, reply_pcm16_at_16k, stage_timings_ms).
        stage_timings_ms breaks down where the time actually went — use this
        to diagnose "why is this slow" instead of guessing from logs.
        """
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio, sample_rate=sample_rate)
        asr_ms = (time.perf_counter() - t0) * 1000
        user_text = transcript.text
        logger.info(f"Heard ({transcript.language}): {user_text!r}")

        t1 = time.perf_counter()
        reply_text = await self.respond(user_text, state)
        respond_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        reply_audio = await self.tts.synthesize_pcm16(reply_text, voice=self.identity.voice, sample_rate=16000)
        tts_ms = (time.perf_counter() - t2) * 1000

        timings = {"asr_ms": round(asr_ms, 1), "respond_ms": round(respond_ms, 1), "tts_ms": round(tts_ms, 1)}
        return reply_text, user_text, reply_audio, timings

    async def respond(self, user_text: str, state: ConversationState) -> str:
        if not user_text.strip():
            return ""

        state.add("user", user_text)

        # 1. Fast-path playbooks (regex, no LLM, no network) — unchanged,
        #    still the cheapest/fastest path and tried first.
        fast = self.fast_router.match(user_text)
        if fast is not None:
            logger.info(f"Fast-path hit: intent={fast.intent} playbook={fast.playbook_id}")
            reply = fast.response_text
            self._record_route(state.conversation_id, "fast_path")
            await self._record_turn(user_text, reply, state)
            state.add("assistant", reply)
            return reply

        # 2. Smart understanding (intent/emotion/stage) drives both the
        #    quick response-bank check and smart_memory's tracking below.
        smart_memory_summary = await self._get_smart_memory_context(state.conversation_id)
        understanding = self.smart_understanding.understand(user_text, smart_memory_summary)

        selected = await self.response_selector.select_response(understanding, smart_memory_summary)
        if selected.is_selected and selected.text:
            logger.info(f"SmartResponseSelector hit: {selected.selection_reason}")
            reply = selected.text
            self._record_route(state.conversation_id, "response_selector")
            await self._record_turn(user_text, reply, state, understanding=understanding)
            state.add("assistant", reply)
            return reply

        # 3. LLM fallback (homemath-routed free providers), unchanged.
        reply = self.llm_router.chat(state.history, system_prompt=self.identity.system_prompt)
        provider = self.llm_router.last_provider_used or "unknown"
        self._record_route(state.conversation_id, f"llm:{provider}")
        await self._record_turn(user_text, reply, state, understanding=understanding)
        state.add("assistant", reply)
        return reply

    def _record_route(self, conversation_id: str, route: str) -> None:
        self.stats.record(route)
        self.last_route_by_conversation[conversation_id] = route

    async def _get_smart_memory_context(self, conversation_id: str) -> dict:
        try:
            return await self.smart_memory.get_summary(conversation_id)
        except Exception as e:
            logger.warning(f"smart_memory lookup failed, continuing without it: {e}")
            return {}

    async def _record_turn(
        self,
        user_text: str,
        reply_text: str,
        state: ConversationState,
        understanding=None,
    ) -> None:
        """Best-effort: updates smart_memory and context_manager. Never
        allowed to break a turn — a caller always gets their reply even if
        this bookkeeping fails."""
        try:
            if understanding is not None:
                for objection in understanding.objections:
                    await self.smart_memory.record_objection(state.conversation_id, objection, user_text)
                for signal in understanding.buying_signals:
                    await self.smart_memory.record_buying_signal(state.conversation_id, signal, user_text)
                await self.smart_memory.record_emotional_state(
                    state.conversation_id, understanding.emotional_state.value
                )
            await self.smart_memory.increment_turn(state.conversation_id, is_customer=True, response_length=0)
            await self.smart_memory.increment_turn(
                state.conversation_id, is_customer=False, response_length=len(reply_text)
            )
        except Exception as e:
            logger.warning(f"smart_memory update failed (non-fatal): {e}")

        try:
            if state.conversation_id not in self.context_manager.sessions:
                await self.context_manager.create_session(state.conversation_id, user_id=state.conversation_id)
            await self.context_manager.add_turn(state.conversation_id, user_text, reply_text)
        except Exception as e:
            logger.warning(f"context_manager update failed (non-fatal): {e}")
