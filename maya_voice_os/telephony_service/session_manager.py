"""In-memory session tracking for call flows with TTL eviction."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_TTL_SECONDS = float(os.getenv("SESSION_TTL_SECONDS", "3600"))


@dataclass
class Exchange:
    timestamp: float
    transcript: Optional[str]
    response: Optional[str]
    audio_url: Optional[str]


@dataclass
class SessionState:
    session_id: str
    conversation_id: str
    from_number: Optional[str]
    to_number: Optional[str]
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    exchanges: List[Exchange] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    status: str = "initialised"
    current_processing_task: Optional[asyncio.Task] = None
    awaiting_confirmation: bool = False
    confirmation_retries: int = 0

    def record_exchange(
        self,
        *,
        transcript: Optional[str],
        response: Optional[str],
        audio_url: Optional[str],
    ) -> None:
        now = time.time()
        self.last_activity = now
        self.exchanges.append(Exchange(now, transcript, response, audio_url))


class SessionManager:
    def __init__(self, ttl_seconds: Optional[float] = None) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self.ttl_seconds = DEFAULT_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)

    def _is_expired(self, state: SessionState, now: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (now - state.last_activity) > self.ttl_seconds

    def _prune_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired_ids = [
            session_id
            for session_id, state in self._sessions.items()
            if self._is_expired(state, now)
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    async def prune_expired(self, *, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        async with self._lock:
            expired_ids = [
                session_id
                for session_id, state in self._sessions.items()
                if self._is_expired(state, now)
            ]
            for session_id in expired_ids:
                self._sessions.pop(session_id, None)

    async def create_session(
        self,
        *,
        from_number: Optional[str],
        to_number: Optional[str],
        call_id: Optional[str] = None,
    ) -> SessionState:
        async with self._lock:
            self._prune_locked()
            session_id = call_id or str(uuid.uuid4())
            conversation_id = str(uuid.uuid4())
            state = SessionState(
                session_id=session_id,
                conversation_id=conversation_id,
                from_number=from_number,
                to_number=to_number,
            )
            self._sessions[session_id] = state
            return state

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            self._prune_locked()
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if self._is_expired(state, time.time()):
                self._sessions.pop(session_id, None)
                return None
            return state

    async def upsert(self, state: SessionState) -> None:
        async with self._lock:
            self._prune_locked()
            state.last_activity = time.time()
            self._sessions[state.session_id] = state

    async def list_sessions(self) -> List[SessionState]:  # pragma: no cover - admin helper
        async with self._lock:
            self._prune_locked()
            return list(self._sessions.values())


session_manager = SessionManager()
