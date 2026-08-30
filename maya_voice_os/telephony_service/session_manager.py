"""In-memory session tracking for call flows."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        *,
        from_number: Optional[str],
        to_number: Optional[str],
        call_id: Optional[str] = None,
    ) -> SessionState:
        async with self._lock:
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
            return self._sessions.get(session_id)

    async def upsert(self, state: SessionState) -> None:
        async with self._lock:
            self._sessions[state.session_id] = state

    async def list_sessions(self) -> List[SessionState]:  # pragma: no cover - admin helper
        async with self._lock:
            return list(self._sessions.values())


session_manager = SessionManager()
