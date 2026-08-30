import asyncio
import time

from maya_voice_os.telephony_service.session_manager import SessionManager


def test_session_manager_prunes_expired_sessions():
    async def _run() -> None:
        manager = SessionManager(ttl_seconds=0.01)
        state = await manager.create_session(from_number="+100", to_number="+200")
        state.last_activity = time.time() - 60

        await manager.prune_expired()

        assert await manager.get(state.session_id) is None
        assert state.session_id not in {s.session_id for s in await manager.list_sessions()}

    asyncio.run(_run())
