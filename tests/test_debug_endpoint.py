from types import SimpleNamespace

from maya_voice_os.orchestration_service.server import _debug_snapshot
import maya_voice_os.orchestration_service.server as server_module


def test_debug_snapshot_includes_runtime_health():
    snapshot = _debug_snapshot()
    assert "active_sessions" in snapshot
    assert "sessions" in snapshot
    assert "llm_circuit_breakers" in snapshot
    assert "queue_depth" in snapshot
    assert "backpressure" in snapshot
    assert "llm_route_history" in snapshot
    assert "fallback_count_last_10" in snapshot

    assert isinstance(snapshot["active_sessions"], int)
    assert isinstance(snapshot["sessions"], list)
    assert isinstance(snapshot["llm_circuit_breakers"], dict)
    assert isinstance(snapshot["queue_depth"], int)
    assert snapshot["backpressure"] in {"normal", "moderate", "high"}


def test_debug_snapshot_records_provider_fallback_history():
    server_module.pipeline = SimpleNamespace(
        llm_router=SimpleNamespace(
            _circuit_breaker={},
            route_history=[{
                "primary_provider": "groq",
                "selected_provider": "ollama",
                "fallback_count": 1,
                "reason": "timeout",
                "attempts": [
                    {"provider": "groq", "status": "failed", "reason": "timeout", "elapsed_ms": 2100.0},
                    {"provider": "ollama", "status": "success", "reason": "success", "elapsed_ms": 1300.0},
                ],
            }]
        )
    )

    snapshot = _debug_snapshot()
    assert snapshot["fallback_count_last_10"] == 1
    assert snapshot["last_route_trace"]["selected_provider"] == "ollama"
    assert snapshot["last_route_trace"]["reason"] == "timeout"
