import os
from types import SimpleNamespace


def test_barge_in_demo_mode_is_exposed_via_env(monkeypatch):
    monkeypatch.setenv("BARGE_IN_DEMO_MODE", "true")
    import importlib
    import maya_voice_os.telephony_service.adapters.twilio_adapter as twilio_adapter

    importlib.reload(twilio_adapter)
    assert twilio_adapter.BARGE_IN_DEMO_MODE is True


def test_demo_gap_plan_includes_interruptible_pauses():
    import maya_voice_os.telephony_service.adapters.twilio_adapter as twilio_adapter

    gaps = twilio_adapter.build_demo_gap_plan(duration_ms=5000)
    assert gaps
    assert all(gap["gap_ms"] > 0 for gap in gaps)
    assert any(gap["at_ms"] >= 1000 for gap in gaps)
