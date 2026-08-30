import os

from maya_voice_os.asr_service.engine import filter_hallucination
from maya_voice_os.llm_service.llm_router import LLMRouter


def test_safe_mode_filters_more_hallucinations():
    text, was_hallucination = filter_hallucination("please subscribe", safe_mode=True)
    assert was_hallucination is True
    assert text == ""


def test_safe_mode_uses_conservative_llm_fallback():
    original = os.environ.get("SAFE_MODE")
    os.environ["SAFE_MODE"] = "true"
    try:
        router = LLMRouter()
        response = router.chat([], system_prompt="hi")
        assert "not sure" in response.lower()
        assert "rephrase" in response.lower()
    finally:
        if original is None:
            os.environ.pop("SAFE_MODE", None)
        else:
            os.environ["SAFE_MODE"] = original
