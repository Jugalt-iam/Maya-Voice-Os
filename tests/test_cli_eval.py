import math

from maya_voice_os.cli import _percentile, _score_fixed_test_set


def test_percentile_values_are_stable():
    samples = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert _percentile(samples, 50) == 300.0
    assert _percentile(samples, 95) == 500.0
    assert _percentile(samples, 99) == 500.0


def test_fixed_test_set_score_handles_expected_and_unexpected_results():
    expected = ["hello", "goodbye"]
    actual = ["hello", "nope"]
    score = _score_fixed_test_set(expected, actual)
    assert score == 0.5
    assert math.isclose(score, 0.5, rel_tol=0.0, abs_tol=1e-9)
