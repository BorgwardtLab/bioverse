from bioverse.trainer import finite_score, score_improved


def test_finite_score_rejects_nan_and_none():
    assert finite_score(1.5) == 1.5
    assert finite_score("2") == 2.0
    assert finite_score(None) is None
    assert finite_score(float("nan")) is None
    assert finite_score("bad") is None


def test_score_improved_higher_is_better():
    assert score_improved(0.2, None, higher=True)
    assert score_improved(0.3, 0.2, higher=True)
    assert not score_improved(0.2, 0.2, higher=True)
    assert not score_improved(0.19, 0.2, higher=True)
    assert not score_improved(float("nan"), 0.2, higher=True)


def test_score_improved_lower_is_better():
    assert score_improved(1.0, None, higher=False)
    assert score_improved(0.9, 1.0, higher=False)
    assert not score_improved(1.0, 1.0, higher=False)
    assert not score_improved(1.1, 1.0, higher=False)


def test_score_improved_min_delta():
    assert not score_improved(0.21, 0.2, higher=True, min_delta=0.05)
    assert score_improved(0.26, 0.2, higher=True, min_delta=0.05)
    assert not score_improved(0.96, 1.0, higher=False, min_delta=0.05)
    assert score_improved(0.94, 1.0, higher=False, min_delta=0.05)
