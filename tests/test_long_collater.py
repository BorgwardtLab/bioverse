import numpy as np

from bioverse.collaters.long import _as_long_array


def test_as_long_array_squeezes_singleton_nesting():
    features = np.zeros((4, 1, 1, 10, 20))
    out = _as_long_array(features)
    assert out.shape == (40, 20)


def test_as_long_array_keeps_frames():
    frames = np.zeros((10, 3, 3))
    out = _as_long_array(frames)
    assert out.shape == (10, 3, 3)


def test_as_long_array_flattens_equal_length_positions():
    pos = np.zeros((4, 10, 3))
    out = _as_long_array(pos)
    assert out.shape == (40, 3)
