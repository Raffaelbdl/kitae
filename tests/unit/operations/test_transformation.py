from gymnasium.spaces import Box, Discrete
import numpy as np
import pytest

from kitae.operations.transformation import linear_interpolation
from kitae.operations.transformation import inverse_linear_interpolation
from kitae.operations.transformation import normalize_frames
from kitae.operations.transformation import action_clip


def test_linear_interpolation():
    assert linear_interpolation(0.0, -1.0, 1.0) == -1.0
    assert linear_interpolation(0.5, -2.0, 0.0) == -1.0
    assert linear_interpolation(-0.5, -2.0, 0.0) == -2.0


def test_inverse_linear_interpolation():
    assert inverse_linear_interpolation(0.0, -1.0, 1.0) == 0.5
    assert inverse_linear_interpolation(0.5, -2.0, 0.0) == 1.0
    assert inverse_linear_interpolation(-0.5, -2.0, 0.0) == 0.75


def test_normalize_frames():
    frame = np.arange(256).reshape((16, 16, 1))
    normalized = normalize_frames(frame)
    assert normalized.shape == frame.shape == (16, 16, 1)
    assert np.max(normalized) == 1.0
    assert np.min(normalized) == -1.0


def test_action_clip():
    box = Box(-1.0, 1.0, (10,), dtype=np.float32)
    dis = Discrete(10)

    a0 = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    with pytest.raises(TypeError):
        action_clip(a0, dis)

    _a0 = action_clip(a0, box)
    assert np.allclose(a0, _a0)

    a1 = np.linspace(-2.0, 1.0, 10, dtype=np.float32)
    _a1 = action_clip(a1, box)
    expected_a1 = np.linspace(-2.0, 1.0, 10, dtype=np.float32)
    expected_a1[:3] = -1.0
    assert np.allclose(_a1, expected_a1)
