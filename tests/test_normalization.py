"""Tests for SkyPlot-provided Matplotlib normalizations."""

import numpy as np
from matplotlib.colors import FuncNorm

from skyplot import PlanckLogNorm


def test_planck_log_norm_uses_map_limits_and_requested_transform() -> None:
    norm = PlanckLogNorm()
    values = np.array([-1.0e4, -10.0, 0.0, 10.0, 1.0e4])

    assert isinstance(norm, FuncNorm)
    assert norm.vmin is None
    assert norm.vmax is None
    forward = np.arcsinh(0.5 * values / 10.0) / np.log(10.0)
    expected = (forward - forward[0]) / (forward[-1] - forward[0])
    np.testing.assert_allclose(norm(values), expected)
    assert norm.vmin == -1.0e4
    assert norm.vmax == 1.0e4
    np.testing.assert_allclose(norm.inverse(norm(values)), values)


def test_planck_log_norm_accepts_color_limits() -> None:
    norm = PlanckLogNorm(vmin=-300.0, vmax=300.0)

    assert norm.vmin == -300.0
    assert norm.vmax == 300.0
    np.testing.assert_allclose(norm([-300.0, 0.0, 300.0]), [0.0, 0.5, 1.0])


def test_planck_log_norm_accepts_a_custom_linear_threshold() -> None:
    values = np.array([-300.0, -20.0, 0.0, 20.0, 300.0])
    norm = PlanckLogNorm(linthresh=20.0)

    forward = np.arcsinh(0.5 * values / 20.0) / np.log(10.0)
    expected = (forward - forward[0]) / (forward[-1] - forward[0])
    np.testing.assert_allclose(norm(values), expected)


def test_planck_log_norm_rejects_invalid_linear_threshold() -> None:
    with np.testing.assert_raises_regex(ValueError, "finite positive"):
        PlanckLogNorm(linthresh=0.0)
