"""Tests for skyplot saving utilities."""

import healpy as hp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from skyplot.plotting import mollweide
from skyplot.saving import save_figure


def test_save_figure_svg_inferred_from_suffix(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map_export.svg")

    assert out.suffix == ".svg"
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_png(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(1).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map.png", figsize=(6, 3), dpi=150)

    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_defaults_to_png_without_suffix(tmp_path) -> None:
    hp_map = np.random.default_rng(3).normal(size=hp.nside2npix(8))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map_export")

    assert out == tmp_path / "map_export.png"
    assert out.exists()


def test_save_figure_unsupported_format(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(2).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "map.webp")
