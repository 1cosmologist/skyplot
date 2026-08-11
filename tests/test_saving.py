"""Tests for skyplot saving utilities."""

import json

import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from skyplot.plotting import mollweide
from skyplot.saving import save_figure


def test_save_figure_json_with_explicit_format(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(0).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map_export", output_format="json")

    assert out.suffix == ".json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["backend"] == "matplotlib-cartopy"
    assert "projection" in payload


def test_save_figure_png(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(1).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map.png", figsize=(6, 3), dpi=150)

    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_html(tmp_path) -> None:
    nside = 8
    hp_map = np.random.default_rng(2).normal(size=hp.nside2npix(nside))
    fig = mollweide(hp_map, resolution="low")

    out = save_figure(fig, tmp_path / "map.html")

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<img" in text
    assert "data:image/png;base64" in text
