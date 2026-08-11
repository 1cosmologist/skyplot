"""Tests for gnomonic plotting utility."""

import healpy as hp
import matplotlib
import numpy as np
import pytest

from skyplot.plotting import gnomonic

matplotlib.use("Agg")


class _LinearDummyWCS:
    """Simple linear lon/lat -> pixel mapping for tests."""

    def __init__(self, nrows: int, ncols: int) -> None:
        self.nrows = nrows
        self.ncols = ncols

    def all_world2pix(self, world, origin):
        arr = np.asarray(world, dtype=float)
        lon = arr[:, 0]
        lat = arr[:, 1]
        x = ((lon + 180.0) / 360.0) * (self.ncols - 1)
        y = ((lat + 90.0) / 180.0) * (self.nrows - 1)
        return np.column_stack([x, y])


class _NdmapLike:
    def __init__(self, data: np.ndarray, wcs) -> None:
        self._data = data
        self.wcs = wcs

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


def test_gnomonic_returns_figure_for_healpix() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = gnomonic(hp_map, center=(45.0, -20.0), xsize=80, ysize=64, pixel_size_arcmin=6.0)

    assert isinstance(fig, matplotlib.figure.Figure)
    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection"] == "gnomonic"
    assert payload["shape"] == [64, 80]


def test_gnomonic_accepts_2d_wcs_data() -> None:
    nrows, ncols = 20, 40
    data = np.linspace(0.0, 1.0, nrows * ncols).reshape(nrows, ncols)
    wcs = _LinearDummyWCS(nrows=nrows, ncols=ncols)

    fig = gnomonic(
        data,
        wcs=wcs,
        center=(20.0, -10.0),
        xsize=48,
        ysize=32,
        pixel_size_arcmin=10.0,
        interpolate=False,
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["shape"] == [32, 48]
    assert payload["projection_kwargs"]["center"] == [20.0, -10.0]


def test_gnomonic_accepts_ndmap_like_with_wcs_attribute() -> None:
    nrows, ncols = 20, 40
    data = np.linspace(0.0, 1.0, nrows * ncols).reshape(nrows, ncols)
    wcs = _LinearDummyWCS(nrows=nrows, ncols=ncols)
    ndmap_like = _NdmapLike(data, wcs)

    fig = gnomonic(ndmap_like, center=(0.0, 0.0), xsize=32, ysize=24)

    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xsize": 0},
        {"ysize": 0},
        {"pixel_size_arcmin": 0.0},
        {"center": (0.0,)},
        {"center": (0.0, 100.0)},
    ],
)
def test_gnomonic_rejects_invalid_parameters(kwargs) -> None:
    hp_map = np.ones(hp.nside2npix(8))

    with pytest.raises(ValueError):
        gnomonic(hp_map, **kwargs)
