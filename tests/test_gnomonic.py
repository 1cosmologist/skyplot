"""Tests for gnomonic plotting utility."""

import healpy as hp
import matplotlib
import numpy as np
import pytest

from skyplot.plotlib import _gnomonic_inverse
from skyplot.plotting import gnomonic

matplotlib.use("Agg")


class _LinearDummyWCS:
    """Simple linear lon/lat -> pixel mapping for tests."""

    def __init__(self, nrows: int, ncols: int) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.world_axis_physical_types = ("pos.eq.ra", "pos.eq.dec")

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


def test_gnomonic_inverse_uses_true_tangent_plane_scale() -> None:
    plane_x = np.array([[np.tan(np.radians(60.0))]])
    lon, lat = _gnomonic_inverse(
        lon0_deg=0.0,
        lat0_deg=0.0,
        x_plane=plane_x,
        y_plane=np.zeros_like(plane_x),
    )

    assert lon[0, 0] == pytest.approx(60.0)
    assert lat[0, 0] == pytest.approx(0.0)


def test_gnomonic_overlay_orientation_is_idempotent() -> None:
    hp_map = np.ones(hp.nside2npix(2))
    fig = gnomonic(hp_map, xsize=8, ysize=8, add_colorbar=False)
    ax = fig.axes[0]
    assert ax.xaxis_inverted()

    gnomonic(hp_map, ax=ax, xsize=8, ysize=8, add_colorbar=False)
    assert ax.xaxis_inverted()

    gnomonic(
        hp_map,
        ax=ax,
        xsize=8,
        ysize=8,
        add_colorbar=False,
        astro_orientation=False,
    )
    assert not ax.xaxis_inverted()


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
