"""Tests for skyplot plotting utilities."""
import sys
import types

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")
pytest.importorskip("cartopy")
import cartopy.crs as ccrs

from skyplot.plotting import (
    AVAILABLE_PROJECTIONS,
    RESOLUTION_PRESETS,
    add_gridlines,
    equidistantconic,
    gnomonic,
    mollweide,
    orthographic,
    platecarree,
)


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


@pytest.mark.parametrize(
    "func",
    [mollweide, orthographic, platecarree, equidistantconic, gnomonic],
)
def test_projection_plotters_return_matplotlib_figure(func) -> None:
    nside = 8
    rng = np.random.default_rng(10)
    hp_map = rng.normal(size=hp.nside2npix(nside))

    fig = func(hp_map, n_theta=18, n_phi=36)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1


def test_projection_plotter_accepts_2d_wcs_data() -> None:
    nrows, ncols = 20, 40
    data = np.linspace(0.0, 1.0, nrows * ncols).reshape(nrows, ncols)
    wcs = _LinearDummyWCS(nrows=nrows, ncols=ncols)

    fig = mollweide(data, wcs=wcs, n_theta=12, n_phi=24, interpolate=False)

    assert isinstance(fig, matplotlib.figure.Figure)
    payload = getattr(fig, "_skyplot_payload")
    assert payload["shape"] == [12, 24]


def test_projection_plotter_accepts_ndmap_like_with_wcs_attribute() -> None:
    nrows, ncols = 20, 40
    data = np.linspace(0.0, 1.0, nrows * ncols).reshape(nrows, ncols)
    wcs = _LinearDummyWCS(nrows=nrows, ncols=ncols)
    ndmap_like = _NdmapLike(data, wcs)

    fig = platecarree(ndmap_like, n_theta=10, n_phi=18, interpolate=False)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_projection_plotter_rejects_2d_input_without_wcs() -> None:
    data = np.zeros((20, 40), dtype=float)

    with pytest.raises(ValueError, match="2D map input requires WCS"):
        orthographic(data, n_theta=8, n_phi=16)


def test_projection_plotter_can_overlay_on_existing_axes() -> None:
    nside = 8
    hp_map_a = np.ones(hp.nside2npix(nside))
    hp_map_b = np.full(hp.nside2npix(nside), 2.0)

    fig = mollweide(
        hp_map_a,
        n_theta=12,
        n_phi=24,
        pcolormesh_kwargs={"alpha": 0.8},
    )
    ax = fig.axes[0]

    fig_overlay = mollweide(
        hp_map_b,
        ax=ax,
        n_theta=12,
        n_phi=24,
        show_gridlines=False,
        add_colorbar=False,
        pcolormesh_kwargs={"alpha": 0.4},
    )

    assert fig_overlay is fig
    assert len(fig.axes) == 2


def test_available_projections_is_restricted() -> None:
    assert AVAILABLE_PROJECTIONS == (
        "mollweide",
        "orthographic",
        "platecarree",
        "equidistantconic",
        "gnomonic",
    )


def test_gnomonic_defaults_and_payload_shape() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = gnomonic(hp_map)

    assert isinstance(fig, matplotlib.figure.Figure)
    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection"] == "gnomonic"
    assert payload["shape"] == [500, 500]
    assert payload["projection_kwargs"]["pixel_size_arcmin"] == pytest.approx(5.0)


def test_gnomonic_accepts_2d_wcs_data() -> None:
    nrows, ncols = 20, 40
    data = np.linspace(0.0, 1.0, nrows * ncols).reshape(nrows, ncols)
    wcs = _LinearDummyWCS(nrows=nrows, ncols=ncols)

    fig = gnomonic(
        data,
        wcs=wcs,
        center=(20.0, -10.0),
        xsize=64,
        ysize=32,
        pixel_size_arcmin=10.0,
        interpolate=False,
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["shape"] == [32, 64]
    assert payload["projection_kwargs"]["center"] == [20.0, -10.0]


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


def test_resolution_preset_values() -> None:
    assert RESOLUTION_PRESETS["low"] == (480, 960)
    assert RESOLUTION_PRESETS["medium"] == (720, 1440)
    assert RESOLUTION_PRESETS["high"] == (1440, 2880)


def test_mollweide_figsize_and_dpi() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = mollweide(
        hp_map,
        n_theta=16,
        n_phi=32,
        figsize=(10.0, 5.0),
        dpi=200,
    )

    w, h = fig.get_size_inches()
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(5.0)
    assert fig.dpi == pytest.approx(200.0)


def test_mollweide_accepts_matplotlib_cmap_name() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = mollweide(
        hp_map,
        n_theta=12,
        n_phi=24,
        cmap="viridis",
    )

    assert isinstance(fig, matplotlib.figure.Figure)


def test_projection_kwargs_are_recorded_in_payload() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = orthographic(
        hp_map,
        n_theta=12,
        n_phi=24,
        projection_kwargs={"central_longitude": 20.0, "central_latitude": -10.0},
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection"] == "orthographic"
    assert payload["projection_kwargs"] == {"central_longitude": 20.0, "central_latitude": -10.0}


def test_equidistantconic_extent_is_recorded_in_payload() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        projection_kwargs={"central_longitude": 0.0, "central_latitude": -40.0},
        extent=(-100.0, 20.0, -80.0, -10.0),
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection"] == "equidistantconic"
    assert payload["extent"] == [-100.0, 20.0, -80.0, -10.0]


def test_equidistantconic_defaults_projection_center_from_extent() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        extent=(-100.0, 20.0, -80.0, -10.0),
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection_kwargs"]["central_longitude"] == pytest.approx(-40.0)
    assert payload["projection_kwargs"]["central_latitude"] == pytest.approx(-45.0)


def test_equidistantconic_keeps_explicit_projection_center_over_extent() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        extent=(-100.0, 20.0, -80.0, -10.0),
        projection_kwargs={"central_longitude": 5.0, "central_latitude": -30.0},
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection_kwargs"]["central_longitude"] == pytest.approx(5.0)
    assert payload["projection_kwargs"]["central_latitude"] == pytest.approx(-30.0)


def test_equidistantconic_defaults_cutoff_to_avoid_horizontal_clipping() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        extent=(-100.0, 20.0, -80.0, -10.0),
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection_kwargs"]["cutoff"] == pytest.approx(-90.0)


def test_equidistantconic_keeps_explicit_cutoff() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        extent=(-100.0, 20.0, -80.0, -10.0),
        projection_kwargs={"cutoff": -50.0},
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["projection_kwargs"]["cutoff"] == pytest.approx(-50.0)


@pytest.mark.parametrize(
    "extent",
    [
        (-100.0, 20.0, -10.0),
        (-100.0, 20.0, -10.0, -20.0),
        (10.0, -20.0, -80.0, -10.0),
        (-100.0, 20.0, -95.0, -10.0),
        "-100,20,-80,-10",
    ],
)
def test_equidistantconic_rejects_invalid_extent(extent) -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    with pytest.raises(ValueError, match="extent"):
        equidistantconic(
            hp_map,
            n_theta=8,
            n_phi=16,
            extent=extent,
        )


def test_projection_kwargs_passthrough_to_cartopy() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    with pytest.raises(TypeError):
        mollweide(
            hp_map,
            n_theta=8,
            n_phi=16,
            projection_kwargs={"not_a_real_kwarg": 1},
        )


def test_mollweide_colormaps_name_resolution(monkeypatch) -> None:
    class _DummyCmap:
        def __call__(self, x):
            x = np.asarray(x)
            return np.column_stack([x, np.zeros_like(x), 1.0 - x, np.ones_like(x)])

    fake_module = types.SimpleNamespace(get_cmap=lambda name: _DummyCmap() if name == "batlow" else None)
    monkeypatch.setitem(sys.modules, "colormaps", fake_module)

    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = mollweide(
        hp_map,
        n_theta=12,
        n_phi=24,
        cmap="batlow",
    )

    assert isinstance(fig, matplotlib.figure.Figure)


def test_gridline_defaults_in_payload() -> None:
    hp_map = np.ones(hp.nside2npix(8))
    fig = mollweide(hp_map, n_theta=8, n_phi=16)

    payload = getattr(fig, "_skyplot_payload")
    assert payload["show_gridlines"] is True
    assert payload["gridline_color"] == "black"
    assert payload["gridline_linestyle"] == "-"
    assert payload["gridline_linewidth"] == pytest.approx(0.2)
    assert payload["lon_gridline_spacing_deg"] == pytest.approx(30.0)
    assert payload["lat_gridline_spacing_deg"] == pytest.approx(30.0)
    assert payload["colorbar_orientation"] == "horizontal"


def test_rejects_nonpositive_gridline_linewidth() -> None:
    hp_map = np.ones(hp.nside2npix(8))

    with pytest.raises(ValueError, match="linewidth must be positive"):
        mollweide(
            hp_map,
            n_theta=8,
            n_phi=16,
            gridline_kwargs={"linewidth": 0.0},
        )


def test_rejects_nonpositive_lon_gridline_spacing() -> None:
    hp_map = np.ones(hp.nside2npix(8))

    with pytest.raises(ValueError, match="lon_gridline_spacing_deg must be positive"):
        mollweide(
            hp_map,
            n_theta=8,
            n_phi=16,
            gridline_kwargs={"lon_gridline_spacing_deg": 0.0},
        )


def test_custom_gridline_separation_controls() -> None:
    hp_map = np.ones(hp.nside2npix(8))
    fig = mollweide(
        hp_map,
        n_theta=8,
        n_phi=16,
        gridline_kwargs={
            "lon_gridline_spacing_deg": 45.0,
            "lat_gridline_spacing_deg": 20.0,
        },
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["lon_gridline_spacing_deg"] == pytest.approx(45.0)
    assert payload["lat_gridline_spacing_deg"] == pytest.approx(20.0)


def test_rejects_nonpositive_lat_gridline_spacing() -> None:
    hp_map = np.ones(hp.nside2npix(8))

    with pytest.raises(ValueError, match="lat_gridline_spacing_deg must be positive"):
        mollweide(
            hp_map,
            n_theta=8,
            n_phi=16,
            gridline_kwargs={"lat_gridline_spacing_deg": 0.0},
        )


def test_add_gridlines_accepts_existing_geo_axes() -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    gl = add_gridlines(ax, lon_gridline_spacing_deg=60.0, lat_gridline_spacing_deg=30.0)
    assert gl is not None
