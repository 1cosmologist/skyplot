"""Tests for skyplot plotting utilities."""
import sys
import types

import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
import skyplot.plotlib as plotlib
import skyplot.plotting as plotting

matplotlib.use("Agg")
pytest.importorskip("cartopy")
import cartopy.crs as ccrs

from skyplot.plotting import (
    AVAILABLE_PROJECTIONS,
    DPI_PRESETS,
    FONT_SIZE_PRESETS,
    RESOLUTION_PRESETS,
    add_gridlines,
    equidistantconic,
    gnomonic,
    mollweide,
    orthographic,
    platecarree,
)
from skyplot.plotlib import (
    _display_grid_cache,
    _get_display_grid,
    _sample_wcs_map,
    _transform_display_coordinates,
)


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


class _ReversedWorldAxisWCS:
    """Test WCS whose native world order is (Dec, RA)."""

    world_axis_physical_types = ("pos.eq.dec", "pos.eq.ra")

    def __init__(self) -> None:
        self.last_world = None

    def all_world2pix(self, world, origin):
        self.last_world = np.asarray(world, dtype=float)
        return np.column_stack([self.last_world[:, 1], self.last_world[:, 0]])


class _UnlabelledReversedWorldAxisWCS:
    """Test WCS with no angular-axis metadata."""

    def all_world2pix(self, world, origin):
        arr = np.asarray(world, dtype=float)
        return np.column_stack([arr[:, 1], arr[:, 0]])


class _PeriodicLongitudeWCS:
    """Simple WCS that accepts only the [0, 360) longitude interval."""

    world_axis_physical_types = ("pos.eq.ra", "pos.eq.dec")

    def all_world2pix(self, world, origin):
        arr = np.asarray(world, dtype=float)
        return np.column_stack([arr[:, 0], arr[:, 1]])


class _NdmapLike:
    def __init__(self, data: np.ndarray, wcs) -> None:
        self._data = data
        self.wcs = wcs

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


def test_plotting_public_api_delegates_to_plotlib() -> None:
    assert plotting.mollweide.__module__ == "skyplot.plotting"
    assert plotting.add_gridlines.__module__ == "skyplot.plotting"
    assert callable(plotlib.plot_with_projection)
    assert not hasattr(plotlib, "plot_mollweide")
    assert not hasattr(plotlib, "plot_gnomonic")


def test_display_grid_cache_reuses_immutable_geometry() -> None:
    _display_grid_cache.clear()

    lon_a, lat_a = _get_display_grid(8, 16)
    lon_b, lat_b = _get_display_grid(8, 16)

    assert lon_a is lon_b
    assert lat_a is lat_b
    assert not lon_a.flags.writeable
    assert not lat_a.flags.writeable
    assert lon_a.shape == (8, 16)
    assert np.all(np.diff(lon_a[0]) >= 0.0)
    assert np.all(np.diff(lat_a[:, 0]) >= 0.0)


def test_coordinate_transform_converts_display_grid_to_input_frame() -> None:
    """A Galactic display coordinate is sampled from the matching ICRS point."""
    display_lon = np.array([[0.0]])
    display_lat = np.array([[0.0]])

    source_lon, source_lat, source_label, display_label = _transform_display_coordinates(
        display_lon,
        display_lat,
        coordinate_frame=None,
        coordinate_transform=("icrs", "galactic"),
    )

    assert source_lon[0, 0] == pytest.approx(266.4051, abs=1e-3)
    assert source_lat[0, 0] == pytest.approx(-28.9362, abs=1e-3)
    assert source_label is None
    assert display_label == ["icrs", "galactic"]


def test_wcs_sampling_infers_reversed_ra_dec_world_axis_order() -> None:
    data = np.arange(25, dtype=float).reshape(5, 5)
    wcs = _ReversedWorldAxisWCS()

    values = _sample_wcs_map(
        data,
        wcs=wcs,
        lon=np.array([[1.0]]),
        lat=np.array([[2.0]]),
        interpolate=False,
        world_axis_mapping=None,
    )

    assert np.array_equal(wcs.last_world, [[2.0, 1.0]])
    assert values[0, 0] == 11.0


def test_wcs_sampling_requires_explicit_mapping_when_metadata_is_ambiguous() -> None:
    data = np.arange(25, dtype=float).reshape(5, 5)
    wcs = _UnlabelledReversedWorldAxisWCS()
    lon = np.array([[1.0]])
    lat = np.array([[2.0]])

    with pytest.raises(ValueError, match="world_axis_mapping"):
        _sample_wcs_map(
            data,
            wcs=wcs,
            lon=lon,
            lat=lat,
            interpolate=False,
            world_axis_mapping=None,
        )

    values = _sample_wcs_map(
        data,
        wcs=wcs,
        lon=lon,
        lat=lat,
        interpolate=False,
        world_axis_mapping=(1, 0),
    )
    assert values[0, 0] == 11.0


def test_wcs_sampling_wraps_longitude_and_clamps_edge_neighbors() -> None:
    data = np.arange(5 * 360, dtype=float).reshape(5, 360)
    wcs = _PeriodicLongitudeWCS()

    values = _sample_wcs_map(
        data,
        wcs=wcs,
        lon=np.array([[-1.0, 359.0]]),
        lat=np.array([[4.0, 4.0]]),
        interpolate=True,
        world_axis_mapping=None,
    )

    # -1 degrees is sampled via its 359-degree WCS alias. The last image row
    # and column are valid and use clamped bilinear neighbors.
    assert np.array_equal(values, [[data[4, 359], data[4, 359]]])


@pytest.mark.parametrize(
    "func",
    [mollweide, orthographic, platecarree, equidistantconic],
)
def test_non_gnomonic_projections_record_coordinate_transform(func) -> None:
    hp_map = np.ones(hp.nside2npix(2))

    fig = func(
        hp_map,
        n_theta=8,
        n_phi=16,
        coordinate_frame="galactic",
        coordinate_transform=("galactic", "icrs"),
    )

    payload = getattr(fig, "_skyplot_payload")
    assert payload["coordinate_frame"] == "galactic"
    assert payload["coordinate_transform"] == ["galactic", "icrs"]


def test_badvalue_and_badcolor_are_applied_to_plot_colormap() -> None:
    hp_map = np.ones(hp.nside2npix(2))
    hp_map[0] = hp.UNSEEN

    fig = mollweide(
        hp_map,
        n_theta=8,
        n_phi=16,
        interpolate=False,
        badcolor="magenta",
    )

    quad = fig.axes[0].collections[0]
    assert np.allclose(quad.cmap.get_bad(), to_rgba("magenta"))
    payload = getattr(fig, "_skyplot_payload")
    assert payload["badvalue"] == hp.UNSEEN
    assert payload["badcolor"] == "magenta"


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


def test_projection_plotter_rejects_complex_map() -> None:
    hp_map = np.ones(hp.nside2npix(2), dtype=complex)

    with pytest.raises(ValueError, match="real-valued"):
        mollweide(hp_map, n_theta=8, n_phi=16)


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


def test_overlay_mask_draws_only_invalid_pixels() -> None:
    hp_map = np.ones(hp.nside2npix(2))
    allowed_mask = np.ones_like(hp_map, dtype=bool)
    allowed_mask[:12] = False

    fig = mollweide(
        hp_map,
        n_theta=8,
        n_phi=16,
        show_gridlines=False,
        add_colorbar=False,
    )
    ax = fig.axes[0]
    mollweide(
        allowed_mask,
        ax=ax,
        n_theta=8,
        n_phi=16,
        overlay_mask=True,
        overlay_color="magenta",
        cmap="viridis",
        vmin=-5.0,
        vmax=5.0,
        show_gridlines=True,
        add_colorbar=True,
    )

    overlay = ax.collections[-1]
    values = overlay.get_array()
    assert np.ma.isMaskedArray(values)
    assert np.any(np.ma.getmaskarray(values))
    assert overlay.get_alpha() == pytest.approx(0.25)
    assert overlay.cmap(0.0) == pytest.approx(to_rgba("magenta"))
    assert len(fig.axes) == 1
    payload = getattr(fig, "_skyplot_payload")
    assert payload["vmin"] is None
    assert payload["vmax"] is None


def test_overlay_mask_requires_binary_input() -> None:
    non_binary_mask = np.ones(hp.nside2npix(2))
    non_binary_mask[0] = 2.0

    with pytest.raises(ValueError, match="binary mask"):
        mollweide(
            non_binary_mask,
            n_theta=8,
            n_phi=16,
            overlay_mask=True,
        )


def test_existing_axes_extent_overrides_extent_argument() -> None:
    hp_map = np.ones(hp.nside2npix(8))
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent((-80.0, -20.0, 10.0, 50.0), crs=ccrs.PlateCarree())

    result = platecarree(
        hp_map,
        ax=ax,
        extent=(-150.0, -100.0, -60.0, -20.0),
        n_theta=12,
        n_phi=24,
        show_gridlines=False,
        add_colorbar=False,
    )

    payload = getattr(result, "_skyplot_payload")
    assert result is fig
    assert payload["extent"] == pytest.approx([-80.0, -20.0, 10.0, 50.0])
    assert ax.get_extent(crs=ccrs.PlateCarree()) == pytest.approx(
        [-80.0, -20.0, 10.0, 50.0]
    )


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


@pytest.mark.parametrize("resolution", ["low", "medium", "high"])
def test_resolution_presets_apply_larger_font_sizes(monkeypatch, resolution) -> None:
    hp_map = np.ones(hp.nside2npix(2))
    monkeypatch.setitem(plotlib.RESOLUTION_PRESETS, resolution, (8, 16))
    fig = mollweide(
        hp_map,
        resolution=resolution,
        title="Readable title",
        show_gridlines=False,
        add_colorbar=True,
    )

    ax, colorbar_ax = fig.axes
    base_size = FONT_SIZE_PRESETS[resolution]
    assert fig.dpi == DPI_PRESETS[resolution]
    assert getattr(fig, "_skyplot_payload")["dpi"] == DPI_PRESETS[resolution]
    assert ax.title.get_fontsize() == pytest.approx(1.2 * base_size)
    assert colorbar_ax.xaxis.label.get_size() == pytest.approx(base_size)
    assert colorbar_ax.xaxis.get_ticklabels()[0].get_size() == pytest.approx(0.85 * base_size)


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


def test_equidistantconic_does_not_record_unsupported_cutoff() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    fig = equidistantconic(
        hp_map,
        n_theta=12,
        n_phi=24,
        extent=(-100.0, 20.0, -80.0, -10.0),
    )

    payload = getattr(fig, "_skyplot_payload")
    assert "cutoff" not in payload["projection_kwargs"]


def test_equidistantconic_rejects_unsupported_cutoff() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside))

    with pytest.raises(ValueError, match="does not support 'cutoff'"):
        equidistantconic(
            hp_map,
            n_theta=12,
            n_phi=24,
            extent=(-100.0, 20.0, -80.0, -10.0),
            projection_kwargs={"cutoff": -50.0},
        )


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
