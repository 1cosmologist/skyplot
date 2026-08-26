#######################################################################
# This file is a part of SkyPlot
#
# SkyPlot
# Copyright (C) 2026  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about SkyPlot please visit 
# <https://github.com/1cosmologist/skyplot> or contact Shamik Ghosh 
# at shamik@lbl.gov
#
#########################################################################

"""Matplotlib + Cartopy visualization routines for HEALPix sky maps."""

from __future__ import annotations
from importlib import import_module
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure

from importlib import resources

from .sampling import make_theta_phi_grid, sample_at_angles, sample_full_sky

_SUPPORTED_PROJECTIONS_ORDER = (
    "mollweide",
    "orthographic",
    "platecarree",
    "equidistantconic",
    "gnomonic",
)


def _get_cartopy_crs_module() -> Any:
    """Import cartopy.crs lazily and return the module object."""
    try:
        return import_module("cartopy.crs")
    except Exception as exc:
        raise ImportError(
            "cartopy is required for skyplot projections. Install with `pip install cartopy`."
        ) from exc


AVAILABLE_PROJECTIONS = _SUPPORTED_PROJECTIONS_ORDER

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "low": (480, 960),
    "medium": (720, 1440),
    "high": (1440, 2880),
}

_last_figure: Figure | None = None


def _get_last_figure() -> Figure | None:
    """Return the most recently created skyplot figure, if any."""
    return _last_figure


def _validate_extent(extent: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    """Validate geographic extent as ``(lon_min, lon_max, lat_min, lat_max)``."""
    if extent is None:
        return None
    if isinstance(extent, (str, bytes)):
        raise ValueError("extent must be a 4-element sequence: (lon_min, lon_max, lat_min, lat_max).")
    try:
        n_items = len(extent)
    except TypeError as exc:
        raise ValueError("extent must be a 4-element sequence: (lon_min, lon_max, lat_min, lat_max).") from exc
    if n_items != 4:
        raise ValueError("extent must be a 4-element sequence: (lon_min, lon_max, lat_min, lat_max).")

    lon_min, lon_max, lat_min, lat_max = map(float, extent)
    if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
        raise ValueError("extent values must be finite numbers.")
    if lat_min < -90.0 or lat_max > 90.0:
        raise ValueError("extent latitude bounds must be within [-90, 90].")
    if lat_min >= lat_max:
        raise ValueError("extent requires lat_min < lat_max.")
    if lon_min >= lon_max:
        raise ValueError("extent requires lon_min < lon_max.")
    return lon_min, lon_max, lat_min, lat_max


def _resolve_input_map_and_wcs(
    map_data: np.ndarray,
    wcs: Any | None,
) -> tuple[np.ndarray, Any | None]:
    """Resolve map input to ndarray and optional WCS metadata.

    Rules
    -----
    - 1D input is treated as a HEALPix map.
    - 2D input is treated as a WCS map and requires either an explicit
      ``wcs`` argument or a ``.wcs`` attribute on the input object.
    """
    data_arr = np.asarray(map_data)

    if data_arr.ndim == 1:
        return data_arr, None

    if data_arr.ndim != 2:
        raise ValueError("map input must be 1D (HEALPix) or 2D (WCS-backed image).")

    resolved_wcs = wcs if wcs is not None else getattr(map_data, "wcs", None)
    if resolved_wcs is None:
        raise ValueError(
            "2D map input requires WCS metadata via `wcs=` or a `.wcs` attribute on the input object."
        )
    return data_arr, resolved_wcs


def _sample_wcs_map(
    data: np.ndarray,
    *,
    wcs: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    interpolate: bool,
) -> np.ndarray:
    """Sample a 2D WCS-backed map at lon/lat (degrees) positions."""
    if not hasattr(wcs, "all_world2pix"):
        raise ValueError("Provided wcs object must implement all_world2pix(...).")

    nrows, ncols = data.shape
    world = np.column_stack([lon.reshape(-1), lat.reshape(-1)])
    pix_raw = wcs.all_world2pix(world, 0)

    # Support both Nx2 array returns and tuple-of-arrays returns.
    if isinstance(pix_raw, tuple):
        if len(pix_raw) != 2:
            raise ValueError("wcs.all_world2pix must return pixel x/y coordinates.")
        x = np.asarray(pix_raw[0], dtype=float).reshape(-1)
        y = np.asarray(pix_raw[1], dtype=float).reshape(-1)
    else:
        pix = np.asarray(pix_raw, dtype=float)
        if pix.ndim != 2 or pix.shape[1] != 2:
            raise ValueError("wcs.all_world2pix must return an array with shape (N, 2).")
        x = pix[:, 0]
        y = pix[:, 1]

    sampled = np.full(x.shape, np.nan, dtype=float)

    if interpolate:
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        valid = (x0 >= 0) & (y0 >= 0) & (x1 < ncols) & (y1 < nrows)
        if np.any(valid):
            xv = x[valid]
            yv = y[valid]
            x0v = x0[valid]
            y0v = y0[valid]
            x1v = x1[valid]
            y1v = y1[valid]

            wx = xv - x0v
            wy = yv - y0v

            v00 = data[y0v, x0v]
            v10 = data[y0v, x1v]
            v01 = data[y1v, x0v]
            v11 = data[y1v, x1v]

            sampled[valid] = (
                (1.0 - wx) * (1.0 - wy) * v00
                + wx * (1.0 - wy) * v10
                + (1.0 - wx) * wy * v01
                + wx * wy * v11
            )
    else:
        xi = np.rint(x).astype(int)
        yi = np.rint(y).astype(int)
        valid = (xi >= 0) & (yi >= 0) & (xi < ncols) & (yi < nrows)
        if np.any(valid):
            sampled[valid] = data[yi[valid], xi[valid]]

    return sampled.reshape(lon.shape)

def _as_listed_cmap(cmap_obj: Any, *, n: int = 256) -> ListedColormap:
    """Convert a colormap-like object to a Matplotlib ListedColormap."""
    arr: Any
    if callable(cmap_obj):
        arr = np.asarray(cmap_obj(np.linspace(0.0, 1.0, n)))
    elif hasattr(cmap_obj, "colors"):
        arr = np.asarray(cmap_obj.colors)
    else:
        arr = np.asarray(cmap_obj)

    if arr.ndim == 2 and arr.shape[0] in (3, 4) and arr.shape[1] > 4:
        arr = arr.T

    if arr.ndim != 2:
        raise ValueError("Resolved colormap could not be converted to a 2D color array.")

    if arr.shape[1] not in (3, 4):
        if arr.shape[1] > 4:
            arr = arr[:, :4]
        else:
            raise ValueError("Resolved colormap must provide RGB or RGBA colors.")

    arr = arr.astype(float)
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    return ListedColormap(arr)


def _resolve_cmap(cmap: str | Sequence[Any]) -> str | Colormap:
    """Resolve cmap input to a Matplotlib colormap."""
    if not isinstance(cmap, str):
        return _as_listed_cmap(cmap)

    try:
        plt.get_cmap(cmap)
        return cmap
    except Exception:
        pass

    if cmap in ["planck", "planck_log"]:
        cmap_path = resources.files("skyplot.data").joinpath(f"{cmap}.dat")
        return ListedColormap(np.loadtxt(cmap_path) / 255.0, cmap)

    try:
        cm = import_module("colormaps")
    except Exception as exc:
        raise ValueError(
            f"Unknown cmap '{cmap}'. It is not a Matplotlib cmap and colormaps is unavailable."
        ) from exc

    cmap_obj: Any | None = None
    if hasattr(cm, "get_cmap"):
        try:
            cmap_obj = cm.get_cmap(cmap)
        except Exception:
            cmap_obj = None

    if cmap_obj is None and hasattr(cm, cmap):
        cmap_obj = getattr(cm, cmap)

    if cmap_obj is None and hasattr(cm, "cmaps"):
        cmaps = getattr(cm, "cmaps")
        try:
            cmap_obj = cmaps[cmap]
        except Exception:
            cmap_obj = None

    if cmap_obj is None:
        raise ValueError(f"Unknown cmap '{cmap}' for both Matplotlib and colormaps package.")

    return _as_listed_cmap(cmap_obj)


def _resolve_gnomonic_center(center: Sequence[float]) -> tuple[float, float]:
    """Validate and normalize gnomonic center as (lon_deg, lat_deg)."""
    if isinstance(center, (str, bytes)):
        raise ValueError("center must be a 2-element sequence: (lon_deg, lat_deg).")
    try:
        n_items = len(center)
    except TypeError as exc:
        raise ValueError("center must be a 2-element sequence: (lon_deg, lat_deg).") from exc
    if n_items != 2:
        raise ValueError("center must be a 2-element sequence: (lon_deg, lat_deg).")

    lon_deg, lat_deg = map(float, center)
    if not np.isfinite([lon_deg, lat_deg]).all():
        raise ValueError("center values must be finite numbers.")
    if lat_deg < -90.0 or lat_deg > 90.0:
        raise ValueError("center latitude must be within [-90, 90].")
    return lon_deg, lat_deg


def _gnomonic_inverse(
    *,
    lon0_deg: float,
    lat0_deg: float,
    x_deg: np.ndarray,
    y_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tangent-plane offsets (degrees) to lon/lat (degrees)."""
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    x = np.radians(x_deg)
    y = np.radians(y_deg)

    rho = np.sqrt(x * x + y * y)
    c = np.arctan(rho)

    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    rho_safe = np.where(rho == 0.0, 1.0, rho)

    lat = np.arcsin(cos_c * sin_lat0 + (y * sin_c * cos_lat0) / rho_safe)
    lon = lon0 + np.arctan2(
        x * sin_c,
        rho_safe * cos_lat0 * cos_c - y * sin_lat0 * sin_c,
    )

    at_center = rho == 0.0
    if np.any(at_center):
        lon = np.where(at_center, lon0, lon)
        lat = np.where(at_center, lat0, lat)

    lon_deg = np.degrees(lon)
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
    lat_deg = np.degrees(lat)
    return lon_deg, lat_deg


def add_gridlines(
    ax: Any,
    *,
    color: str = "black",
    linestyle: str = "-",
    linewidth: float = 0.2,
    lon_gridline_spacing_deg: float = 30.,
    lat_gridline_spacing_deg: float = 30.,
    alpha: float = 1.0,
) -> Any:
    """Add and customize Cartopy gridlines on an existing GeoAxes.

    Parameters
    ----------
    ax : Any
        Existing Cartopy GeoAxes object.
    color : str, optional
        Gridline color.
    linestyle : str, optional
        Gridline linestyle.
    linewidth : float, optional
        Gridline line width.
    lon_gridline_spacing_deg : float, optional
        Longitude gridline separation in degrees.
    lat_gridline_spacing_deg : float, optional
        Latitude gridline separation in degrees.
    alpha : float, optional
        Gridline alpha value.

    Returns
    -------
    Any
        The Cartopy gridliner object returned by ``ax.gridlines``.

    Raises
    ------
    ValueError
        If ``linewidth`` or gridline spacings are non-positive.
    """
    if linewidth <= 0:
        raise ValueError("linewidth must be positive.")
    if lon_gridline_spacing_deg <= 0:
        raise ValueError("lon_gridline_spacing_deg must be positive.")
    if lat_gridline_spacing_deg <= 0:
        raise ValueError("lat_gridline_spacing_deg must be positive.")

    ccrs = _get_cartopy_crs_module()
    lon_ticks = np.arange(
        -180.0 + lon_gridline_spacing_deg,
        180.0 + 1e-6,
        lon_gridline_spacing_deg,
    )
    lat_ticks = np.arange(
        -90.0 + lat_gridline_spacing_deg,
        90.0 + 1e-6,
        lat_gridline_spacing_deg,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(lat_ticks)
    return gl


def _plot_with_projection(
    map_data: np.ndarray,
    *,
    projection_name: str,
    projection_factory: Callable[..., Any],
    projection_kwargs: dict[str, Any] | None = None,
    extent: Sequence[float] | None = None,
    wcs: Any | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> Figure:
    """Shared renderer for projection-specific public plotting functions."""
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if len(figsize) != 2 or figsize[0] <= 0.0 or figsize[1] <= 0.0:
        raise ValueError("figsize must be a two-element tuple of positive values.")

    if resolution is not None:
        if resolution not in RESOLUTION_PRESETS:
            supported = ", ".join(sorted(RESOLUTION_PRESETS))
            raise ValueError(f"Unsupported resolution '{resolution}'. Choose one of: {supported}")
        n_theta, n_phi = RESOLUTION_PRESETS[resolution]

    validated_extent = _validate_extent(extent)
    projection_kwargs = {} if projection_kwargs is None else dict(projection_kwargs)
    data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
    ccrs = _get_cartopy_crs_module()
    resolved_cmap = _resolve_cmap(cmap)

    if data_arr.ndim == 1:
        lon, lat, values = sample_full_sky(
            data_arr,
            n_theta=n_theta,
            n_phi=n_phi,
            nest=nest,
            interpolate=interpolate,
        )
    else:
        theta, phi = make_theta_phi_grid(n_theta=n_theta, n_phi=n_phi)
        lon = np.degrees(phi)
        lon = ((lon + 180.0) % 360.0) - 180.0
        lat = 90.0 - np.degrees(theta)
        values = _sample_wcs_map(
            data_arr,
            wcs=resolved_wcs,
            lon=lon,
            lat=lat,
            interpolate=interpolate,
        )

    # Astronomy-style orientation: phi increases from right to left.
    lon = -lon
    lon = ((lon + 180.0) % 360.0) - 180.0
    sort_idx = np.argsort(lon[0, :])
    lon = lon[:, sort_idx]
    lat = lat[:, sort_idx]
    values = values[:, sort_idx]

    # Keep latitude monotonic south->north for stable projected pcolormesh.
    lat_sort_idx = np.argsort(lat[:, 0])
    lon = lon[lat_sort_idx, :]
    lat = lat[lat_sort_idx, :]
    values = values[lat_sort_idx, :]

    created_fig = ax is None
    if ax is None:
        map_crs = projection_factory(**projection_kwargs)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={"projection": map_crs})
        if validated_extent is None:
            ax.set_global()
        else:
            ax.set_extent(validated_extent, crs=ccrs.PlateCarree())
    else:
        fig = ax.figure

    applied_gridline_kwargs: dict[str, Any] = {
        "color": "black",
        "linestyle": "-",
        "linewidth": 0.2,
        "lon_gridline_spacing_deg": 30.0,
        "lat_gridline_spacing_deg": 30.0,
        "alpha": 1.0,
    }
    if gridline_kwargs is not None:
        applied_gridline_kwargs.update(gridline_kwargs)

    if show_gridlines:
        add_gridlines(ax, **applied_gridline_kwargs)

    mesh_kwargs: dict[str, Any] = {
        "shading": "nearest",
        "rasterized": True,
    }
    if pcolormesh_kwargs is not None:
        mesh_kwargs.update(pcolormesh_kwargs)

    quad = ax.pcolormesh(
        lon,
        lat,
        values,
        transform=ccrs.PlateCarree(),
        cmap=resolved_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        **mesh_kwargs,
    )

    if add_colorbar:
        cbar = fig.colorbar(
            quad,
            ax=ax,
            orientation="horizontal",
            pad=0.06,
            fraction=0.035,
            aspect=40,
        )
        cbar.set_label(colorbar_title)

    if title:
        ax.set_title(title)

    fig.tight_layout()

    # Used by save_figure for lightweight JSON export.
    fig._skyplot_payload = {  # type: ignore[attr-defined]
        "backend": "matplotlib-cartopy",
        "projection": projection_name,
        "projection_kwargs": projection_kwargs,
        "extent": list(validated_extent) if validated_extent is not None else None,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "vmin": vmin,
        "vmax": vmax,
        "norm": str(norm) if norm is not None else None,
        "cmap": str(cmap),
        "colorbar_title": colorbar_title,
        "title": title,
        "show_gridlines": show_gridlines,
        "gridline_color": applied_gridline_kwargs["color"],
        "gridline_linestyle": applied_gridline_kwargs["linestyle"],
        "gridline_linewidth": applied_gridline_kwargs["linewidth"],
        "lon_gridline_spacing_deg": applied_gridline_kwargs["lon_gridline_spacing_deg"],
        "lat_gridline_spacing_deg": applied_gridline_kwargs["lat_gridline_spacing_deg"],
        "colorbar_orientation": "horizontal",
    }

    # Prevent matplotlib's Jupyter inline backend from auto-displaying this
    # figure a second time in addition to the one shown via the return value.
    if created_fig:
        plt.close(fig)

    global _last_figure
    _last_figure = fig

    return fig


def mollweide(
    map_data: np.ndarray,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    wcs: Any | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Mollweide projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        Input sky map. A 1D array is interpreted as a HEALPix map. A 2D array
        is interpreted as a WCS-backed image and requires ``wcs=`` or a
        ``.wcs`` attribute on the input object.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.Mollweide``.
    wcs : Any or None, optional
        WCS object used when ``map_data`` is 2D. Must implement
        ``all_world2pix``.
    ax : Any or None, optional
        Existing GeoAxes to draw into. Use this to overlay multiple maps on the
        same projection.
    n_theta, n_phi : int, optional
        Sampling grid shape when resampling map values to plotting coordinates.
    resolution : {"low", "medium", "high"} or None, optional
        Convenience preset overriding ``n_theta`` and ``n_phi``.
    nest : bool, optional
        HEALPix NEST ordering toggle for 1D HEALPix inputs.
    interpolate : bool, optional
        Enable interpolation during sampling.
    cmap : str or sequence, optional
        Colormap name or colormap-like values.
    vmin, vmax : float or None, optional
        Color scaling bounds.
    norm : str or matplotlib.colors.Normalize or None, optional
        Normalization applied to map values before colormapping, forwarded to
        ``ax.pcolormesh``. Accepts a registered Matplotlib scale name (e.g.
        ``"log"``) or a ``Normalize`` instance.
    colorbar_title : str, optional
        Label text for the colorbar.
    title : str or None, optional
        Axes title.
    show_gridlines : bool, optional
        Whether to draw geographic gridlines.
    gridline_kwargs : dict[str, Any] or None, optional
        Keyword arguments for :func:`add_gridlines`.
    pcolormesh_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments passed to ``ax.pcolormesh``. Useful for overlay
        styling (e.g. ``alpha``).
    add_colorbar : bool, optional
        Whether to add a colorbar for this layer.
    figsize : tuple[float, float], optional
        Figure size used only when creating a new figure.
    dpi : int, optional
        Figure DPI used only when creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    ccrs = _get_cartopy_crs_module()
    return _plot_with_projection(
        map_data,
        projection_name="mollweide",
        projection_factory=ccrs.Mollweide,
        projection_kwargs=projection_kwargs,
        wcs=wcs,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_title=colorbar_title,
        title=title,
        show_gridlines=show_gridlines,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar,
        figsize=figsize,
        dpi=dpi,
    )


def orthographic(
    map_data: np.ndarray,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    wcs: Any | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Orthographic projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        1D HEALPix map or 2D WCS-backed map.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.Orthographic``.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    n_theta, n_phi, resolution, nest, interpolate, cmap, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    ccrs = _get_cartopy_crs_module()
    return _plot_with_projection(
        map_data,
        projection_name="orthographic",
        projection_factory=ccrs.Orthographic,
        projection_kwargs=projection_kwargs,
        wcs=wcs,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_title=colorbar_title,
        title=title,
        show_gridlines=show_gridlines,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar,
        figsize=figsize,
        dpi=dpi,
    )


def platecarree(
    map_data: np.ndarray,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    extent: Sequence[float] | None = None,
    wcs: Any | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy PlateCarree projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        1D HEALPix map or 2D WCS-backed map.
    extent: sequence[float] or None, optional
        Geographic render window as ``(lon_min, lon_max, lat_min, lat_max)``.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.PlateCarree``.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    n_theta, n_phi, resolution, nest, interpolate, cmap, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    validated_extent = _validate_extent(extent)
    
    ccrs = _get_cartopy_crs_module()
    return _plot_with_projection(
        map_data,
        projection_name="platecarree",
        projection_factory=ccrs.PlateCarree,
        projection_kwargs=projection_kwargs,
        extent=validated_extent,
        wcs=wcs,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_title=colorbar_title,
        title=title,
        show_gridlines=show_gridlines,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar,
        figsize=figsize,
        dpi=dpi,
    )


def equidistantconic(
    map_data: np.ndarray,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    extent: Sequence[float] | None = None,
    wcs: Any | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy EquidistantConic projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        1D HEALPix map or 2D WCS-backed map.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments for ``cartopy.crs.EquidistantConic``. The wrapper
        accepts ``cutoff`` and applies it when supported by the CRS instance.
    extent : sequence[float] or None, optional
        Geographic render window as ``(lon_min, lon_max, lat_min, lat_max)``.
        When provided, missing ``central_longitude``/``central_latitude`` are
        inferred from the extent center.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    n_theta, n_phi, resolution, nest, interpolate, cmap, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    ccrs = _get_cartopy_crs_module()
    validated_extent = _validate_extent(extent)
    resolved_projection_kwargs = {} if projection_kwargs is None else dict(projection_kwargs)

    # Avoid horizontal clipping by default. Keep explicit user values.
    resolved_projection_kwargs.setdefault("cutoff", -90.0)

    if validated_extent is not None:
        lon_min, lon_max, lat_min, lat_max = validated_extent
        resolved_projection_kwargs.setdefault("central_longitude", 0.5 * (lon_min + lon_max))
        resolved_projection_kwargs.setdefault("central_latitude", 0.5 * (lat_min + lat_max))

    def _equidistantconic_factory(**kwargs: Any) -> Any:
        factory_kwargs = dict(kwargs)
        cutoff_value = factory_kwargs.pop("cutoff", None)
        crs = ccrs.EquidistantConic(**factory_kwargs)
        if cutoff_value is not None and hasattr(crs, "cutoff"):
            try:
                setattr(crs, "cutoff", float(cutoff_value))
            except Exception:
                pass
        return crs

    return _plot_with_projection(
        map_data,
        projection_name="equidistantconic",
        projection_factory=_equidistantconic_factory,
        projection_kwargs=resolved_projection_kwargs,
        extent=validated_extent,
        wcs=wcs,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_title=colorbar_title,
        title=title,
        show_gridlines=show_gridlines,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar,
        figsize=figsize,
        dpi=dpi,
    )


def gnomonic(
    map_data: np.ndarray,
    *,
    center: Sequence[float] = (0.0, 0.0),
    xsize: int = 500,
    ysize: int = 500,
    n_theta: int | None = None,
    n_phi: int | None = None,
    pixel_size_arcmin: float = 5.0,
    wcs: Any | None = None,
    ax: Any | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    add_colorbar: bool = True,
    astro_orientation: bool = True,
    figsize: tuple[float, float] = (6.5, 6.5),
    dpi: int = 300,
    imshow_kwargs: dict[str, Any] | None = None,
) -> Figure:
    """Plot a local gnomonic view using Matplotlib ``imshow`` only.

    Parameters
    ----------
    map_data : numpy.ndarray
        Input sky map. A 1D array is interpreted as a HEALPix map. A 2D array
        is interpreted as a WCS-backed image and requires ``wcs=`` or a
        ``.wcs`` attribute on the input object.
    center : sequence[float], optional
        Tangent point as ``(lon_deg, lat_deg)``.
    xsize, ysize : int, optional
        Patch size in pixels along x/y.
    n_theta, n_phi : int or None, optional
        Convenience overrides for ``ysize``/``xsize`` respectively, so
        :func:`gnomonic` can be called interchangeably with the other
        projection functions.
    pixel_size_arcmin : float, optional
        Angular size per pixel in arcminutes.
    wcs : Any or None, optional
        WCS object used when ``map_data`` is 2D. Must implement
        ``all_world2pix``.
    ax : Any or None, optional
        Existing Axes to draw into for overlay workflows.
    nest, interpolate, cmap, vmin, vmax, colorbar_title, title, add_colorbar,
    figsize, dpi
        Same behavior as :func:`mollweide` where applicable.
    norm : str or matplotlib.colors.Normalize or None, optional
        Normalization applied to map values before colormapping, forwarded to
        ``ax.imshow``. Accepts a registered Matplotlib scale name (e.g.
        ``"log"``) or a ``Normalize`` instance.
    astro_orientation : bool, optional
        If True, invert the x-axis so longitude increases to the left.
    imshow_kwargs : dict[str, Any] or None, optional
        Extra keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if len(figsize) != 2 or figsize[0] <= 0.0 or figsize[1] <= 0.0:
        raise ValueError("figsize must be a two-element tuple of positive values.")
    if n_theta is not None:
        ysize = n_theta
    if n_phi is not None:
        xsize = n_phi
    if xsize <= 0 or ysize <= 0:
        raise ValueError("xsize and ysize must be positive integers.")
    if pixel_size_arcmin <= 0.0:
        raise ValueError("pixel_size_arcmin must be positive.")

    lon0_deg, lat0_deg = _resolve_gnomonic_center(center)
    data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
    resolved_cmap = _resolve_cmap(cmap)

    x_pix = np.arange(xsize, dtype=float) - 0.5 * (xsize - 1)
    y_pix = np.arange(ysize, dtype=float) - 0.5 * (ysize - 1)
    x_arcmin = x_pix * float(pixel_size_arcmin)
    y_arcmin = y_pix * float(pixel_size_arcmin)

    x_deg_grid, y_deg_grid = np.meshgrid(x_arcmin / 60.0, y_arcmin / 60.0)
    lon_deg, lat_deg = _gnomonic_inverse(
        lon0_deg=lon0_deg,
        lat0_deg=lat0_deg,
        x_deg=x_deg_grid,
        y_deg=y_deg_grid,
    )

    if data_arr.ndim == 1:
        values = sample_at_angles(
            data_arr,
            lon_deg,
            lat_deg,
            nest=nest,
            lonlat=True,
            interpolate=interpolate,
        )
    else:
        values = _sample_wcs_map(
            data_arr,
            wcs=resolved_wcs,
            lon=lon_deg,
            lat=lat_deg,
            interpolate=interpolate,
        )

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    draw_kwargs: dict[str, Any] = {
        "origin": "lower",
        "interpolation": "nearest",
    }
    if imshow_kwargs is not None:
        draw_kwargs.update(imshow_kwargs)

    half_pix = 0.5 * float(pixel_size_arcmin)
    extent = [
        float(x_arcmin[0] - half_pix),
        float(x_arcmin[-1] + half_pix),
        float(y_arcmin[0] - half_pix),
        float(y_arcmin[-1] + half_pix),
    ]

    im = ax.imshow(
        values,
        cmap=resolved_cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extent=extent,
        **draw_kwargs,
    )

    if astro_orientation:
        ax.invert_xaxis()

    ax.set_xlabel("Delta lon [arcmin]")
    ax.set_ylabel("Delta lat [arcmin]")

    if add_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            fraction=0.05,
            aspect=40,
        )
        cbar.set_label(colorbar_title)

    if title:
        ax.set_title(title)

    fig.tight_layout()

    fig._skyplot_payload = {  # type: ignore[attr-defined]
        "backend": "matplotlib-gnomonic",
        "projection": "gnomonic",
        "projection_kwargs": {
            "center": [lon0_deg, lat0_deg],
            "xsize": int(xsize),
            "ysize": int(ysize),
            "pixel_size_arcmin": float(pixel_size_arcmin),
            "astro_orientation": bool(astro_orientation),
        },
        "extent": [float(v) for v in extent],
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "vmin": vmin,
        "vmax": vmax,
        "norm": str(norm) if norm is not None else None,
        "cmap": str(cmap),
        "colorbar_title": colorbar_title,
        "title": title,
        "show_gridlines": False,
        "colorbar_orientation": "horizontal",
    }

    # Prevent matplotlib's Jupyter inline backend from auto-displaying this
    # figure a second time in addition to the one shown via the return value.
    if created_fig:
        plt.close(fig)

    global _last_figure
    _last_figure = fig

    return fig
