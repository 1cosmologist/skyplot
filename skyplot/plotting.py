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

"""Public plotting API for SkyPlot.

Implementation details live in :mod:`skyplot.plotlib`; this module keeps the
stable, user-facing plotting surface in one small place.
"""

from typing import Any, Literal, Sequence

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from . import plotlib as _plotlib
from .plotlib import (
    AVAILABLE_PROJECTIONS,
    FONT_SIZE_PRESETS,
    DPI_PRESETS,
    RESOLUTION_PRESETS,
    plot_with_projection,
    _gnomonic_inverse,
    _get_cartopy_crs_module,
    _resolve_gnomonic_center,
    _resolve_input_map_and_wcs,
    _resolve_cmap,
    _sample_wcs_map,
    _validate_extent,
    _with_bad_color,
)
from .sampling import sample_at_angles


def add_gridlines(
    ax: Any,
    *,
    color: str = "black",
    linestyle: str = "-",
    linewidth: float = 0.2,
    lon_gridline_spacing_deg: float = 30.0,
    lat_gridline_spacing_deg: float = 30.0,
    alpha: float = 1.0,
) -> Any:
    """Add Cartopy gridlines to an existing GeoAxes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes receiving the gridlines; required.
    color : str, default="black"
        Gridline color.
    linestyle : str, default="-"
        Matplotlib gridline style.
    linewidth : float, default=0.2
        Positive gridline width in points.
    lon_gridline_spacing_deg : float, default=30.0
        Positive longitude separation in degrees.
    lat_gridline_spacing_deg : float, default=30.0
        Positive latitude separation in degrees.
    alpha : float, default=1.0
        Gridline opacity.
    """
    if linewidth <= 0:
        raise ValueError("linewidth must be positive.")
    if lon_gridline_spacing_deg <= 0:
        raise ValueError("lon_gridline_spacing_deg must be positive.")
    if lat_gridline_spacing_deg <= 0:
        raise ValueError("lat_gridline_spacing_deg must be positive.")

    ccrs = _get_cartopy_crs_module()
    lon_ticks = np.arange(-180.0 + lon_gridline_spacing_deg, 180.0 + 1e-6, lon_gridline_spacing_deg)
    lat_ticks = np.arange(-90.0 + lat_gridline_spacing_deg, 90.0 + 1e-6, lat_gridline_spacing_deg)
    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=linewidth,
        color=color, linestyle=linestyle, alpha=alpha,
    )
    gridliner.xlocator = mticker.FixedLocator(lon_ticks)
    gridliner.ylocator = mticker.FixedLocator(lat_ticks)
    return gridliner


def _gnomonic_grid_spacing(span_deg: float) -> float:
    """Choose a readable angular grid interval for a local sky view."""
    candidates = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0])
    target = max(span_deg / 4.0, candidates[0])
    return float(candidates[np.argmin(np.abs(np.log(candidates / target)))])


def _add_gnomonic_gridlines(
    ax: Any,
    *,
    x_arcmin: np.ndarray,
    y_arcmin: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    lon0_deg: float,
    lat0_deg: float,
    gridline_kwargs: dict[str, Any] | None,
) -> None:
    """Add curved longitude and latitude graticules to a gnomonic axes."""
    lon_unwrapped = lon0_deg + ((lon_deg - lon0_deg + 180.0) % 360.0) - 180.0
    lon_min, lon_max = float(np.nanmin(lon_unwrapped)), float(np.nanmax(lon_unwrapped))
    lat_min, lat_max = float(np.nanmin(lat_deg)), float(np.nanmax(lat_deg))
    lon_step = _gnomonic_grid_spacing(lon_max - lon_min)
    lat_step = _gnomonic_grid_spacing(lat_max - lat_min)
    lon_levels = np.arange(np.ceil(lon_min / lon_step) * lon_step, lon_max, lon_step)
    lat_levels = np.arange(np.ceil(lat_min / lat_step) * lat_step, lat_max, lat_step)

    style: dict[str, Any] = {
        "colors": "black", "linestyles": "-", "linewidths": 0.2, "alpha": 1.0,
    }
    if gridline_kwargs is not None:
        style.update(gridline_kwargs)
    for singular, plural in (("color", "colors"), ("linestyle", "linestyles"), ("linewidth", "linewidths")):
        if singular in style:
            style[plural] = style.pop(singular)
    # A gnomonic graticule is curved. Rectangular-axis ticks would represent
    # coordinates only along the central row/column and are therefore wrong at
    # the plot edges. Keep the coordinate contours unlabeled.
    if lon_levels.size:
        ax.contour(x_arcmin, y_arcmin, lon_unwrapped, levels=lon_levels, **style)
    if lat_levels.size:
        ax.contour(x_arcmin, y_arcmin, lat_deg, levels=lat_levels, **style)
    ax.set_xticks([])
    ax.set_yticks([])


def mollweide(
    map_data: Any,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    coordinate_frame: str | None = None,
    coordinate_transform: Sequence[str] | None = None,
    wcs: Any | None = None,
    world_axis_mapping: Sequence[int] | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "roma_r",
    badvalue: float | None = hp.UNSEEN,
    badcolor: Any = "grey",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    plot_mode: Literal["map", "overlay_mask", "vector_field"] = "map",
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float | None = None,
    zorder: float | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Mollweide projection.

    Parameters
    ----------
    map_data : numpy.ndarray or sequence
        Required 1D HEALPix map or 2D WCS-backed image. In vector mode, a
        two-element ``(U, V)`` sequence of matching component maps.
    projection_kwargs : dict or None, default=None
        Keyword arguments passed to Cartopy's ``Mollweide`` CRS.
    coordinate_frame : str or None, default=None
        Metadata label for the source coordinate frame.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` Astropy sampling transform. Common
        frame names are ``"icrs"`` (equatorial), ``"galactic"``, and
        ``"geocentrictrueecliptic"`` (ecliptic); for example,
        ``("galactic", "icrs")`` displays a Galactic map in equatorial
        coordinates.
    wcs : object or None, default=None
        WCS for a 2D input; its ``all_world2pix`` method is required.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit ``(longitude_axis, latitude_axis)`` WCS world-axis mapping.
    ax : GeoAxes or None, default=None
        Existing axes for an overlay; ``None`` creates an axes.
    n_theta, n_phi : int, defaults=720, 1440
        Colatitude and longitude sampling-grid sizes.
    resolution : {"low", "medium", "high"} or None, default=None
        Preset that overrides ``n_theta`` and ``n_phi`` and selects a larger
        readable font size (14, 16, or 18 points) and a 120, 200, or 300 DPI
        new figure for low, medium, or high.
    nest : bool, default=False
        Treat a 1D HEALPix map as NEST ordered.
    interpolate : bool, default=True
        Interpolate sampled values instead of nearest-pixel lookup.
    cmap : str or sequence, default="roma_r"
        Matplotlib or ``colormaps`` colormap specification.
    badvalue : float or None, default=healpy.UNSEEN
        Input sentinel converted to missing data; ``None`` disables sentinel matching.
    badcolor : color, default="grey"
        Color for missing, non-finite, or sentinel samples.
    vmin, vmax : float or None, defaults=None, None
        Optional lower and upper color-scale limits.
    norm : str or matplotlib.colors.Normalize or None, default=None
        Matplotlib normalization forwarded to ``pcolormesh``.
    colorbar_title : str, default="Map value"
        Label for the optional colorbar.
    title : str or None, default=None
        Axes title.
    show_gridlines : bool, default=True
        Draw Cartopy gridlines.
    plot_mode : {"map", "overlay_mask", "vector_field"}, default="map"
        Render a scalar map, a binary-mask overlay, or a transparent vector
        overlay. Vector mode requires a two-element ``(U, V)`` map sequence
        and ``ax=`` from a previously rendered magnitude map.
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
        This legacy switch is equivalent to ``plot_mode="overlay_mask"``.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float or None, default=None
        Layer opacity. Mask overlays default to ``0.25`` when omitted; an
        explicitly supplied value is used unchanged.
    zorder : float or None, default=None
        Artist drawing order. Scalar maps default to 1, vector fields to 2,
        and mask overlays to 3.
    gridline_kwargs : dict or None, default=None
        Overrides for gridline color, style, width, spacing, or opacity.
    pcolormesh_kwargs : dict or None, default=None
        Extra keyword arguments passed to ``GeoAxes.pcolormesh``.
    vector_kwargs : dict or None, default=None
        Options for the vector artist in vector mode. Set ``method`` to
        ``"streamplot"`` (default) or ``"quiver"``. ``cmap``, ``vmin``,
        ``vmax``, and ``norm`` are ignored in vector mode.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    figsize : tuple[float, float], default=(8.0, 5.0)
        Figure size in inches when creating axes.
    dpi : int, default=300
        Figure resolution when creating axes.
    """
    ccrs = _get_cartopy_crs_module()
    return plot_with_projection(
        map_data, projection_name="mollweide", projection_factory=ccrs.Mollweide,
        projection_kwargs=projection_kwargs, coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform, wcs=wcs,
        world_axis_mapping=world_axis_mapping, ax=ax, n_theta=n_theta, n_phi=n_phi,
        resolution=resolution, nest=nest, interpolate=interpolate, cmap=cmap,
        badvalue=badvalue, badcolor=badcolor, vmin=vmin, vmax=vmax, norm=norm,
        colorbar_title=colorbar_title, title=title, show_gridlines=show_gridlines,
        gridline_adder=add_gridlines,
        plot_mode=plot_mode, overlay_mask=overlay_mask, overlay_color=overlay_color, alpha=alpha,
        zorder=zorder,
        gridline_kwargs=gridline_kwargs, pcolormesh_kwargs=pcolormesh_kwargs,
        vector_kwargs=vector_kwargs,
        add_colorbar=add_colorbar, figsize=figsize, dpi=dpi,
    )


def orthographic(
    map_data: Any,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    coordinate_frame: str | None = None,
    coordinate_transform: Sequence[str] | None = None,
    wcs: Any | None = None,
    world_axis_mapping: Sequence[int] | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "roma_r",
    badvalue: float | None = hp.UNSEEN,
    badcolor: Any = "grey",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    plot_mode: Literal["map", "overlay_mask", "vector_field"] = "map",
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float | None = None,
    zorder: float | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (5.5, 6.5),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Orthographic projection.

    Parameters
    ----------
    map_data : numpy.ndarray or sequence
        Required 1D HEALPix map or 2D WCS-backed image. In vector mode, a
        two-element ``(U, V)`` sequence of matching component maps.
    projection_kwargs : dict or None, default=None
        Keyword arguments passed to Cartopy's ``Orthographic`` CRS.
    coordinate_frame : str or None, default=None
        Metadata label for the source coordinate frame.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` Astropy sampling transform. Common
        frame names are ``"icrs"`` (equatorial), ``"galactic"``, and
        ``"geocentrictrueecliptic"`` (ecliptic).
    wcs : object or None, default=None
        WCS for a 2D map input.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit longitude/latitude WCS world-axis indices.
    ax : GeoAxes or None, default=None
        Existing axes for an overlay; ``None`` creates axes.
    n_theta, n_phi : int, defaults=720, 1440
        Colatitude and longitude sampling-grid sizes.
    resolution : {"low", "medium", "high"} or None, default=None
        Preset overriding both sampling-grid sizes and selecting a 14, 16, or
        18 point base font and 120, 200, or 300 DPI new figure for low,
        medium, or high, respectively.
    nest : bool, default=False
        Use HEALPix NEST ordering for a 1D map.
    interpolate : bool, default=True
        Interpolate sampled map values.
    cmap : str or sequence, default="roma_r"
        Colormap specification.
    badvalue : float or None, default=healpy.UNSEEN
        Missing-data sentinel; ``None`` disables sentinel matching.
    badcolor : color, default="grey"
        Missing-data color.
    vmin, vmax : float or None, defaults=None, None
        Optional color-scale limits.
    norm : str or Normalize or None, default=None
        Color normalization passed to ``pcolormesh``.
    colorbar_title : str, default="Map value"
        Optional colorbar label.
    title : str or None, default=None
        Optional axes title.
    show_gridlines : bool, default=True
        Draw geographic gridlines.
    plot_mode : {"map", "overlay_mask", "vector_field"}, default="map"
        Render a scalar map, a binary-mask overlay, or a transparent vector
        overlay. Vector mode requires a two-element ``(U, V)`` map sequence
        and ``ax=`` from a previously rendered magnitude map.
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
        This legacy switch is equivalent to ``plot_mode="overlay_mask"``.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float or None, default=None
        Layer opacity. Mask overlays default to ``0.25`` when omitted; an
        explicitly supplied value is used unchanged.
    zorder : float or None, default=None
        Artist drawing order. Scalar maps default to 1, vector fields to 2,
        and mask overlays to 3.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh keyword arguments.
    vector_kwargs : dict or None, default=None
        Options for the vector artist in vector mode. Set ``method`` to
        ``"streamplot"`` (default) or ``"quiver"``. Scalar color arguments
        are ignored in vector mode.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    figsize : tuple[float, float], default=(5.5, 6.5)
        New-figure size in inches.
    dpi : int, default=300
        New-figure resolution.
    """
    ccrs = _get_cartopy_crs_module()
    return plot_with_projection(
        map_data, projection_name="orthographic", projection_factory=ccrs.Orthographic,
        projection_kwargs=projection_kwargs, coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform, wcs=wcs,
        world_axis_mapping=world_axis_mapping, ax=ax, n_theta=n_theta, n_phi=n_phi,
        resolution=resolution, nest=nest, interpolate=interpolate, cmap=cmap,
        badvalue=badvalue, badcolor=badcolor, vmin=vmin, vmax=vmax, norm=norm,
        colorbar_title=colorbar_title, title=title, show_gridlines=show_gridlines,
        gridline_adder=add_gridlines,
        plot_mode=plot_mode, overlay_mask=overlay_mask, overlay_color=overlay_color, alpha=alpha,
        zorder=zorder,
        gridline_kwargs=gridline_kwargs, pcolormesh_kwargs=pcolormesh_kwargs,
        vector_kwargs=vector_kwargs,
        add_colorbar=add_colorbar, figsize=figsize, dpi=dpi,
    )


def platecarree(
    map_data: Any,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    extent: Sequence[float] | None = None,
    coordinate_frame: str | None = None,
    coordinate_transform: Sequence[str] | None = None,
    wcs: Any | None = None,
    world_axis_mapping: Sequence[int] | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "roma_r",
    badvalue: float | None = hp.UNSEEN,
    badcolor: Any = "grey",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    plot_mode: Literal["map", "overlay_mask", "vector_field"] = "map",
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float | None = None,
    zorder: float | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy PlateCarree projection.

    Parameters
    ----------
    map_data : numpy.ndarray or sequence
        Required 1D HEALPix map or 2D WCS-backed image. In vector mode, a
        two-element ``(U, V)`` sequence of matching component maps.
    projection_kwargs : dict or None, default=None
        Keyword arguments for Cartopy's ``PlateCarree`` CRS.
    extent : sequence[float] or None, default=None
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees for new axes;
        an existing ``ax`` retains its own extent.
    coordinate_frame : str or None, default=None
        Source-frame metadata label.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` Astropy sampling transform. Common
        frame names are ``"icrs"`` (equatorial), ``"galactic"``, and
        ``"geocentrictrueecliptic"`` (ecliptic).
    wcs : object or None, default=None
        WCS for 2D input.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit longitude/latitude WCS axes.
    ax : GeoAxes or None, default=None
        Existing overlay axes; ``None`` creates axes.
    n_theta, n_phi : int, defaults=720, 1440
        Sampling-grid dimensions.
    resolution : {"low", "medium", "high"} or None, default=None
        Sampling-grid preset and 14, 16, or 18 point base font for low,
        medium, or high, respectively; also selects 120, 200, or 300 DPI.
    nest : bool, default=False
        Use HEALPix NEST ordering.
    interpolate : bool, default=True
        Interpolate samples.
    cmap : str or sequence, default="roma_r"
        Colormap specification.
    badvalue : float or None, default=healpy.UNSEEN
        Missing-data sentinel; ``None`` disables it.
    badcolor : color, default="grey"
        Missing-data color.
    vmin, vmax : float or None, defaults=None, None
        Color-scale limits.
    norm : str or Normalize or None, default=None
        Mesh color normalization.
    colorbar_title : str, default="Map value"
        Colorbar label.
    title : str or None, default=None
        Axes title.
    show_gridlines : bool, default=True
        Draw gridlines.
    plot_mode : {"map", "overlay_mask", "vector_field"}, default="map"
        Render a scalar map, a binary-mask overlay, or a transparent vector
        overlay. Vector mode requires a two-element ``(U, V)`` map sequence
        and ``ax=`` from a previously rendered magnitude map.
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
        This legacy switch is equivalent to ``plot_mode="overlay_mask"``.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float or None, default=None
        Layer opacity. Mask overlays default to ``0.25`` when omitted; an
        explicitly supplied value is used unchanged.
    zorder : float or None, default=None
        Artist drawing order. Scalar maps default to 1, vector fields to 2,
        and mask overlays to 3.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh options.
    vector_kwargs : dict or None, default=None
        Options for the vector artist in vector mode. Set ``method`` to
        ``"streamplot"`` (default) or ``"quiver"``. Scalar color arguments
        are ignored in vector mode.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    figsize : tuple[float, float], default=(8.0, 5.0)
        New-figure size.
    dpi : int, default=300
        New-figure resolution.
    """
    validated_extent = _validate_extent(extent)
    ccrs = _get_cartopy_crs_module()
    return plot_with_projection(
        map_data, projection_name="platecarree", projection_factory=ccrs.PlateCarree,
        projection_kwargs=projection_kwargs, extent=validated_extent,
        coordinate_frame=coordinate_frame, coordinate_transform=coordinate_transform,
        wcs=wcs, world_axis_mapping=world_axis_mapping, ax=ax, n_theta=n_theta,
        n_phi=n_phi, resolution=resolution, nest=nest, interpolate=interpolate,
        cmap=cmap, badvalue=badvalue, badcolor=badcolor, vmin=vmin, vmax=vmax,
        norm=norm, colorbar_title=colorbar_title, title=title,
        show_gridlines=show_gridlines, gridline_adder=add_gridlines,
        plot_mode=plot_mode, overlay_mask=overlay_mask,
        overlay_color=overlay_color, alpha=alpha, zorder=zorder,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs, add_colorbar=add_colorbar,
        vector_kwargs=vector_kwargs,
        figsize=figsize, dpi=dpi,
    )


def equidistantconic(
    map_data: Any,
    *,
    projection_kwargs: dict[str, Any] | None = None,
    extent: Sequence[float] | None = None,
    coordinate_frame: str | None = None,
    coordinate_transform: Sequence[str] | None = None,
    wcs: Any | None = None,
    world_axis_mapping: Sequence[int] | None = None,
    ax: Any | None = None,
    n_theta: int = 720,
    n_phi: int = 1440,
    resolution: Literal["low", "medium", "high"] | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "roma_r",
    badvalue: float | None = hp.UNSEEN,
    badcolor: Any = "grey",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = True,
    plot_mode: Literal["map", "overlay_mask", "vector_field"] = "map",
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float | None = None,
    zorder: float | None = None,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy EquidistantConic projection.

    Parameters
    ----------
    map_data : numpy.ndarray or sequence
        Required 1D HEALPix map or 2D WCS-backed image. In vector mode, a
        two-element ``(U, V)`` sequence of matching component maps.
    projection_kwargs : dict or None, default=None
        Cartopy ``EquidistantConic`` options; ``cutoff`` is unsupported.
    extent : sequence[float] or None, default=None
        Geographic bounds for new axes and default projection-center inference.
    coordinate_frame : str or None, default=None
        Source-frame metadata label.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` Astropy sampling transform. Common
        frame names are ``"icrs"`` (equatorial), ``"galactic"``, and
        ``"geocentrictrueecliptic"`` (ecliptic).
    wcs : object or None, default=None
        WCS for 2D input.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit longitude/latitude WCS world-axis mapping.
    ax : GeoAxes or None, default=None
        Existing overlay axes; ``None`` creates axes.
    n_theta, n_phi : int, defaults=720, 1440
        Sampling-grid dimensions.
    resolution : {"low", "medium", "high"} or None, default=None
        Sampling-grid preset and 14, 16, or 18 point base font for low,
        medium, or high, respectively; also selects 120, 200, or 300 DPI.
    nest : bool, default=False
        Use HEALPix NEST ordering.
    interpolate : bool, default=True
        Interpolate samples.
    cmap : str or sequence, default="roma_r"
        Colormap specification.
    badvalue : float or None, default=healpy.UNSEEN
        Missing-data sentinel; ``None`` disables it.
    badcolor : color, default="grey"
        Missing-data color.
    vmin, vmax : float or None, defaults=None, None
        Color-scale limits.
    norm : str or Normalize or None, default=None
        Mesh color normalization.
    colorbar_title : str, default="Map value"
        Colorbar label.
    title : str or None, default=None
        Axes title.
    show_gridlines : bool, default=True
        Draw gridlines.
    plot_mode : {"map", "overlay_mask", "vector_field"}, default="map"
        Render a scalar map, a binary-mask overlay, or a transparent vector
        overlay. Vector mode requires a two-element ``(U, V)`` map sequence
        and ``ax=`` from a previously rendered magnitude map.
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
        This legacy switch is equivalent to ``plot_mode="overlay_mask"``.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float or None, default=None
        Layer opacity. Mask overlays default to ``0.25`` when omitted; an
        explicitly supplied value is used unchanged.
    zorder : float or None, default=None
        Artist drawing order. Scalar maps default to 1, vector fields to 2,
        and mask overlays to 3.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh options.
    vector_kwargs : dict or None, default=None
        Options for the vector artist in vector mode. Set ``method`` to
        ``"streamplot"`` (default) or ``"quiver"``. Scalar color arguments
        are ignored in vector mode.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    figsize : tuple[float, float], default=(8.0, 5.0)
        New-figure size.
    dpi : int, default=300
        New-figure resolution.
    """
    ccrs = _get_cartopy_crs_module()
    validated_extent = _validate_extent(extent)
    resolved_projection_kwargs = {} if projection_kwargs is None else dict(projection_kwargs)
    if "cutoff" in resolved_projection_kwargs:
        raise ValueError(
            "Cartopy's EquidistantConic CRS does not support 'cutoff'. "
            "Use extent=(lon_min, lon_max, lat_min, lat_max) instead."
        )
    if validated_extent is not None:
        lon_min, lon_max, lat_min, lat_max = validated_extent
        resolved_projection_kwargs.setdefault("central_longitude", 0.5 * (lon_min + lon_max))
        resolved_projection_kwargs.setdefault("central_latitude", 0.5 * (lat_min + lat_max))

    return plot_with_projection(
        map_data,
        projection_name="equidistantconic",
        projection_factory=ccrs.EquidistantConic,
        projection_kwargs=resolved_projection_kwargs,
        extent=validated_extent,
        coordinate_frame=coordinate_frame, coordinate_transform=coordinate_transform,
        wcs=wcs, world_axis_mapping=world_axis_mapping, ax=ax, n_theta=n_theta,
        n_phi=n_phi, resolution=resolution, nest=nest, interpolate=interpolate,
        cmap=cmap, badvalue=badvalue, badcolor=badcolor, vmin=vmin, vmax=vmax,
        norm=norm, colorbar_title=colorbar_title, title=title,
        show_gridlines=show_gridlines, gridline_adder=add_gridlines,
        plot_mode=plot_mode, overlay_mask=overlay_mask,
        overlay_color=overlay_color, alpha=alpha, zorder=zorder,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs, add_colorbar=add_colorbar,
        vector_kwargs=vector_kwargs,
        figsize=figsize, dpi=dpi,
    )


def gnomonic(
    map_data: Any,
    *,
    center: Sequence[float] = (0.0, 0.0),
    xsize: int = 500,
    ysize: int = 500,
    n_theta: int | None = None,
    n_phi: int | None = None,
    pixel_size_arcmin: float = 5.0,
    wcs: Any | None = None,
    world_axis_mapping: Sequence[int] | None = None,
    ax: Any | None = None,
    nest: bool = False,
    interpolate: bool = True,
    cmap: str | Sequence[Any] = "roma_r",
    badvalue: float | None = hp.UNSEEN,
    badcolor: Any = "grey",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: str | Normalize | None = None,
    colorbar_title: str = "Map value",
    title: str | None = None,
    show_gridlines: bool = False,
    gridline_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    plot_mode: Literal["map", "overlay_mask", "vector_field"] = "map",
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float | None = None,
    zorder: float | None = None,
    astro_orientation: bool = True,
    figsize: tuple[float, float] = (5.5, 6.5),
    dpi: int = 300,
    imshow_kwargs: dict[str, Any] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
) -> Figure:
    """Plot a local gnomonic view using Matplotlib.

    Parameters
    ----------
    map_data : numpy.ndarray or sequence
        Required 1D HEALPix map or 2D WCS-backed image. In vector mode, a
        two-element ``(U, V)`` sequence of matching component maps.
    center : sequence[float], default=(0.0, 0.0)
        Tangent point as ``(longitude_deg, latitude_deg)``.
    xsize, ysize : int, defaults=500, 500
        Output width and height in pixels.
    n_theta, n_phi : int or None, defaults=None, None
        Optional ``ysize`` and ``xsize`` overrides, respectively.
    pixel_size_arcmin : float, default=5.0
        Tangent-point pixel scale in arcminutes.
    wcs : object or None, default=None
        WCS for 2D input.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit longitude/latitude WCS world-axis mapping.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes for an overlay; ``None`` creates axes. Vector mode
        requires axes from a prior magnitude-map rendering.
    nest : bool, default=False
        Use HEALPix NEST ordering.
    interpolate : bool, default=True
        Interpolate sampled values.
    cmap : str or sequence, default="roma_r"
        Colormap specification.
    badvalue : float or None, default=healpy.UNSEEN
        Missing-data sentinel; ``None`` disables it.
    badcolor : color, default="grey"
        Missing-data color.
    vmin, vmax : float or None, defaults=None, None
        Color-scale limits.
    norm : str or Normalize or None, default=None
        Image color normalization.
    colorbar_title : str, default="Map value"
        Colorbar label.
    title : str or None, default=None
        Axes title.
    show_gridlines : bool, default=False
        Draw unlabeled curved longitude and latitude graticules in the tangent
        plane. The axes otherwise display the center, patch size, and pixel
        size rather than coordinate ticks.
    gridline_kwargs : dict or None, default=None
        Matplotlib contour-style overrides for gnomonic gridlines, such as
        ``color``, ``linestyle``, ``linewidth``, or ``alpha``.
    plot_mode : {"map", "overlay_mask", "vector_field"}, default="map"
        Render a scalar map, a binary-mask overlay, or a transparent vector
        overlay. Use a separate scalar-map call to render vector magnitude.
    overlay_mask : bool, default=False
        Legacy alias for ``plot_mode="overlay_mask"``.
    vector_kwargs : dict or None, default=None
        Options for the vector artist in vector mode. Set ``method`` to
        ``"streamplot"`` (default) or ``"quiver"``. Scalar color arguments
        are ignored in vector mode.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    alpha : float or None, default=None
        Image opacity. Mask overlays default to ``0.25`` when omitted; an
        explicitly supplied value is used unchanged.
    zorder : float or None, default=None
        Artist drawing order. Scalar maps default to 1, vector fields to 2,
        and mask overlays to 3.
    astro_orientation : bool, default=True
        Display increasing longitude to the left.
    figsize : tuple[float, float], default=(5.5, 5.5)
        New-figure size in inches.
    dpi : int, default=300
        New-figure resolution.
    imshow_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to ``Axes.imshow``.
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
    if 0.5 * max(xsize, ysize) * pixel_size_arcmin >= 90.0 * 60.0:
        raise ValueError("Gnomonic views must remain within 90 degrees of the tangent point.")

    lon0_deg, lat0_deg = _resolve_gnomonic_center(center)
    plot_mode = _plotlib._resolve_plot_mode(plot_mode, overlay_mask)
    overlay_mask = plot_mode == "overlay_mask"
    vector_field = plot_mode == "vector_field"
    if vector_field and ax is None:
        raise ValueError(
            "plot_mode='vector_field' requires ax= from a prior magnitude-map rendering."
        )
    if vector_field:
        u_data, u_wcs, v_data, v_wcs = _plotlib._resolve_vector_maps(map_data, wcs)
        cmap = "Greys"
        vmin = vmax = norm = None
        add_colorbar = False
        show_gridlines = False
    else:
        data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
        if overlay_mask:
            _plotlib._validate_binary_mask(data_arr)
            cmap = [overlay_color]
            vmin = vmax = None
            alpha = 0.25 if alpha is None else alpha
            add_colorbar = False
    if alpha is None:
        alpha = 1.0
    zorder_is_explicit = zorder is not None
    if zorder is None:
        zorder = 3.0 if overlay_mask else 2.0 if vector_field else 1.0
    resolved_cmap = _with_bad_color(_resolve_cmap(cmap), badcolor)
    x_pix = np.arange(xsize, dtype=float) - 0.5 * (xsize - 1)
    y_pix = np.arange(ysize, dtype=float) - 0.5 * (ysize - 1)
    plane_pixel_size = np.tan(np.radians(float(pixel_size_arcmin) / 60.0))
    x_plane = x_pix * plane_pixel_size
    y_plane = y_pix * plane_pixel_size
    x_plane_arcmin = np.degrees(x_plane) * 60.0
    y_plane_arcmin = np.degrees(y_plane) * 60.0
    x_plane_grid, y_plane_grid = np.meshgrid(x_plane, y_plane)
    lon_deg, lat_deg = _gnomonic_inverse(
        lon0_deg=lon0_deg, lat0_deg=lat0_deg, x_plane=x_plane_grid, y_plane=y_plane_grid
    )
    if vector_field:
        u_values = _plotlib._sample_map_values(
            u_data, wcs=u_wcs, lon=lon_deg, lat=lat_deg, nest=nest,
            interpolate=interpolate, world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )
        v_values = _plotlib._sample_map_values(
            v_data, wcs=v_wcs, lon=lon_deg, lat=lat_deg, nest=nest,
            interpolate=interpolate, world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )
        values = np.hypot(u_values, v_values)
    else:
        values = _plotlib._sample_map_values(
            data_arr, wcs=resolved_wcs, lon=lon_deg, lat=lat_deg, nest=nest,
            interpolate=interpolate, world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )
    if overlay_mask:
        values = np.ma.masked_where(values != 0, values)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    draw_kwargs: dict[str, Any] = {
        "origin": "lower",
        "interpolation": "nearest",
        "alpha": alpha,
    }
    if imshow_kwargs is not None:
        draw_kwargs.update(imshow_kwargs)
    draw_kwargs["zorder"] = zorder
    half_pix = 0.5 * np.degrees(plane_pixel_size) * 60.0
    extent = [
        float(x_plane_arcmin[0] - half_pix), float(x_plane_arcmin[-1] + half_pix),
        float(y_plane_arcmin[0] - half_pix), float(y_plane_arcmin[-1] + half_pix),
    ]
    image = None
    if not vector_field:
        image = ax.imshow(
            values, cmap=resolved_cmap, vmin=vmin, vmax=vmax, norm=norm,
            extent=extent, **draw_kwargs,
        )
    vector_method = None
    if vector_field:
        vector_method = _plotlib._draw_vector_field(
            ax, lon=x_plane_grid * 180.0 / np.pi * 60.0,
            lat=y_plane_grid * 180.0 / np.pi * 60.0, u=u_values, v=v_values,
            vector_kwargs=vector_kwargs, n_theta=ysize, n_phi=xsize,
            figsize=tuple(fig.get_size_inches()), resolution=None, zorder=zorder,
            force_zorder=zorder_is_explicit,
        )
    if astro_orientation != ax.xaxis_inverted():
        ax.invert_xaxis()
    if show_gridlines:
        _add_gnomonic_gridlines(
            ax,
            x_arcmin=x_plane_grid * 180.0 / np.pi * 60.0,
            y_arcmin=y_plane_grid * 180.0 / np.pi * 60.0,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            lon0_deg=lon0_deg,
            lat0_deg=lat0_deg,
            gridline_kwargs=gridline_kwargs,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    patch_width_deg = np.degrees(2.0 * np.arctan(0.5 * xsize * plane_pixel_size))
    patch_height_deg = np.degrees(2.0 * np.arctan(0.5 * ysize * plane_pixel_size))
    ax.set_xlabel(f"Center: ({lon0_deg:g}°, {lat0_deg:g}°)")
    ax.set_ylabel(
        f"Patch: {patch_width_deg:.3g}° × {patch_height_deg:.3g}°   "
        f"(Pixel size: {pixel_size_arcmin:g}')"
    )
    if add_colorbar:
        assert image is not None
        colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, aspect=40)
        colorbar.set_label(colorbar_title)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig._skyplot_payload = {  # type: ignore[attr-defined]
        "backend": "matplotlib-gnomonic", "projection": "gnomonic",
        "projection_kwargs": {"center": [lon0_deg, lat0_deg], "xsize": int(xsize), "ysize": int(ysize), "pixel_size_arcmin": float(pixel_size_arcmin), "astro_orientation": bool(astro_orientation)},
        "extent": [float(value) for value in extent], "shape": [int(values.shape[0]), int(values.shape[1])],
        "vmin": vmin, "vmax": vmax, "norm": str(norm) if norm is not None else None,
        "cmap": str(cmap), "badvalue": badvalue, "badcolor": str(badcolor),
        "colorbar_title": colorbar_title, "title": title, "show_gridlines": show_gridlines,
        "plot_mode": plot_mode, "overlay_mask": overlay_mask,
        "vector_method": vector_method,
        "alpha": alpha,
        "zorder": zorder,
        "colorbar_orientation": "horizontal",
    }
    if created_fig:
        plt.close(fig)
    _plotlib._last_figure = fig
    return fig

__all__ = [
    "AVAILABLE_PROJECTIONS",
    "DPI_PRESETS",
    "FONT_SIZE_PRESETS",
    "RESOLUTION_PRESETS",
    "add_gridlines",
    "equidistantconic",
    "gnomonic",
    "mollweide",
    "orthographic",
    "platecarree",
]
