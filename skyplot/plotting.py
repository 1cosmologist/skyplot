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


def mollweide(
    map_data: np.ndarray,
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
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float = 1.0,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Mollweide projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
    projection_kwargs : dict or None, default=None
        Keyword arguments passed to Cartopy's ``Mollweide`` CRS.
    coordinate_frame : str or None, default=None
        Metadata label for the source coordinate frame.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` used to sample transformed coordinates.
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
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float, default=1.0
        Layer opacity; mask overlays use ``0.25``.
    gridline_kwargs : dict or None, default=None
        Overrides for gridline color, style, width, spacing, or opacity.
    pcolormesh_kwargs : dict or None, default=None
        Extra keyword arguments passed to ``GeoAxes.pcolormesh``.
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
        overlay_mask=overlay_mask, overlay_color=overlay_color, alpha=alpha,
        gridline_kwargs=gridline_kwargs, pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar, figsize=figsize, dpi=dpi,
    )


def orthographic(
    map_data: np.ndarray,
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
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float = 1.0,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (5.5, 6.5),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy Orthographic projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
    projection_kwargs : dict or None, default=None
        Keyword arguments passed to Cartopy's ``Orthographic`` CRS.
    coordinate_frame : str or None, default=None
        Metadata label for the source coordinate frame.
    coordinate_transform : sequence[str] or None, default=None
        Source/display frame pair used before sampling.
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
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float, default=1.0
        Layer opacity; mask overlays use ``0.25``.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh keyword arguments.
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
        overlay_mask=overlay_mask, overlay_color=overlay_color, alpha=alpha,
        gridline_kwargs=gridline_kwargs, pcolormesh_kwargs=pcolormesh_kwargs,
        add_colorbar=add_colorbar, figsize=figsize, dpi=dpi,
    )


def platecarree(
    map_data: np.ndarray,
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
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float = 1.0,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy PlateCarree projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
    projection_kwargs : dict or None, default=None
        Keyword arguments for Cartopy's ``PlateCarree`` CRS.
    extent : sequence[float] or None, default=None
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees for new axes;
        an existing ``ax`` retains its own extent.
    coordinate_frame : str or None, default=None
        Source-frame metadata label.
    coordinate_transform : sequence[str] or None, default=None
        Source/display coordinate-frame pair used for sampling.
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
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float, default=1.0
        Layer opacity; mask overlays use ``0.25``.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh options.
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
        overlay_mask=overlay_mask,
        overlay_color=overlay_color, alpha=alpha,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs, add_colorbar=add_colorbar,
        figsize=figsize, dpi=dpi,
    )


def equidistantconic(
    map_data: np.ndarray,
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
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float = 1.0,
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Plot a sky map using a Cartopy EquidistantConic projection.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
    projection_kwargs : dict or None, default=None
        Cartopy ``EquidistantConic`` options; ``cutoff`` is unsupported.
    extent : sequence[float] or None, default=None
        Geographic bounds for new axes and default projection-center inference.
    coordinate_frame : str or None, default=None
        Source-frame metadata label.
    coordinate_transform : sequence[str] or None, default=None
        Source/display frame pair used for sampling.
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
    overlay_mask : bool, default=False
        Render a binary allowed-pixel mask as an invalid-pixel overlay.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float, default=1.0
        Layer opacity; mask overlays use ``0.25``.
    gridline_kwargs, pcolormesh_kwargs : dict or None, defaults=None, None
        Extra gridline and mesh options.
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
        overlay_mask=overlay_mask,
        overlay_color=overlay_color, alpha=alpha,
        gridline_kwargs=gridline_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs, add_colorbar=add_colorbar,
        figsize=figsize, dpi=dpi,
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
    add_colorbar: bool = True,
    alpha: float = 1.0,
    astro_orientation: bool = True,
    figsize: tuple[float, float] = (5.5, 5.5),
    dpi: int = 300,
    imshow_kwargs: dict[str, Any] | None = None,
) -> Figure:
    """Plot a local gnomonic view using Matplotlib.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
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
        Existing axes for an overlay; ``None`` creates axes.
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
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    alpha : float, default=1.0
        Image opacity.
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
    data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
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
    if data_arr.ndim == 1:
        values = sample_at_angles(
            data_arr, lon_deg, lat_deg, nest=nest, lonlat=True,
            interpolate=interpolate, badvalue=badvalue,
        )
    else:
        values = _sample_wcs_map(
            data_arr, wcs=resolved_wcs, lon=lon_deg, lat=lat_deg,
            interpolate=interpolate, world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )

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
    half_pix = 0.5 * np.degrees(plane_pixel_size) * 60.0
    extent = [
        float(x_plane_arcmin[0] - half_pix), float(x_plane_arcmin[-1] + half_pix),
        float(y_plane_arcmin[0] - half_pix), float(y_plane_arcmin[-1] + half_pix),
    ]
    image = ax.imshow(
        values, cmap=resolved_cmap, vmin=vmin, vmax=vmax, norm=norm,
        extent=extent, **draw_kwargs,
    )
    if astro_orientation != ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel("Tangent-plane x [arcmin]")
    ax.set_ylabel("Tangent-plane y [arcmin]")
    if add_colorbar:
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
        "colorbar_title": colorbar_title, "title": title, "show_gridlines": False,
        "alpha": alpha,
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
