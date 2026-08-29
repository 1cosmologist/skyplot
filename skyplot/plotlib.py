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
import healpy as hp
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure

from importlib import resources

from .sampling import make_theta_phi_grid, sample_at_angles

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

# Font sizes are in typographic points.  They are deliberately larger than
# Matplotlib's defaults so labels remain legible in the raster output for each
# sampling preset.
FONT_SIZE_PRESETS: dict[str, float] = {
    "low": 14.0,
    "medium": 16.0,
    "high": 18.0,
}

DPI_PRESETS: dict[str, int] = {
    "low": 120,
    "medium": 200,
    "high": 300,
}

_last_figure: Figure | None = None


def _get_last_figure() -> Figure | None:
    """Return the most recently created skyplot figure, if any."""
    return _last_figure


def _font_size_for_resolution(resolution: Literal["low", "medium", "high"] | None) -> float:
    """Return the base font size for a sampling preset.

    Parameters
    ----------
    resolution : {"low", "medium", "high"} or None
        Sampling preset. ``None`` uses the ``"medium"`` size of ``16.0``
        points, which matches the default grid density.

    Returns
    -------
    float
        Base font size in points.
    """
    return FONT_SIZE_PRESETS["medium" if resolution is None else resolution]


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
    if np.iscomplexobj(data_arr):
        raise ValueError(
            "SkyPlot expects real-valued maps. Select a real, imaginary, "
            "magnitude, or phase component before plotting."
        )

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


def _resolve_wcs_world_axis_mapping(
    wcs: Any,
    world_axis_mapping: Sequence[int] | None,
) -> tuple[int, int]:
    """Return the WCS world-axis indices for longitude and latitude.

    The returned pair is ``(lon_axis, lat_axis)`` in the order expected by
    ``all_world2pix``. Astropy/WCSLIB metadata is preferred; an explicit pair
    is required when a WCS does not identify a longitude-like and a
    latitude-like axis.
    """
    wcsprm = getattr(wcs, "wcs", None)
    world_n_dim = getattr(wcs, "world_n_dim", None)
    if world_n_dim is None:
        world_n_dim = getattr(wcsprm, "naxis", 2)
    if world_n_dim != 2:
        raise ValueError(
            "WCS-backed 2D maps must have exactly two world axes; reduce the "
            "WCS to its plotted axes before passing it to skyplot."
        )

    if world_axis_mapping is not None:
        if isinstance(world_axis_mapping, (str, bytes)):
            raise ValueError(
                "world_axis_mapping must be a two-integer sequence: "
                "(longitude_axis, latitude_axis)."
            )
        try:
            lon_axis, lat_axis = world_axis_mapping
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "world_axis_mapping must be a two-integer sequence: "
                "(longitude_axis, latitude_axis)."
            ) from exc
        if (
            not isinstance(lon_axis, (int, np.integer))
            or not isinstance(lat_axis, (int, np.integer))
            or lon_axis == lat_axis
            or not (0 <= lon_axis < world_n_dim)
            or not (0 <= lat_axis < world_n_dim)
        ):
            raise ValueError(
                "world_axis_mapping must contain distinct zero-based world-axis "
                "indices in the form (longitude_axis, latitude_axis)."
            )
        return int(lon_axis), int(lat_axis)

    lng = getattr(wcsprm, "lng", -1)
    lat = getattr(wcsprm, "lat", -1)
    if isinstance(lng, (int, np.integer)) and isinstance(lat, (int, np.integer)):
        if 0 <= lng < world_n_dim and 0 <= lat < world_n_dim and lng != lat:
            return int(lng), int(lat)

    physical_types = getattr(wcs, "world_axis_physical_types", None)
    ctype = getattr(wcsprm, "ctype", None)
    if ctype is None:
        ctype = getattr(wcs, "ctype", None)
    physical_types = [] if physical_types is None else list(physical_types)
    ctype = [] if ctype is None else list(ctype)
    physical_types.extend([None] * (world_n_dim - len(physical_types)))
    ctype.extend([None] * (world_n_dim - len(ctype)))

    def _role(physical_type: Any, axis_ctype: Any) -> str | None:
        physical = "" if physical_type is None else str(physical_type).lower()
        ctype_name = "" if axis_ctype is None else str(axis_ctype).upper()
        if physical.endswith((".lon", ".ra")) or any(
            name in ctype_name for name in ("RA--", "GLON", "ELON", "HLON", "SLON", "LON-")
        ):
            return "lon"
        if physical.endswith((".lat", ".dec")) or any(
            name in ctype_name for name in ("DEC-", "GLAT", "ELAT", "HLAT", "SLAT", "LAT-")
        ):
            return "lat"
        return None

    roles = [_role(physical_types[i], ctype[i]) for i in range(world_n_dim)]
    if roles.count("lon") == 1 and roles.count("lat") == 1:
        return roles.index("lon"), roles.index("lat")

    raise ValueError(
        "Could not infer longitude/latitude world axes from WCS metadata. "
        "Pass world_axis_mapping=(longitude_axis, latitude_axis) using "
        "zero-based WCS world-axis indices."
    )


def _sample_wcs_map(
    data: np.ndarray,
    *,
    wcs: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    interpolate: bool,
    world_axis_mapping: Sequence[int] | None,
    badvalue: float | None = hp.UNSEEN,
) -> np.ndarray:
    """Sample a 2D WCS-backed map at lon/lat (degrees) positions."""
    if not hasattr(wcs, "all_world2pix"):
        raise ValueError("Provided wcs object must implement all_world2pix(...).")

    data = np.asarray(data, dtype=float)
    bad_mask = ~np.isfinite(data)
    if badvalue is not None:
        bad_mask |= data == badvalue
    if np.any(bad_mask):
        data = data.copy()
        data[bad_mask] = np.nan

    nrows, ncols = data.shape
    lon_axis, lat_axis = _resolve_wcs_world_axis_mapping(wcs, world_axis_mapping)
    n_samples = lon.size
    world = np.empty((n_samples, 2), dtype=float)
    world[:, lon_axis] = lon.reshape(-1)
    world[:, lat_axis] = lat.reshape(-1)

    # A full-sky WCS can represent longitudes in a different 360-degree
    # interval than the display grid. Evaluate all equivalent longitude
    # aliases in one WCS call, then retain the first coordinate inside the
    # source image footprint.
    world_trials = np.concatenate([world, world.copy(), world.copy()])
    world_trials[n_samples : 2 * n_samples, lon_axis] += 360.0
    world_trials[2 * n_samples :, lon_axis] -= 360.0
    pix_raw = wcs.all_world2pix(world_trials, 0)

    # Support both Nx2 array returns and tuple-of-arrays returns.
    if isinstance(pix_raw, tuple):
        if len(pix_raw) != 2:
            raise ValueError("wcs.all_world2pix must return pixel x/y coordinates.")
        x_trials = np.asarray(pix_raw[0], dtype=float).reshape(3, n_samples)
        y_trials = np.asarray(pix_raw[1], dtype=float).reshape(3, n_samples)
    else:
        pix = np.asarray(pix_raw, dtype=float)
        if pix.ndim != 2 or pix.shape != (3 * n_samples, 2):
            raise ValueError("wcs.all_world2pix must return an array with shape (N, 2).")
        x_trials = pix[:, 0].reshape(3, n_samples)
        y_trials = pix[:, 1].reshape(3, n_samples)

    inside_trials = (
        np.isfinite(x_trials)
        & np.isfinite(y_trials)
        & (x_trials >= -0.5)
        & (x_trials <= ncols - 0.5)
        & (y_trials >= -0.5)
        & (y_trials <= nrows - 0.5)
    )
    trial_index = np.argmax(inside_trials, axis=0)
    sample_index = np.arange(n_samples)
    x = x_trials[trial_index, sample_index]
    y = y_trials[trial_index, sample_index]
    valid = inside_trials[trial_index, sample_index]

    sampled = np.full(x.shape, np.nan, dtype=float)

    if interpolate:
        x0 = np.floor(np.where(valid, x, 0.0)).astype(int)
        y0 = np.floor(np.where(valid, y, 0.0)).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        if np.any(valid):
            xv = x[valid]
            yv = y[valid]
            x0v = np.clip(x0[valid], 0, ncols - 1)
            y0v = np.clip(y0[valid], 0, nrows - 1)
            x1v = np.clip(x1[valid], 0, ncols - 1)
            y1v = np.clip(y1[valid], 0, nrows - 1)

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
        if np.any(valid):
            xi = np.clip(np.rint(x[valid]).astype(int), 0, ncols - 1)
            yi = np.clip(np.rint(y[valid]).astype(int), 0, nrows - 1)
            sampled[valid] = data[yi, xi]

    return sampled.reshape(lon.shape)


def _transform_display_coordinates(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    coordinate_frame: str | None,
    coordinate_transform: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, str | None, list[str] | None]:
    """Transform display-frame coordinates into the input map frame.

    ``coordinate_transform`` is a ``(source_frame, display_frame)`` pair.
    Display-frame coordinates are transformed back to the source frame before
    sampling so each plotted value remains attached to its physical sky
    position. ``coordinate_frame`` is descriptive metadata only.
    """
    if coordinate_transform is None:
        return lon, lat, _coordinate_frame_label(coordinate_frame), None

    if isinstance(coordinate_transform, (str, bytes)):
        raise ValueError(
            "coordinate_transform must be a two-string sequence: "
            "(source_frame, display_frame)."
        )
    try:
        source_frame, display_frame = coordinate_transform
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "coordinate_transform must be a two-string sequence: "
            "(source_frame, display_frame)."
        ) from exc
    if not isinstance(source_frame, str) or not isinstance(display_frame, str):
        raise ValueError(
            "coordinate_transform must be a two-string sequence: "
            "(source_frame, display_frame)."
        )


    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ImportError(
            "coordinate_transform requires astropy. Install skyplot with its "
            "Astropy dependency."
        ) from exc

    try:
        display_coordinates = SkyCoord(
            lon.reshape(-1) * u.deg,
            lat.reshape(-1) * u.deg,
            frame=display_frame,
        )
        source_coordinates = display_coordinates.transform_to(source_frame)
        source_lon = source_coordinates.spherical.lon.to_value(u.deg).reshape(lon.shape)
        source_lat = source_coordinates.spherical.lat.to_value(u.deg).reshape(lat.shape)
    except Exception as exc:
        raise ValueError(
            "coordinate_transform source_frame and display_frame must be "
            "valid Astropy celestial frames."
        ) from exc

    return (
        source_lon,
        source_lat,
        _coordinate_frame_label(coordinate_frame),
        [source_frame, display_frame],
    )


def _coordinate_frame_label(frame: Any | None) -> str | None:
    """Return a compact, serializable frame label for figure metadata."""
    if frame is None:
        return None
    if isinstance(frame, str):
        return frame
    name = getattr(frame, "name", None)
    return str(name) if name is not None else frame.__class__.__name__


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


def _with_bad_color(cmap: str | Colormap, badcolor: Any) -> Colormap:
    """Copy a colormap and configure its missing-data color."""
    resolved = plt.get_cmap(cmap)
    copied = resolved.copy()
    copied.set_bad(badcolor)
    return copied


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
    x_plane: np.ndarray,
    y_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dimensionless tangent-plane coordinates to lon/lat degrees."""
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    x = np.asarray(x_plane, dtype=float)
    y = np.asarray(y_plane, dtype=float)

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


def plot_gridlines(
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


def plot_with_projection(
    map_data: np.ndarray,
    *,
    projection_name: str,
    projection_factory: Callable[..., Any],
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
    gridline_kwargs: dict[str, Any] | None = None,
    pcolormesh_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
) -> Figure:
    """Render a sampled sky map on a supplied Cartopy projection.

    This is the implementation primitive behind the named projection
    renderers. Most callers should use :mod:`skyplot.plotting`; this function
    is useful when an application supplies its own Cartopy CRS factory.

    Parameters
    ----------
    map_data : numpy.ndarray
        Required 1D HEALPix map or 2D WCS-backed image.
    projection_name : str
        Required metadata name for the projection.
    projection_factory : callable
        Required callable that constructs the Cartopy CRS.
    projection_kwargs : dict or None, default=None
        CRS-constructor keyword arguments.
    extent : sequence[float] or None, default=None
        New-axes geographic extent as ``(lon_min, lon_max, lat_min, lat_max)``.
    coordinate_frame : str or None, default=None
        Source-frame metadata label.
    coordinate_transform : sequence[str] or None, default=None
        ``(source_frame, display_frame)`` sampling transform.
    wcs : object or None, default=None
        WCS used for a 2D input.
    world_axis_mapping : sequence[int] or None, default=None
        Explicit longitude/latitude WCS world-axis mapping.
    ax : GeoAxes or None, default=None
        Existing axes for an overlay; ``None`` creates axes.
    n_theta : int, default=720
        Colatitude sampling-grid size.
    n_phi : int, default=1440
        Longitude sampling-grid size.
    resolution : {"low", "medium", "high"} or None, default=None
        Preset overriding ``n_theta`` and ``n_phi``. It also uses a 14, 16,
        or 18 point base font and a 120, 200, or 300 DPI new figure for low,
        medium, or high, respectively.
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
    vmin : float or None, default=None
        Lower color-scale limit.
    vmax : float or None, default=None
        Upper color-scale limit.
    norm : str or Normalize or None, default=None
        Mesh color normalization.
    colorbar_title : str, default="Map value"
        Colorbar label.
    title : str or None, default=None
        Axes title.
    show_gridlines : bool, default=True
        Draw gridlines.
    gridline_kwargs : dict or None, default=None
        Gridline option overrides.
    pcolormesh_kwargs : dict or None, default=None
        Extra mesh keyword arguments.
    add_colorbar : bool, default=True
        Add a horizontal colorbar.
    figsize : tuple[float, float], default=(8.0, 5.0)
        New-figure size in inches.
    dpi : int, default=300
        New-figure resolution.

    The ``projection_factory`` creates a Cartopy CRS. When ``ax`` is
    provided, its projection and viewport are retained for overlay rendering.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if len(figsize) != 2 or figsize[0] <= 0.0 or figsize[1] <= 0.0:
        raise ValueError("figsize must be a two-element tuple of positive values.")

    if resolution is not None:
        if resolution not in RESOLUTION_PRESETS:
            supported = ", ".join(sorted(RESOLUTION_PRESETS))
            raise ValueError(f"Unsupported resolution '{resolution}'. Choose one of: {supported}")
        n_theta, n_phi = RESOLUTION_PRESETS[resolution]
        if ax is None:
            dpi = DPI_PRESETS[resolution]

    font_size = _font_size_for_resolution(resolution)

    validated_extent = _validate_extent(extent)
    projection_kwargs = {} if projection_kwargs is None else dict(projection_kwargs)
    data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
    ccrs = _get_cartopy_crs_module()
    resolved_cmap = _with_bad_color(_resolve_cmap(cmap), badcolor)

    theta, phi = make_theta_phi_grid(n_theta=n_theta, n_phi=n_phi)
    display_lon = np.degrees(phi)
    display_lon = ((display_lon + 180.0) % 360.0) - 180.0
    display_lat = 90.0 - np.degrees(theta)

    lon, lat, source_frame_label, display_frame_label = _transform_display_coordinates(
        display_lon,
        display_lat,
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
    )

    if data_arr.ndim == 1:
        values = sample_at_angles(
            data_arr,
            lon,
            lat,
            nest=nest,
            lonlat=True,
            interpolate=interpolate,
            badvalue=badvalue,
        )
    else:
        values = _sample_wcs_map(
            data_arr,
            wcs=resolved_wcs,
            lon=lon,
            lat=lat,
            interpolate=interpolate,
            world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )

    # Sampling happens in the map's frame, while the projected grid remains
    # in the requested display frame.
    lon = display_lon
    lat = display_lat

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

        # An existing axes owns its viewport.  In particular, an ``extent``
        # passed for a newly-created axes must not make the metadata claim a
        # different rendered region when layering onto an existing GeoAxes.
        validated_extent = tuple(
            float(value) for value in ax.get_extent(crs=ccrs.PlateCarree())
        )

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
        plot_gridlines(ax, **applied_gridline_kwargs)

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
        cbar.set_label(colorbar_title, fontsize=font_size)
        cbar.ax.tick_params(labelsize=0.85 * font_size)

    if title:
        ax.set_title(title, fontsize=1.2 * font_size)
    ax.tick_params(labelsize=0.85 * font_size)

    fig.tight_layout()

    # Used by save_figure for lightweight JSON export.
    fig._skyplot_payload = {  # type: ignore[attr-defined]
        "backend": "matplotlib-cartopy",
        "projection": projection_name,
        "projection_kwargs": projection_kwargs,
        "extent": list(validated_extent) if validated_extent is not None else None,
        "coordinate_frame": source_frame_label,
        "coordinate_transform": display_frame_label,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "vmin": vmin,
        "vmax": vmax,
        "norm": str(norm) if norm is not None else None,
        "cmap": str(cmap),
        "badvalue": badvalue,
        "badcolor": str(badcolor),
        "colorbar_title": colorbar_title,
        "title": title,
        "show_gridlines": show_gridlines,
        "gridline_color": applied_gridline_kwargs["color"],
        "gridline_linestyle": applied_gridline_kwargs["linestyle"],
        "gridline_linewidth": applied_gridline_kwargs["linewidth"],
        "lon_gridline_spacing_deg": applied_gridline_kwargs["lon_gridline_spacing_deg"],
        "lat_gridline_spacing_deg": applied_gridline_kwargs["lat_gridline_spacing_deg"],
        "colorbar_orientation": "horizontal",
        "font_size": font_size,
        "dpi": dpi,
    }

    # Prevent matplotlib's Jupyter inline backend from auto-displaying this
    # figure a second time in addition to the one shown via the return value.
    if created_fig:
        plt.close(fig)

    global _last_figure
    _last_figure = fig

    return fig


def plot_mollweide(
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
        Input sky map. A 1D array is interpreted as a HEALPix map. A 2D array
        is interpreted as a WCS-backed image and requires ``wcs=`` or a
        ``.wcs`` attribute on the input object.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.Mollweide``.
    coordinate_frame : str or None, optional
        Optional coordinate-frame label written to the figure metadata. It
        does not affect the Astropy conversion.
    coordinate_transform : sequence[str] or None, optional
        Two Astropy frame names as ``(source_frame, display_frame)``. The
        display grid is transformed into ``source_frame`` before sampling.
        This works independently of ``coordinate_frame``.
    wcs : Any or None, optional
        WCS object used when ``map_data`` is 2D. Must implement
        ``all_world2pix``.
    world_axis_mapping : sequence[int] or None, optional
        Zero-based WCS world-axis indices as ``(longitude_axis,
        latitude_axis)``. Normally inferred from WCS metadata (including
        RA/Dec and lon/lat axis pairs). Supply this for a WCS whose metadata
        does not identify the two angular axes.
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
    badvalue : float or None, optional
        Input sentinel rendered as missing data. Defaults to ``healpy.UNSEEN``.
        NaN values are always rendered as missing data.
    badcolor : color, optional
        Matplotlib color used for missing data. Defaults to ``"grey"``.
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
        Keyword arguments for :func:`plot_gridlines`.
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
    return plot_with_projection(
        map_data,
        projection_name="mollweide",
        projection_factory=ccrs.Mollweide,
        projection_kwargs=projection_kwargs,
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
        wcs=wcs,
        world_axis_mapping=world_axis_mapping,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        badvalue=badvalue,
        badcolor=badcolor,
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


def plot_orthographic(
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
        1D HEALPix map or 2D WCS-backed map.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.Orthographic``.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    coordinate_frame, coordinate_transform, world_axis_mapping, n_theta, n_phi, resolution, nest,
    interpolate, cmap, badvalue, badcolor, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`plot_mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    ccrs = _get_cartopy_crs_module()
    return plot_with_projection(
        map_data,
        projection_name="orthographic",
        projection_factory=ccrs.Orthographic,
        projection_kwargs=projection_kwargs,
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
        wcs=wcs,
        world_axis_mapping=world_axis_mapping,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        badvalue=badvalue,
        badcolor=badcolor,
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


def plot_platecarree(
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
        1D HEALPix map or 2D WCS-backed map.
    extent: sequence[float] or None, optional
        Geographic render window as ``(lon_min, lon_max, lat_min, lat_max)``.
        When ``ax`` is supplied, the existing axes extent is used instead.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``cartopy.crs.PlateCarree``.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    coordinate_frame, coordinate_transform, world_axis_mapping, n_theta, n_phi, resolution, nest,
    interpolate, cmap, badvalue, badcolor, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`plot_mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
    """
    validated_extent = _validate_extent(extent)
    
    ccrs = _get_cartopy_crs_module()
    return plot_with_projection(
        map_data,
        projection_name="platecarree",
        projection_factory=ccrs.PlateCarree,
        projection_kwargs=projection_kwargs,
        extent=validated_extent,
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
        wcs=wcs,
        world_axis_mapping=world_axis_mapping,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        badvalue=badvalue,
        badcolor=badcolor,
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


def plot_equidistantconic(
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
        1D HEALPix map or 2D WCS-backed map.
    projection_kwargs : dict[str, Any] or None, optional
        Keyword arguments for ``cartopy.crs.EquidistantConic``. Cartopy's
        EquidistantConic CRS does not support a ``cutoff`` option; use
        ``extent=`` to control the rendered geographic region.
    extent : sequence[float] or None, optional
        Geographic render window as ``(lon_min, lon_max, lat_min, lat_max)``.
        When provided, missing ``central_longitude``/``central_latitude`` are
        inferred from the extent center. When ``ax`` is supplied, the existing
        axes extent is used for rendering.
    wcs : Any or None, optional
        WCS object used for 2D map inputs.
    ax : Any or None, optional
        Existing GeoAxes to draw into for overlay workflows.
    coordinate_frame, coordinate_transform, world_axis_mapping, n_theta, n_phi, resolution, nest,
    interpolate, cmap, badvalue, badcolor, vmin, vmax, norm,
    colorbar_title, title, show_gridlines, gridline_kwargs,
    pcolormesh_kwargs, add_colorbar, figsize, dpi
        Same behavior as :func:`plot_mollweide`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the rendered map.
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
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
        wcs=wcs,
        world_axis_mapping=world_axis_mapping,
        ax=ax,
        n_theta=n_theta,
        n_phi=n_phi,
        resolution=resolution,
        nest=nest,
        interpolate=interpolate,
        cmap=cmap,
        badvalue=badvalue,
        badcolor=badcolor,
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


def plot_gnomonic(
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
    astro_orientation: bool = True,
    figsize: tuple[float, float] = (5.5, 5.5),
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
        :func:`plot_gnomonic` can be called interchangeably with the other
        projection functions.
    pixel_size_arcmin : float, optional
        Angular size per pixel in arcminutes.
    wcs : Any or None, optional
        WCS object used when ``map_data`` is 2D. Must implement
        ``all_world2pix``.
    world_axis_mapping : sequence[int] or None, optional
        Zero-based WCS world-axis indices as ``(longitude_axis,
        latitude_axis)`` when they cannot be inferred from metadata.
    ax : Any or None, optional
        Existing Axes to draw into for overlay workflows.
    nest, interpolate, cmap, badvalue, badcolor, vmin, vmax, colorbar_title, title, add_colorbar,
    figsize, dpi
        Same behavior as :func:`plot_mollweide` where applicable.
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
    if 0.5 * max(xsize, ysize) * pixel_size_arcmin >= 90.0 * 60.0:
        raise ValueError(
            "Gnomonic views must remain within 90 degrees of the tangent point."
        )

    lon0_deg, lat0_deg = _resolve_gnomonic_center(center)
    data_arr, resolved_wcs = _resolve_input_map_and_wcs(map_data, wcs)
    resolved_cmap = _with_bad_color(_resolve_cmap(cmap), badcolor)

    x_pix = np.arange(xsize, dtype=float) - 0.5 * (xsize - 1)
    y_pix = np.arange(ysize, dtype=float) - 0.5 * (ysize - 1)
    # In a gnomonic projection the tangent-plane coordinate is tan(angle),
    # not the angle itself. The supplied size is the local angular pixel scale
    # at the tangent point, so its tangent is the constant plane-pixel step.
    plane_pixel_size = np.tan(np.radians(float(pixel_size_arcmin) / 60.0))
    x_plane = x_pix * plane_pixel_size
    y_plane = y_pix * plane_pixel_size
    x_plane_arcmin = np.degrees(x_plane) * 60.0
    y_plane_arcmin = np.degrees(y_plane) * 60.0

    x_plane_grid, y_plane_grid = np.meshgrid(x_plane, y_plane)
    lon_deg, lat_deg = _gnomonic_inverse(
        lon0_deg=lon0_deg,
        lat0_deg=lat0_deg,
        x_plane=x_plane_grid,
        y_plane=y_plane_grid,
    )

    if data_arr.ndim == 1:
        values = sample_at_angles(
            data_arr,
            lon_deg,
            lat_deg,
            nest=nest,
            lonlat=True,
            interpolate=interpolate,
            badvalue=badvalue,
        )
    else:
        values = _sample_wcs_map(
            data_arr,
            wcs=resolved_wcs,
            lon=lon_deg,
            lat=lat_deg,
            interpolate=interpolate,
            world_axis_mapping=world_axis_mapping,
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
    }
    if imshow_kwargs is not None:
        draw_kwargs.update(imshow_kwargs)

    half_pix = 0.5 * np.degrees(plane_pixel_size) * 60.0
    extent = [
        float(x_plane_arcmin[0] - half_pix),
        float(x_plane_arcmin[-1] + half_pix),
        float(y_plane_arcmin[0] - half_pix),
        float(y_plane_arcmin[-1] + half_pix),
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

    if astro_orientation != ax.xaxis_inverted():
        ax.invert_xaxis()

    ax.set_xlabel("Tangent-plane x [arcmin]")
    ax.set_ylabel("Tangent-plane y [arcmin]")

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
        "badvalue": badvalue,
        "badcolor": str(badcolor),
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
