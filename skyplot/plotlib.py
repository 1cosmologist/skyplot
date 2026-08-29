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
from collections import OrderedDict
from importlib import import_module
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
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

# Cached display geometry is independent of map values and WCS metadata. Keep
# only two grids: a high-resolution pair of float64 lon/lat arrays is about
# 66 MiB, while retaining high and medium costs roughly 83 MiB.
_DISPLAY_GRID_CACHE_MAXSIZE = 2
_display_grid_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()


def _get_last_figure() -> Figure | None:
    """Return the most recently created skyplot figure, if any."""
    return _last_figure


def _get_display_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an immutable, astronomy-oriented display grid from a small LRU cache.

    Parameters
    ----------
    n_theta : int
        Required colatitude grid size.
    n_phi : int
        Required longitude grid size.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Read-only ``(longitude_deg, latitude_deg)`` arrays in the exact order
        used for Cartopy rendering.

    Notes
    -----
    The cache stores display geometry only. WCS conversion and map sampling
    always run for the current map, WCS object, coordinate transform, and
    world-axis mapping, so cached coordinates cannot be reused as pixel
    coordinates for an incompatible map.
    """
    key = (n_theta, n_phi)
    cached = _display_grid_cache.get(key)
    if cached is not None:
        _display_grid_cache.move_to_end(key)
        return cached

    theta, phi = make_theta_phi_grid(n_theta=n_theta, n_phi=n_phi)
    lon = np.degrees(phi)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = 90.0 - np.degrees(theta)

    # Match healpy.mollview's default ``flip="astro"`` convention: increasing
    # phi (east) moves from right to left on the displayed sky.
    lon = -lon
    lon = ((lon + 180.0) % 360.0) - 180.0
    lon_sort_idx = np.argsort(lon[0, :])
    lon = lon[:, lon_sort_idx]
    lat = lat[:, lon_sort_idx]

    lat_sort_idx = np.argsort(lat[:, 0])
    lon = lon[lat_sort_idx, :]
    lat = lat[lat_sort_idx, :]
    lon.setflags(write=False)
    lat.setflags(write=False)
    _display_grid_cache[key] = (lon, lat)
    if len(_display_grid_cache) > _DISPLAY_GRID_CACHE_MAXSIZE:
        _display_grid_cache.popitem(last=False)
    return lon, lat


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


def _validate_binary_mask(mask: np.ndarray) -> None:
    """Raise when a mask contains values other than boolean/zero/one.

    Parameters
    ----------
    mask : numpy.ndarray
        Required mask array. Boolean arrays are accepted directly; numeric
        arrays must be finite and contain only ``0`` or ``1``.
    """
    values = np.asarray(mask)
    if values.dtype == np.bool_:
        return
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("overlay_mask requires a finite binary mask containing only 0/1 or False/True.")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("overlay_mask requires a binary mask containing only 0/1 or False/True.")


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
    copied.set_bad((0.0, 0.0, 0.0, 0.0) if badcolor is None else badcolor)
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
    gridline_adder: Callable[..., Any] | None = None,
    overlay_mask: bool = False,
    overlay_color: Any = "k",
    alpha: float = 1.0,
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
    gridline_adder : callable or None, default=None
        Function that adds Cartopy gridlines. Required when
        ``show_gridlines=True`` for direct low-level use.
    overlay_mask : bool, default=False
        Treat ``map_data`` as a binary allowed-pixel mask. Invalid (zero or
        false) samples are drawn as a translucent overlay; valid samples are
        masked and transparent. ``vmin`` and ``vmax`` are ignored.
    overlay_color : color, default="k"
        Invalid-pixel overlay color. ``cmap`` is ignored for mask overlays.
    alpha : float, default=1.0
        Opacity of the plotted layer. Mask overlays use ``0.25``.
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
    if overlay_mask:
        _validate_binary_mask(data_arr)
        interpolate = False
        show_gridlines = False
        add_colorbar = False
        vmin = None
        vmax = None
        alpha = 0.25
        cmap = ListedColormap([overlay_color])
    ccrs = _get_cartopy_crs_module()
    resolved_cmap = _with_bad_color(
        _resolve_cmap(cmap),
        (0.0, 0.0, 0.0, 0.0) if overlay_mask else badcolor,
    )

    lon, lat = _get_display_grid(n_theta=n_theta, n_phi=n_phi)
    source_lon, source_lat, source_frame_label, display_frame_label = _transform_display_coordinates(
        lon,
        lat,
        coordinate_frame=coordinate_frame,
        coordinate_transform=coordinate_transform,
    )

    if data_arr.ndim == 1:
        values = sample_at_angles(
            data_arr,
            source_lon,
            source_lat,
            nest=nest,
            lonlat=True,
            interpolate=interpolate,
            badvalue=badvalue,
        )
    else:
        values = _sample_wcs_map(
            data_arr,
            wcs=resolved_wcs,
            lon=source_lon,
            lat=source_lat,
            interpolate=interpolate,
            world_axis_mapping=world_axis_mapping,
            badvalue=badvalue,
        )

    if overlay_mask:
        values = np.ma.masked_where(values != 0, values)

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

    if projection_name == "mollweide" and not ax.xaxis_inverted():
        # Healpy's default astronomy convention has phi increase to the left.
        # Flip the completed projected view instead of altering the map's
        # longitude samples, which preserves Cartopy's central-longitude wrap.
        ax.invert_xaxis()

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
        if gridline_adder is None:
            raise ValueError(
                "gridline_adder is required when show_gridlines=True. "
                "Use skyplot.plotting.add_gridlines."
            )
        gridline_adder(ax, **applied_gridline_kwargs)

    mesh_kwargs: dict[str, Any] = {
        "shading": "nearest",
        "rasterized": True,
        "alpha": alpha,
    }
    if pcolormesh_kwargs is not None:
        mesh_kwargs.update(pcolormesh_kwargs)
    if overlay_mask:
        mesh_kwargs["alpha"] = alpha

    mesh_cmap = resolved_cmap
    if projection_name == "mollweide" and not np.isclose(
        float(projection_kwargs.get("central_longitude", 0.0)) % 360.0,
        0.0,
    ):
        # Cartopy masks cells at the rotated wrap seam.  Its pcolormesh
        # implementation requires that internal mask to be transparent;
        # otherwise it draws a broad badcolor band across the projection.
        mesh_cmap = resolved_cmap.copy()
        mesh_cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    quad = ax.pcolormesh(
        lon,
        lat,
        values,
        transform=ccrs.PlateCarree(),
        cmap=mesh_cmap,
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
        # Titles sit just above the plot, so keep them only modestly larger
        # than the base resolution-scaled text.
        ax.set_title(title, fontsize=1.1 * font_size)
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
        "overlay_mask": overlay_mask,
        "alpha": alpha,
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
