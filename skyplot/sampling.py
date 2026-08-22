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

"""Sampling helpers for HEALPix sky maps."""

from __future__ import annotations

import numpy as np
import healpy as hp


def _as_1d_healpix_map(hp_map: np.ndarray) -> np.ndarray:
    """Validate and normalize a HEALPix map into a 1D float array.

    Parameters
    ----------
    hp_map : numpy.ndarray
        Input HEALPix map. It must be a one-dimensional array with length
        equal to `12 * nside**2`.

    Returns
    -------
    numpy.ndarray
        A one-dimensional float array.

    Raises
    ------
    ValueError
        If the map is not one-dimensional or does not match a valid HEALPix
        number of pixels.
    """
    arr = np.asarray(hp_map)
    if arr.ndim != 1:
        raise ValueError("HEALPix map must be one-dimensional.")

    if isinstance(hp_map, np.ma.MaskedArray):
        arr = hp_map.filled(np.nan)

    arr = arr.astype(float, copy=False)

    try:
        hp.get_nside(arr)
    except Exception as exc:  # pragma: no cover - healpy exception type may vary.
        raise ValueError("Input does not look like a valid HEALPix map.") from exc

    return arr


def sample_at_angles(
    hp_map: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    nest: bool = False,
    lonlat: bool = False,
    interpolate: bool = True,
) -> np.ndarray:
    """Sample a HEALPix map at supplied angular coordinates.

    Parameters
    ----------
    hp_map : numpy.ndarray
        Input HEALPix map, shape ``(npix,)``.
    theta : numpy.ndarray
        Polar angle(s). If ``lonlat=False``, this is colatitude in radians.
        If ``lonlat=True``, this is longitude in degrees.
    phi : numpy.ndarray
        Azimuth angle(s). If ``lonlat=False``, this is longitude in radians.
        If ``lonlat=True``, this is latitude in degrees.
    nest : bool, optional
        Whether input map uses NEST ordering. Default is ``False`` for RING.
    lonlat : bool, optional
        Interpret ``theta`` and ``phi`` as longitude/latitude in degrees when
        ``True``. Default is ``False``.
    interpolate : bool, optional
        Use bilinear interpolation when ``True``. Use nearest-pixel sampling
        when ``False``. Default is ``True``.

    Returns
    -------
    numpy.ndarray
        Sampled map values with broadcasted shape of ``theta`` and ``phi``.

    Raises
    ------
    ValueError
        If input map is invalid or angle arrays have incompatible shapes.
    """
    hp_arr = _as_1d_healpix_map(hp_map)

    theta_arr = np.asarray(theta)
    phi_arr = np.asarray(phi)

    theta_b, phi_b = np.broadcast_arrays(theta_arr, phi_arr)

    # Flattening avoids healpy broadcasting issues for multi-dimensional lon/lat
    # grids (notably with lonlat=True in interpolation mode).
    theta_flat = theta_b.reshape(-1)
    phi_flat = phi_b.reshape(-1)

    if interpolate:
        values = hp.get_interp_val(
            hp_arr,
            theta_flat,
            phi_flat,
            nest=nest,
            lonlat=lonlat,
        )
        return np.asarray(values).reshape(theta_b.shape)

    nside = hp.get_nside(hp_arr)
    pix = hp.ang2pix(nside, theta_flat, phi_flat, nest=nest, lonlat=lonlat)
    return hp_arr[pix].reshape(theta_b.shape)


def make_theta_phi_grid(
    n_theta: int,
    n_phi: int,
    *,
    theta_min: float = 1e-4,
    theta_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular full-sky angular grid in radians.

    Parameters
    ----------
    n_theta : int
        Number of samples in colatitude direction.
    n_phi : int
        Number of samples in longitude direction.
    theta_min : float, optional
        Minimum colatitude in radians. Small positive defaults avoid exact
        pole singularities in some workflows.
    theta_max : float or None, optional
        Maximum colatitude in radians. If ``None``, uses ``pi - theta_min``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(theta_grid, phi_grid)`` each with shape ``(n_theta, n_phi)``.

    Raises
    ------
    ValueError
        If ``n_theta`` or ``n_phi`` is less than 2.
    """
    if n_theta < 2 or n_phi < 2:
        raise ValueError("n_theta and n_phi must both be >= 2.")

    if theta_max is None:
        theta_max = np.pi - theta_min

    theta = np.linspace(theta_min, theta_max, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    return np.meshgrid(theta, phi, indexing="ij")


def sample_full_sky(
    hp_map: np.ndarray,
    *,
    n_theta: int = 240,
    n_phi: int = 480,
    nest: bool = False,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an input HEALPix map on a regular full-sky grid.

    Parameters
    ----------
    hp_map : numpy.ndarray
        Input HEALPix map, shape ``(npix,)``.
    n_theta : int, optional
        Number of colatitude samples in output grid. Default is ``240``.
    n_phi : int, optional
        Number of longitude samples in output grid. Default is ``480``.
    nest : bool, optional
        Whether input map uses NEST ordering. Default is ``False``.
    interpolate : bool, optional
        Use interpolation when ``True``, nearest-pixel lookup otherwise.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(lon_deg, lat_deg, values)`` arrays, all with shape
        ``(n_theta, n_phi)``.

    Notes
    -----
    Longitudes are wrapped to ``[-180, 180)`` degrees to align with Plotly
    geo projections.
    """
    theta, phi = make_theta_phi_grid(n_theta=n_theta, n_phi=n_phi)
    values = sample_at_angles(
        hp_map,
        theta,
        phi,
        nest=nest,
        lonlat=False,
        interpolate=interpolate,
    )

    lon = np.degrees(phi)
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = 90.0 - np.degrees(theta)
    return lon, lat, values
