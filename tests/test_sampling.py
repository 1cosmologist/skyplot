"""Tests for skyplot sampling utilities."""

import healpy as hp
import numpy as np

from skyplot.sampling import make_theta_phi_grid, sample_at_angles, sample_full_sky


def test_make_theta_phi_grid_shape() -> None:
    theta, phi = make_theta_phi_grid(16, 32)
    assert theta.shape == (16, 32)
    assert phi.shape == (16, 32)


def test_sample_at_angles_constant_map() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside), dtype=float)

    theta = np.array([0.2, 1.0, 2.4])
    phi = np.array([0.1, 1.5, 5.0])

    values = sample_at_angles(hp_map, theta, phi, interpolate=True)
    assert np.allclose(values, 1.0)


def test_sample_full_sky_shapes() -> None:
    nside = 16
    rng = np.random.default_rng(7)
    hp_map = rng.normal(size=hp.nside2npix(nside))

    lon, lat, values = sample_full_sky(hp_map, n_theta=24, n_phi=48)

    assert lon.shape == (24, 48)
    assert lat.shape == (24, 48)
    assert values.shape == (24, 48)


def test_sample_at_angles_lonlat_2d_interpolate_shape() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside), dtype=float)

    lon = np.linspace(-180.0, 180.0, 18, endpoint=False)
    lat = np.linspace(-80.0, 80.0, 9)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    values = sample_at_angles(
        hp_map,
        lon_grid,
        lat_grid,
        lonlat=True,
        interpolate=True,
    )

    assert values.shape == lon_grid.shape
    assert np.allclose(values, 1.0)
