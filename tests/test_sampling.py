"""Tests for skyplot sampling utilities."""

import healpy as hp
import numpy as np
import pytest

from skyplot.sampling import make_theta_phi_grid, sample_at_angles, sample_full_sky


def test_make_theta_phi_grid_shape() -> None:
    theta, phi = make_theta_phi_grid(16, 32)
    assert theta.shape == (16, 32)
    assert phi.shape == (16, 32)


@pytest.mark.parametrize("n_theta, n_phi", [(1, 2), (2, 1), (2.0, 4), (4, 2.0), (True, 4)])
def test_make_theta_phi_grid_rejects_invalid_grid_sizes(n_theta, n_phi) -> None:
    with pytest.raises(ValueError, match="integer >= 2"):
        make_theta_phi_grid(n_theta, n_phi)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"theta_min": -0.1}, "theta bounds"),
        ({"theta_max": np.pi + 0.1}, "theta bounds"),
        ({"theta_min": np.nan}, "theta bounds"),
        ({"theta_min": 1.0, "theta_max": 1.0}, "theta_min"),
        ({"phi_min": -0.1}, "phi bounds"),
        ({"phi_max": 2.0 * np.pi + 0.1}, "phi bounds"),
        ({"phi_max": np.inf}, "phi bounds"),
        ({"phi_min": 1.0, "phi_max": 1.0}, "phi_min"),
    ],
)
def test_make_theta_phi_grid_rejects_invalid_angular_bounds(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_theta_phi_grid(4, 8, **kwargs)


def test_make_theta_phi_grid_uses_valid_custom_angular_bounds() -> None:
    theta, phi = make_theta_phi_grid(
        4,
        8,
        theta_min=0.2,
        theta_max=1.2,
        phi_min=0.4,
        phi_max=1.2,
    )

    assert theta[:, 0] == pytest.approx([0.2, 1.0 / 3.0 + 0.2, 2.0 / 3.0 + 0.2, 1.2])
    assert phi[0, 0] == pytest.approx(0.4)
    assert phi[0, -1] < 1.2


def test_sample_at_angles_constant_map() -> None:
    nside = 8
    hp_map = np.ones(hp.nside2npix(nside), dtype=float)

    theta = np.array([0.2, 1.0, 2.4])
    phi = np.array([0.1, 1.5, 5.0])

    values = sample_at_angles(hp_map, theta, phi, interpolate=True)
    assert np.allclose(values, 1.0)


def test_sample_at_angles_masks_unseen_and_nan_values() -> None:
    hp_map = np.ones(hp.nside2npix(2), dtype=float)
    hp_map[0] = hp.UNSEEN
    hp_map[1] = np.nan

    theta, phi = hp.pix2ang(2, [0, 1])
    values = sample_at_angles(hp_map, theta, phi, interpolate=False)

    assert np.isnan(values).all()


def test_sample_at_angles_rejects_complex_map() -> None:
    hp_map = np.ones(hp.nside2npix(2), dtype=complex)

    with pytest.raises(ValueError, match="real-valued"):
        sample_at_angles(hp_map, np.array([1.0]), np.array([1.0]))


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
