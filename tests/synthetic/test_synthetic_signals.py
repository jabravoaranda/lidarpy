from __future__ import annotations

import numpy as np
import xarray as xr

from lidarpy.retrieval.synthetic.generator import (
    generate_particle_properties,
    synthetic_raman_signals_2D,
    synthetic_signals,
    synthetic_signals_2D,
    synthetic_signals_despo,
)


def _ranges() -> np.ndarray:
    return np.arange(7.5, 3000.0, 30.0)


def _time() -> np.ndarray:
    return np.arange(0.0, 20.0, 5.0)


def _assert_profile_contract(profile: xr.DataArray, ranges: np.ndarray) -> None:
    assert isinstance(profile, xr.DataArray)
    assert profile.dims == ("range",)
    assert profile.shape == ranges.shape
    assert np.allclose(profile["range"].values, ranges)
    assert np.isnan(profile.values[:3]).all()
    assert np.isfinite(profile.values[3:]).any()


def test_generate_particle_properties_returns_finite_profiles():
    ranges = _ranges()

    profiles = generate_particle_properties(ranges, wavelength=532)

    assert len(profiles) == 6
    for profile in profiles:
        assert profile.shape == ranges.shape
        assert np.isfinite(profile).all()

    beta_fine, beta_coarse, beta_total, alpha_fine, alpha_coarse, alpha_total = profiles
    assert np.allclose(beta_total, beta_fine + beta_coarse)
    assert np.allclose(alpha_total, alpha_fine + alpha_coarse)


def test_synthetic_signals_elastic_only_contract():
    ranges = _ranges()

    elastic, raman, params = synthetic_signals(
        ranges,
        wavelengths=532,
        number_of_initial_nan_values=3,
    )

    _assert_profile_contract(elastic, ranges)
    assert raman is None
    assert params["ranges"].shape == ranges.shape
    assert params["overlap"].shape == ranges.shape
    assert {"particle_beta", "particle_alpha", "molecular_beta", "molecular_alpha"}.issubset(
        params
    )


def test_synthetic_signals_elastic_and_raman_contract():
    ranges = _ranges()

    elastic, raman, params = synthetic_signals(
        ranges,
        wavelengths=(532, 607),
        k_lidar=(1e11, 1e10),
        number_of_initial_nan_values=3,
    )

    _assert_profile_contract(elastic, ranges)
    assert raman is not None
    _assert_profile_contract(raman, ranges)
    assert {
        "molecular_alpha_raman",
        "molecular_beta_raman",
        "transmittance_raman",
    }.issubset(params)


def test_synthetic_depolarization_signals_contract():
    ranges = _ranges()

    reflected, transmitted, params = synthetic_signals_despo(
        ranges,
        number_of_initial_nan_values=3,
    )

    _assert_profile_contract(reflected, ranges)
    _assert_profile_contract(transmitted, ranges)
    assert params["ranges"].shape == ranges.shape
    assert params["overlap"].shape == ranges.shape
    assert {
        "despolarization_volumic",
        "despolarization_particle",
        "attenuated_molecular_backscatter_total",
    }.issubset(params)


def test_synthetic_signals_2d_contract():
    ranges = _ranges()
    time = _time()

    signal, params = synthetic_signals_2D(
        ranges,
        time,
        number_of_initial_nan_values=3,
    )

    assert isinstance(signal, xr.DataArray)
    assert signal.dims == ("time", "range")
    assert signal.shape == (time.size, ranges.size)
    assert np.allclose(signal["time"].values, time)
    assert np.allclose(signal["range"].values, ranges)
    assert np.isnan(signal.values[:, :3]).all()
    assert np.isfinite(signal.values[:, 3:]).any()
    assert params["particle_beta2D"].dims == ("time", "range")
    assert params["overlap2D"].shape == (time.size, ranges.size)


def test_synthetic_raman_signals_2d_contract():
    ranges = _ranges()
    time = _time()

    signal, params = synthetic_raman_signals_2D(
        ranges,
        time,
        number_of_initial_nan_values=3,
    )

    assert isinstance(signal, xr.DataArray)
    assert signal.dims == ("time", "range")
    assert signal.shape == (time.size, ranges.size)
    assert np.allclose(signal["time"].values, time)
    assert np.allclose(signal["range"].values, ranges)
    assert np.isnan(signal.values[:, :3]).all()
    assert np.isfinite(signal.values[:, 3:]).any()
    assert params["particle_alpha_raman2D"].dims == ("time", "range")
    assert params["molecular_beta_raman2D"].shape == (time.size, ranges.size)
