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


def test_generate_particle_properties_preserves_lidar_ratio_and_angstrom_scaling():
    ranges = _ranges()
    ae = 1.5
    lr = 50.0

    _, _, beta_532, _, _, alpha_532 = generate_particle_properties(
        ranges,
        wavelength=532,
        ae=ae,
        lr=lr,
        synthetic_beta=2.5e-6,
        sigmoid_edge=2500,
    )
    _, _, beta_1064, _, _, alpha_1064 = generate_particle_properties(
        ranges,
        wavelength=1064,
        ae=ae,
        lr=lr,
        synthetic_beta=2.5e-6,
        sigmoid_edge=2500,
    )

    expected_beta_1064 = beta_532 * (1064 / 532) ** -ae
    np.testing.assert_allclose(alpha_532, lr * beta_532)
    np.testing.assert_allclose(alpha_1064, lr * beta_1064)
    np.testing.assert_allclose(beta_1064, expected_beta_1064)


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


def test_synthetic_elastic_signal_matches_lidar_equation_without_overlap():
    ranges = _ranges()
    k_lidar = 1e11

    elastic, _, params = synthetic_signals(
        ranges,
        wavelengths=532,
        k_lidar=k_lidar,
        apply_overlap=False,
        number_of_initial_nan_values=3,
    )

    expected_signal = (
        k_lidar
        * (params["overlap"] / ranges**2)
        * (params["molecular_beta"] + params["particle_beta"])
        * params["transmittance_elastic"] ** 2
    )
    np.testing.assert_allclose(elastic.values[3:], expected_signal.values[3:])
    assert np.all(params["overlap"] == 1.0)


def test_synthetic_transmittance_is_bounded_and_monotonic():
    ranges = _ranges()

    _, _, params = synthetic_signals(
        ranges,
        wavelengths=(532, 607),
        k_lidar=(1e11, 1e10),
        apply_overlap=False,
        number_of_initial_nan_values=3,
    )

    elastic_transmittance = np.asarray(params["transmittance_elastic"])
    raman_transmittance = np.asarray(params["transmittance_raman"])

    assert np.all((elastic_transmittance > 0.0) & (elastic_transmittance <= 1.0))
    assert np.all((raman_transmittance > 0.0) & (raman_transmittance <= 1.0))
    assert np.all(np.diff(elastic_transmittance) <= 0.0)
    assert np.all(np.diff(raman_transmittance) <= 0.0)


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


def test_synthetic_raman_signal_matches_lidar_equation_without_overlap():
    ranges = _ranges()
    k_lidar_raman = 1e10

    elastic, raman, params = synthetic_signals(
        ranges,
        wavelengths=(532, 607),
        k_lidar=(1e11, k_lidar_raman),
        apply_overlap=False,
        number_of_initial_nan_values=3,
    )

    assert raman is not None
    expected_signal = (
        k_lidar_raman
        * (params["overlap"] / ranges**2)
        * params["molecular_beta_raman"]
        * params["transmittance_elastic"]
        * params["transmittance_raman"]
    )
    np.testing.assert_allclose(raman.values[3:], expected_signal.values[3:])
    assert np.isfinite(elastic.values[3:]).all()


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


def test_synthetic_depolarization_preserves_particle_ratio():
    ranges = _ranges()
    despo_particle = 0.33

    _, _, params = synthetic_signals_despo(
        ranges,
        despo_particle=despo_particle,
        number_of_initial_nan_values=3,
    )

    particle_beta_parallel = params["particle_beta_parallel"]
    particle_beta_perpendicular = params["particle_beta_perpendicular"]
    valid = particle_beta_parallel > 0

    np.testing.assert_allclose(
        particle_beta_perpendicular[valid] / particle_beta_parallel[valid],
        despo_particle,
    )
    np.testing.assert_allclose(
        params["despolarization_particle"][valid],
        despo_particle,
    )
    np.testing.assert_allclose(
        params["particle_beta_total"],
        particle_beta_parallel + particle_beta_perpendicular,
    )


def test_synthetic_depolarization_volumic_ratio_matches_components():
    ranges = _ranges()

    _, _, params = synthetic_signals_despo(
        ranges,
        number_of_initial_nan_values=3,
    )

    expected_volumic = (
        params["molecular_beta_perpendicular"] + params["particle_beta_perpendicular"]
    ) / (params["molecular_beta_parallel"] + params["particle_beta_parallel"])
    np.testing.assert_allclose(params["despolarization_volumic"], expected_volumic)


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


def test_synthetic_signals_2d_preserves_lidar_ratio_at_each_time():
    ranges = _ranges()
    time = _time()
    lr = 50.0

    _, params = synthetic_signals_2D(
        ranges,
        time,
        lr=lr,
        number_of_initial_nan_values=3,
    )

    np.testing.assert_allclose(
        params["particle_alpha2D"].values,
        lr * params["particle_beta2D"].values,
    )


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


def test_synthetic_raman_2d_preserves_angstrom_scaling_for_particle_extinction():
    ranges = _ranges()
    time = _time()
    ae = 1.5
    wavelength = 532
    wavelength_raman = 607

    _, params = synthetic_raman_signals_2D(
        ranges,
        time,
        wavelength=wavelength,
        wavelength_raman=wavelength_raman,
        ae=ae,
        number_of_initial_nan_values=3,
    )

    expected_alpha_raman = (
        params["particle_alpha_elastic2D"].values
        * (wavelength_raman / wavelength) ** -ae
    )
    np.testing.assert_allclose(
        params["particle_alpha_raman2D"].values,
        expected_alpha_raman,
    )
