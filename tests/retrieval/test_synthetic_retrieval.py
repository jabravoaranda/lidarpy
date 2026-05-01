from __future__ import annotations

import matplotlib
import numpy as np
from scipy.integrate import cumulative_trapezoid

matplotlib.use("Agg")

from lidarpy.retrieval.klett import iterative_beta_forward, klett_rcs, quasi_beta
from lidarpy.retrieval.raman import retrieve_backscatter, retrieve_extinction
from lidarpy.retrieval.synthetic.generator import synthetic_signals
from lidarpy.utils.utils import signal_to_rcs


def _synthetic_profiles():
    ranges = np.arange(30.0, 6000.0, 30.0)
    elastic, raman, params = synthetic_signals(
        ranges,
        wavelengths=(532, 607),
        k_lidar=(1e11, 1e10),
        ae=1.0,
        lr=50.0,
        apply_overlap=False,
        number_of_initial_nan_values=0,
    )

    assert raman is not None
    return ranges, elastic.values, raman.values, params


def _assert_retrieved_profile(profile: np.ndarray, ranges: np.ndarray) -> None:
    assert profile.shape == ranges.shape
    assert np.isfinite(profile).all()
    assert np.nanmax(profile) > np.nanmin(profile)


def _useful_range_mask(ranges: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (ranges >= 600.0) & (ranges <= 2400.0) & (truth > 1e-9)


def _relative_error(retrieved: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.abs(truth), 1e-12)
    return np.abs(retrieved - truth) / denominator


def _reference_beta(params: dict, ranges: np.ndarray, reference: tuple[float, float]) -> float:
    idx_ref = (ranges >= reference[0]) & (ranges <= reference[1])
    return float(np.nanmean(params["particle_beta"][idx_ref]))


def test_klett_retrieves_finite_backscatter_from_synthetic_elastic_signal():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    particle_beta = klett_rcs(
        rcs,
        ranges,
        params["molecular_beta"],
        reference=(4500.0, 5500.0),
        lr_part=60.0,
        lr_mol=float(8 * np.pi / 3),
    )

    _assert_retrieved_profile(particle_beta, ranges)


def test_klett_backscatter_matches_synthetic_truth_in_useful_range():
    ranges, elastic, _, params = _synthetic_profiles()
    reference = (3000.0, 3500.0)

    rcs = signal_to_rcs(elastic, ranges)
    particle_beta = klett_rcs(
        rcs,
        ranges,
        params["molecular_beta"],
        reference=reference,
        lr_part=50.0,
        lr_mol=float(8 * np.pi / 3),
        beta_aer_ref=_reference_beta(params, ranges, reference),
    )
    truth = np.asarray(params["particle_beta"])
    error = _relative_error(particle_beta, truth)
    useful = _useful_range_mask(ranges, truth)

    assert np.nanmedian(error[useful]) < 0.01
    assert np.nanpercentile(error[useful], 95) < 0.02


def test_iterative_elastic_retrievals_accept_synthetic_elastic_signal():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    quasi_profile = quasi_beta(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        full_overlap_height=30.0,
    )
    forward_profile = iterative_beta_forward(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=60.0,
        start_height=500.0,
        height_top=2500.0,
    )

    _assert_retrieved_profile(np.asarray(quasi_profile), ranges)
    _assert_retrieved_profile(forward_profile, ranges)


def test_quasi_beta_is_single_iteration_approximation_not_exact_inversion():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    particle_beta = quasi_beta(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        full_overlap_height=30.0,
    )
    truth = np.asarray(params["particle_beta"])
    error = _relative_error(np.asarray(particle_beta), truth)
    useful = _useful_range_mask(ranges, truth)

    _assert_retrieved_profile(np.asarray(particle_beta), ranges)
    assert np.nanmedian(error[useful]) > 0.1


def test_iterative_beta_forward_matches_synthetic_truth_when_started_at_first_bin():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    particle_beta = iterative_beta_forward(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        start_height=None,
        height_top=2400.0,
    )
    truth = np.asarray(params["particle_beta"])
    error = _relative_error(particle_beta, truth)
    useful = _useful_range_mask(ranges, truth)

    assert np.nanmedian(error[useful]) < 1e-3
    assert np.nanpercentile(error[useful], 95) < 1e-2


def test_iterative_beta_forward_matches_synthetic_truth_with_start_height_boundary():
    ranges, elastic, _, params = _synthetic_profiles()
    start_height = 600.0

    rcs = signal_to_rcs(elastic, ranges)
    start_idx = np.abs(ranges - start_height).argmin()
    initial_particle_optical_depth = cumulative_trapezoid(
        params["particle_alpha"],
        ranges,
        initial=0.0,
    )[start_idx]
    particle_beta = iterative_beta_forward(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        start_height=start_height,
        height_top=2400.0,
        initial_particle_optical_depth=float(initial_particle_optical_depth),
    )
    truth = np.asarray(params["particle_beta"])
    error = _relative_error(particle_beta, truth)
    useful = _useful_range_mask(ranges, truth) & (ranges >= start_height)

    assert np.nanmedian(error[useful]) < 1e-3
    assert np.nanpercentile(error[useful], 95) < 1e-2


def test_iterative_beta_forward_rejects_negative_initial_particle_optical_depth():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    with np.testing.assert_raises(ValueError):
        iterative_beta_forward(
            rcs,
            calibration_factor=1e11,
            range_profile=ranges,
            params=params,
            lr_part=50.0,
            start_height=600.0,
            initial_particle_optical_depth=-1e-3,
        )


def test_raman_retrievals_accept_synthetic_elastic_and_raman_signals():
    ranges, elastic, raman, params = _synthetic_profiles()

    extinction = retrieve_extinction(
        raman,
        ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=(4500.0, 5500.0),
        particle_angstrom_exponent=1.0,
        full_overlap_height=500.0,
    )
    particle_beta = retrieve_backscatter(
        signal_raman=raman,
        signal_emission=elastic,
        extinction_profile=params["particle_alpha"],
        range_profile=ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=(4500.0, 5500.0),
        particle_angstrom_exponent=1.0,
    )

    _assert_retrieved_profile(extinction, ranges)
    _assert_retrieved_profile(particle_beta, ranges)


def test_raman_extinction_matches_synthetic_truth_in_useful_range():
    ranges, _, raman, params = _synthetic_profiles()

    extinction = retrieve_extinction(
        raman,
        ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=(4500.0, 5500.0),
        particle_angstrom_exponent=1.0,
        full_overlap_height=500.0,
    )
    truth = np.asarray(params["particle_alpha"])
    error = _relative_error(extinction, truth)
    useful = _useful_range_mask(ranges, np.asarray(params["particle_beta"]))

    assert np.nanmedian(error[useful]) < 1e-6
    assert np.nanpercentile(error[useful], 95) < 0.01


def test_raman_backscatter_matches_synthetic_truth_in_useful_range():
    ranges, elastic, raman, params = _synthetic_profiles()
    reference = (4500.0, 5500.0)

    extinction = retrieve_extinction(
        raman,
        ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=reference,
        particle_angstrom_exponent=1.0,
        full_overlap_height=500.0,
    )
    particle_beta = retrieve_backscatter(
        signal_raman=raman,
        signal_emission=elastic,
        extinction_profile=extinction,
        range_profile=ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=reference,
        particle_angstrom_exponent=1.0,
        beta_part_ref=_reference_beta(params, ranges, reference),
    )
    truth = np.asarray(params["particle_beta"])
    error = _relative_error(particle_beta, truth)
    useful = _useful_range_mask(ranges, truth)

    assert np.nanmedian(error[useful]) < 1e-4
    assert np.nanpercentile(error[useful], 95) < 1e-3
