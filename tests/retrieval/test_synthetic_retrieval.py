from __future__ import annotations

import matplotlib
import numpy as np

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
        apply_overlap=False,
        number_of_initial_nan_values=0,
    )

    assert raman is not None
    return ranges, elastic.values, raman.values, params


def _assert_retrieved_profile(profile: np.ndarray, ranges: np.ndarray) -> None:
    assert profile.shape == ranges.shape
    assert np.isfinite(profile).all()
    assert np.nanmax(profile) > np.nanmin(profile)


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


def test_iterative_elastic_retrievals_accept_synthetic_elastic_signal():
    ranges, elastic, _, params = _synthetic_profiles()

    rcs = signal_to_rcs(elastic, ranges)
    quasi_profile = quasi_beta(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=60.0,
        full_overlap_height=500.0,
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
