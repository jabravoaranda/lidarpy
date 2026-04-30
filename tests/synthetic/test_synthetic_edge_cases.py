from __future__ import annotations

import numpy as np

from lidarpy.retrieval.synthetic.generator import (
    synthetic_signals,
    synthetic_signals_2D,
    synthetic_raman_signals_2D,
)


def _ranges() -> np.ndarray:
    return np.arange(7.5, 3000.0, 30.0)


def _time() -> np.ndarray:
    return np.arange(0.0, 20.0, 5.0)


def test_synthetic_signals_scalar_wavelength_is_elastic_only_even_with_raman_argument():
    _, raman, params = synthetic_signals(
        _ranges(),
        wavelengths=532,
        wavelength_raman=607,
        number_of_initial_nan_values=3,
    )

    assert raman is None
    assert "molecular_beta_raman" not in params


def test_synthetic_signals_tuple_wavelengths_enables_raman_channel():
    _, raman, params = synthetic_signals(
        _ranges(),
        wavelengths=(532, 607),
        k_lidar=(1e11, 1e10),
        number_of_initial_nan_values=3,
    )

    assert raman is not None
    assert "molecular_beta_raman" in params


def test_synthetic_signals_force_zero_after_bin_updates_returned_params():
    force_zero_after = 50

    _, _, params = synthetic_signals(
        _ranges(),
        wavelengths=532,
        force_zero_aer_after_bin=force_zero_after,
        number_of_initial_nan_values=3,
    )

    assert np.all(params["particle_alpha"][force_zero_after:] == 0.0)
    assert np.all(params["particle_beta"][force_zero_after:] == 0.0)


def test_synthetic_signals_2d_force_zero_after_bin_currently_targets_time_axis():
    force_zero_after = 2

    _, params = synthetic_signals_2D(
        _ranges(),
        _time(),
        force_zero_aer_after_bin=force_zero_after,
        number_of_initial_nan_values=3,
    )

    particle_alpha = params["particle_alpha2D"].values
    assert np.all(particle_alpha[force_zero_after:, :] == 0.0)
    assert np.isfinite(particle_alpha[:force_zero_after, :]).any()


def test_synthetic_raman_signals_2d_force_zero_after_bin_currently_targets_time_axis():
    force_zero_after = 2

    _, params = synthetic_raman_signals_2D(
        _ranges(),
        _time(),
        force_zero_aer_after_bin=force_zero_after,
        number_of_initial_nan_values=3,
    )

    particle_alpha = params["particle_alpha_raman2D"].values
    assert np.all(particle_alpha[force_zero_after:, :] == 0.0)
    assert np.isfinite(particle_alpha[:force_zero_after, :]).any()
