from typing import Callable
from warnings import warn


import numpy as np
import xarray as xr
from scipy.signal import savgol_filter

from lidarpy.general_utils.optimized import best_slope_fit


def _create_smoother_normalizer(window_size: int) -> Callable[[np.ndarray], np.ndarray]:
    def smooth_and_normalize(row: np.ndarray) -> np.ndarray:
        return savgol_filter(row / np.nanmean(row), window_size, 4)

    return smooth_and_normalize


# # TODO: This part can be paralellized with Numba
# def get_gluing_indexes(
#     signal_an: np.ndarray, rcs_pc: np.ndarray, half_window_bins: int
# ) -> None:

#     # To paralelize with Numba
#     for time_idx in np.arange(signal_an.shape[0]):
#         for h_idx in np.arange(rcs_an.shape[1]):
#             ...


def gluing(
    signal_an: xr.DataArray,
    signal_pc: xr.DataArray,
    range_min: float = 1500,
    range_max: float = 5000,
    criteria_half_window: float = 300,
    adjustment_half_window: float = 2200,
):

    sel_signal_an = signal_an.sel(
        range=slice(range_min - criteria_half_window, range_max + criteria_half_window)
    )
    sel_signal_pc = signal_pc.sel(
        range=slice(range_min - criteria_half_window, range_max + criteria_half_window)
    )

    sel_range = sel_signal_pc.range.values
    window_size_bins = int((criteria_half_window * 2) / (sel_range[1] - sel_range[0]))

    smooth_window = 11  # window_size_bins // 2

    sm_sel_signal_an = np.apply_along_axis(
        _create_smoother_normalizer(smooth_window), 0, sel_signal_an.values
    )
    sm_sel_signal_pc = np.apply_along_axis(
        _create_smoother_normalizer(smooth_window), 0, sel_signal_pc.values
    )

    signal_glued = signal_pc.values.copy()

    gluing_bins = best_slope_fit(
        sm_sel_signal_an, sm_sel_signal_pc, window=window_size_bins
    )
    gluing_ranges = sel_range[gluing_bins]

    for idx, gluing_range in enumerate(gluing_ranges[:]):
        near_idx = (sel_range > (gluing_range - adjustment_half_window)) & (
            sel_range < (gluing_range + adjustment_half_window)
        )

        sel_an = sel_signal_an[dict(time=idx)][near_idx]
        sel_pc = sel_signal_pc[dict(time=idx)][near_idx]

        r_gluing = np.corrcoef(sm_sel_signal_an[idx], sm_sel_signal_pc[idx])[0, 1]

        fit_values = np.polyfit(
            sel_an,
            sel_pc,
            1,
        )

        if r_gluing < 1:  # TODO: Adjust threshold
            signal_an2pc = np.polyval(fit_values, signal_an.values[idx, :])
            signal_glued[idx, :] = np.concatenate(
                [
                    signal_an2pc[signal_pc.range.values < gluing_range],
                    signal_pc[idx, signal_pc.range.values >= gluing_range],
                ]
            )
        else:
            warn(
                f"r2 thereshold condition not satisfied for profile in time index {idx}"
            )
            # TODO: warn or implement filling method

    return signal_glued
