from pdb import set_trace
import numpy as np
import xarray as xr

from lidarpy.general_utils.numerics import windowed_proportional


def gluing(
    signal_an: xr.DataArray,
    signal_pc: xr.DataArray,
    range_min: float = 1100,
    range_max: float = 5000,
    window_meters: float = 500,
    # max_rel_error: float = 0.10,
):

    t_slice = slice(range_min - window_meters / 2, range_max + window_meters / 2)

    rcs_an = signal_an.sel(range=t_slice)
    rcs_an = rcs_an * signal_an.range**2

    rcs_pc = signal_pc.sel(range=t_slice)
    rcs_pc = rcs_pc * signal_pc.range**2

    ranges = rcs_an.range.values
    ranges_complete = signal_an.range.values
    window_size = int(window_meters // (ranges[1] - ranges[0]))

    coeff, error = windowed_proportional(
        rcs_an.values, rcs_pc.values, w_size=window_size
    )

    min_error = np.argmin(error, axis=1)
    gluing_height = signal_an.sel(range=slice(range_min, range_max)).range.values[
        min_error
    ]
    # if len(gluing_height.shape) > 0:
    #     gluing_height = gluing_height[0]

    # TODO: Coefs an be overwritten and smoothen along time axis with savgol filter
    # coeff = coeff[:][min_error]

    rcs_gl = (signal_pc * signal_pc.range**2).values.copy()
    an_values = (signal_an * signal_an.range**2).values

    for time_idx in np.arange(signal_an.time.shape[0]):
        mask = ranges_complete < gluing_height[time_idx]
        # set_trace()
        rcs_gl[time_idx][mask] = (
            an_values[time_idx][mask] * coeff[time_idx][min_error[time_idx]]
        )

        # if error[time_idx][min_error[time_idx]] > max_rel_error:
        #     rcs_gl[time_idx] = np.nan

    return signal_an.copy(data=rcs_gl).rename(signal_an.name[:-1] + "g"), error, gluing_height  # type: ignore

    # rolling = sliding_window_view(np.arange(ranges.shape[0]), int(window_meters // window_size))

    # # xr.full_like(signal_an.sel(range=slice(range_min, range_max)), np.nan)

    # # np.empty((signal_an.shape[0], ranges.shape[0]))

    # for idx, window in enumerate(rolling):
    #     an_window = rcs_an.values.T[window].T
    #     pc_window = rcs_pc.values.T[window].T

    #     ratio = (pc_window / an_window).mean(axis=0)
    #     an_adj = an_window * ratio

    #     error: np.ndarray = np.abs(an_adj - pc_window) / pc_window  # type: ignore
    #     error.mean(axis=1)

    #     # set_trace()

    #     errors[idx] = error.mean(axis=1)
    #     set_trace()
    #     # print(f"Errors calculated for {_range}")

    # return errors


# @nb.njit(parallel = True)
# def evaluate_errors(an: np.ndarray, pc: np.ndarray) -> np.ndarray:


#     set_trace()
