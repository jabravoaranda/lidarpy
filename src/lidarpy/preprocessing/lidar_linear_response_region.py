import numpy as np
from typing import Any, Tuple
from scipy.signal import savgol_filter

from lidarpy.general_utils.fitting import linear_regression

""" LINEAR RESPONSE REGION
"""


def estimate_linear_response_region(
    ranges: np.ndarray[Any, np.dtype[np.float64]],
    an: np.ndarray[Any, np.dtype[np.float64]],
    pc: np.ndarray[Any, np.dtype[np.float64]],
    an_bg: np.ndarray[Any, np.dtype[np.float64]],
    an_bg_thres: float = 0.02,
    pc_thres: float = 15,
    region_range: Tuple = (1500, 10000),
    min_region: float = 1000,
    subregion: float = 500,
    n_step: int = 3,
    correlation_thres: float = 0.85,
):
    """

    Input:
    ranges: 1D array. ranges (m)
    an: analog signal preprocessed. 1D array.
    pc: photoncounting signal preprocessed 1D array.
    an_bg: background value for analog signal
    an_bg_thres = 0.1
    an_min: Min Thresh for AN
    pc_thres = 15MHz. Min Thresh. for PC
    region_range = (1000, 6000). Altitude Region to find linear response. tuple
    min_delta_h (m). length of range interval for LR region
    step_h (m). length of step for moving potential LR region
    r2_thres (>0, <1): threshold for R2 to consider LR
    """

    id_h = np.where(
        np.logical_and(ranges >= region_range[0], ranges <= region_range[1])
    )[0]
    ranges_aux = ranges[id_h]
    pc_aux = savgol_filter(pc[id_h], 21, polyorder=2)
    an_aux = savgol_filter(an[id_h], 21, polyorder=2)
    if np.logical_and(
        (pc_aux < np.float64(pc_thres)).any(),
        (an_aux < np.float64(an_bg * an_bg_thres)).any(),
    ):  #
        idx_min = 0
        while pc_aux[idx_min] > pc_thres:
            idx_min += 1
        idx_max = 0
        while an_aux[idx_max] > an_bg * an_bg_thres:
            idx_max += 1

        if ranges_aux[idx_max] - ranges_aux[idx_min] < min_region:
            idx_max = np.argmax(ranges_aux >= (ranges_aux[idx_min] + min_region))

        rvalue_arr = []
        slope_arr = []
        intercept_arr = []
        h_min_arr = []
        h_max_arr = []
        h_min = ranges[id_h][idx_min]
        h_max = ranges[id_h][idx_max]
        hi = h_min
        step = np.diff(ranges).mean()
        while hi < (h_max - subregion):
            he = hi + subregion
            while he < h_max:
                idx = np.logical_and(ranges >= hi, ranges <= he)
                a, b, rvalue_i = linear_regression(an[idx], pc[idx])
                if rvalue_i >= correlation_thres:
                    slope_arr.append(a)
                    intercept_arr.append(b)
                    rvalue_arr.append(rvalue_i)
                    h_min_arr.append(ranges[idx][0])
                    h_max_arr.append(ranges[idx][-1])
                he += n_step * step
            hi += n_step * step
        if len(rvalue_arr) > 0:
            linear_region = True
            idx_lr = np.nanargmax(np.asarray(rvalue_arr))
            slope = slope_arr[idx_lr]
            intercept = intercept_arr[idx_lr]
            r2 = rvalue_arr[idx_lr]
            h_lr_min = h_min_arr[idx_lr]
            h_lr_max = h_max_arr[idx_lr]
        else:
            linear_region = False
            idx_lr, slope, intercept, r2, h_lr_min, h_lr_max = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
    else:
        linear_region = False
        idx_lr, slope, intercept, r2, h_lr_min, h_lr_max = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    return h_lr_min, h_lr_max, r2, slope, intercept, linear_region
