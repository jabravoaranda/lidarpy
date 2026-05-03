import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from loguru import logger
from scipy import signal
import matplotlib.pyplot as plt

from lidarpy.general_utils.numerics import normalize

""" LAYER DETECTION

    Description of Algorithm:
    -------------------------
    It takes RCS

"""


def find_peaks_1d(rcs, ranges, check=True, plot_results=False):
    """_summary_

    Args:
        rcs (_type_): _description_
        ranges (_type_): _description_
        method (_type_): _description_

    Returns:
        _type_: _description_
    """

    def check_candidate_peak(x, pk_ind):
        """_summary_

        Args:
            x (_type_): _description_
            pk_ind (_type_): _description_

        Returns:
            _type_: _description_
        """
        ok = False

        # Define profiles around the peak candidate
        coarse_width = 21
        coarse_box = pd.DataFrame(
            {"x": x[max(0, pk_ind - coarse_width) : min(len(x), pk_ind + coarse_width)]}
        )

        fine_width = 5
        fine_box = pd.DataFrame(
            {"x": x[max(0, pk_ind - fine_width) : min(len(x), pk_ind + fine_width)]}
        )

        # Rolling avg, std
        roll_w = 3
        coarse_box["avg"] = coarse_box["x"].rolling(roll_w, center=True).mean()
        coarse_box["std"] = coarse_box["x"].rolling(roll_w, center=True).std()

        # The algorithm
        fine_std = fine_box["x"].std()
        coarse_std_avg = coarse_box["std"].mean()
        coarse_std_std = coarse_box["std"].std()

        # Condition to accept candidate
        ok = fine_std > (coarse_std_avg + 1.1 * coarse_std_std)

        return ok, fine_std, coarse_std_avg, coarse_std_std

    # The Method
    logger.debug("Start Method Simple")
    method = "simple"
    # Use normalized RCS
    rcs_norm = normalize(rcs)
    # Focus on intense peaks
    peak_candidates, out_dict = signal.find_peaks(rcs_norm, height=0.98)
    logger.debug("End Method Simple")

    # Check behavior at neighbourhood to decide
    if check:
        logger.debug("Start Check Peak Candidates")
        peak_ind = []
        # methodology based on standard deviation
        for pk in peak_candidates:
            ok = check_candidate_peak(rcs_norm, pk)
            if ok:
                peak_ind.append(pk)
        logger.debug("End Check Peak Candidates")
    else:
        peak_ind = peak_candidates

    if plot_results:
        f, ax = plt.subplots()
        ax.plot(rcs_norm, ranges, label="rcs norm")
        ax.plot(rcs_norm[peak_candidates], ranges[peak_candidates], "o", color="r")
        ax.legend()
        ax.set_title(method)

    return peak_ind


def find_peaks(
    rcs_2d, ranges, min_range=None, max_range=None, method="default", parallel=False
):
    """Simple Method based on scipy.signal.find_peaks
    Useful for Strong Layers

    Args:
        rcs_2d (_type_): _description_
        parallel (bool, optional): _description_. Defaults to True.

    Returns:
        layers (_type_): dictionary for each rcs profiles with layer detection info
    """

    logger.debug("Start Method Find Peaks")

    if method == "default":
        method = "simple"

    # Check rcs_2d is really 2D
    ndims = rcs_2d.ndim
    if ndims == 1:
        rcs_2d = rcs_2d[np.newaxis, :]

    # Clip Range
    if not min_range:
        min_range = ranges[0]
    if not max_range:
        max_range = ranges[-1]
    idx = np.logical_and(ranges >= min_range, ranges <= max_range)
    rcs_2d = rcs_2d[:, idx]
    ranges = ranges[idx]

    # Turn off Parallel if few profiles
    dim_p, dim_h = rcs_2d.shape
    if dim_p > 2000:
        parallel = True

    # Find layers: If many, takes first
    layers = {}
    if parallel:
        with Pool(os.cpu_count()) as pool:
            input = [(x, ranges) for x in rcs_2d]
            peaks_arr = np.array(pool.starmap(find_peaks_1d, input))
        for t, peaks in enumerate(peaks_arr):
            exist = False
            if len(peaks) > 0:
                exist = True
            layers[t] = {"exist": exist, "ranges": ranges[peaks]}
    else:
        for t in range(dim_p):
            peaks = find_peaks_1d(rcs_2d[t, :], ranges)
            exist = False
            if len(peaks) > 0:
                exist = True
            layers[t] = {"exist": exist, "ranges": ranges[peaks]}

    logger.debug("End Method Find Peaks")
    return layers


def detect_layer(
    rcs_2d, ranges, min_range=None, max_range=None, method="find_peaks", **kwargs
):
    """_summary_

    Args:
        rcs_2d (_type_): _description_
        ranges (_type_): _description_
        min_range (_type_, optional): _description_. Defaults to None.
        max_range (_type_, optional): _description_. Defaults to None.
        method (str, optional): _description_. Defaults to 'find_peaks'.
    """

    logger.info("Start Layer Detection")
    if method == "find_peaks":
        layers = find_peaks(rcs_2d, ranges, min_range=min_range, max_range=max_range)
    else:
        logger.critical("Method %s does not exist" % method)
        layers = None

    logger.info("End Layer Detection")
    return layers
