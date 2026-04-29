from asyncio.log import logger
from pdb import set_trace
import numpy as np
from scipy.signal import savgol_filter

from lidarpy.general_utils import utils

""" GLUING

D'Amico, G., Amodeo, A., Mattis, I., Freudenthaler, V., and Pappalardo, G.: EARLINET Single Calculus Chain -
technical - Part 1: Pre-processing of raw lidar data, Atmos. Meas. Tech., 9, 491-507,
https://doi.org/10.5194/amt-9-491-2016, 2016.
"""


def estimate_first_range(
    analog_signal,
    photon_signal,
    ranges,
    photon_bg,
    photon_threshold,
    adc_range,
    adc_bits,
    n_res,
    correlation_threshold,
    min_points,
    full_overlap_range,
):
    """

    Args:
        analog_signal (_type_): preprocessed signal (NOT RCS)
            Signal from analog channel which should be had dark noise, background and trigger delay corrections.
        photon_signal (_type_): preprocessed signal (NOT RCS)
            Signal from photon counting channel, dead time corrected and convert to MHz.
        ranges (_type_): _description_
        photon_bg(_type_): Photoncounting Background (MHz)
        photon_threshold (_type_): MHz
        adc_range (_type_): _description_
            Full scale voltage (for calculation of the Least Significant bit).
        adc_bits (_type_): _description_
            Number of bits (for calculation of the Least Significant bit).
        n_res (_type_): _description_
            Number of times above the LSB that we trust the analog signal. c.f. equation (12) of D'Amico et al.
        correlation_threshold (_type_): _description_
        min_points (_type_): _description_
        full_overlap_range (_type_): _description_
            Full overlap range. Used to limit the search region of the maximum signal.

    Returns:
        _type_: _description_
    """
    # Initialize Output
    region_found = False

    # work with photon signal + photon_background
    photon_signal = photon_signal + photon_bg

    # estimation is performed if a threshold for pc is achieved
    pc_min = np.nanmin(photon_signal)
    if pc_min < photon_threshold:
        # smooth pc signal to avoid selecting noise as peak
        idx_finite = np.isfinite(photon_signal)
        smoothed_pc_signal = photon_signal * 0.0
        smoothed_pc_signal[idx_finite] = savgol_filter(
            photon_signal[idx_finite], 21, polyorder=2
        )

        # with PC, lower height is estimated
        idx_fo_up, z_fo_up = utils.find_nearest_1d(ranges, full_overlap_range[1])
        idx_fo_bo, z_fo_bo = utils.find_nearest_1d(ranges, full_overlap_range[0])
        idx_tmp = idx_fo_bo
        state = False
        while not state:
            if idx_tmp < idx_fo_up:
                if photon_signal[idx_tmp] > photon_threshold:
                    idx_tmp += 1
                else:
                    state = True
            else:
                state = True
        lower_idx = (
            idx_tmp  # np.nanargmax(photon_signal[idx_fo_bo:] < photon_threshold)
        )

        # with AN, upper height is estimated:
        # Smin = Smax / F
        # Smax = adc_range
        # F = (2^adc_bits - 1) / n_res; n_res = N times the resolution
        analog_threshold = n_res * adc_range / ((2**adc_bits) - 1)
        idxs = np.arange(len(analog_signal))
        upper_idx = idx_fo_bo + idxs[analog_signal[idx_fo_bo:] > analog_threshold][-1]

        # Region has enough bins
        if (upper_idx - lower_idx) > min_points:
            correlation = np.corrcoef(
                analog_signal[lower_idx:upper_idx], photon_signal[lower_idx:upper_idx]
            )[0, 1]
            # Linear Correlation is Good Enough
            set_trace()
            if correlation >= correlation_threshold:
                region_found = True
            else:
                logger.warning(
                    "Correlation test not passed. Correlation: {0}. Threshold {1}".format(
                        correlation, correlation_threshold
                    )
                )
        else:
            logger.warning(
                "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(
                    lower_idx, upper_idx
                )
            )
    else:  # wrong initial guess for photon_threshold
        logger.warning("wrong initial guess for photon_threshold")

    if not region_found:
        lower_idx, upper_idx = False, False

    return lower_idx, upper_idx, region_found, correlation


def calculate_residual_slope(analog_segment, photon_segment, use_photon_as_reference):
    """ """
    c_analog, c_photon = calculate_gluing_values(
        analog_segment, photon_segment, use_photon_as_reference
    )

    residuals = c_analog * analog_segment - c_photon * photon_segment

    fit_values, cov = np.polyfit(np.arange(len(residuals)), residuals, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def calculate_slope(analog_segment, photon_segment, use_photon_as_reference):
    """ """
    if use_photon_as_reference:
        fit_values, cov = np.polyfit(analog_segment, photon_segment, 1, cov=True)
    else:
        fit_values, cov = np.polyfit(photon_segment, analog_segment, 1, cov=True)

    k = fit_values[0]  # Slope
    dk = np.sqrt(np.diag(cov)[0])  # Check here: https://stackoverflow.com/a/27293227

    return k, dk


def optimize_with_slope_test(
    analog_signal,
    photon_signal,
    low_idx_start,
    up_idx_start,
    slope_threshold,
    step,
    use_photon_as_reference,
):
    """ """
    low_idx, up_idx = low_idx_start, up_idx_start

    glue_found = False
    first_round = True

    N = up_idx - low_idx

    while not glue_found and (N > 5):

        analog_segment = analog_signal[low_idx:up_idx]
        photon_segment = photon_signal[low_idx:up_idx]

        if N <= 30:
            k, dk = calculate_residual_slope(
                analog_segment, photon_segment, use_photon_as_reference
            )

            if np.abs(k) < slope_threshold * dk:
                glue_found = True
            else:
                # print("Changing indices")
                # update indices for next loop
                if first_round:
                    up_idx -= step
                else:
                    low_idx += step

                N = up_idx - low_idx

                # If first round finished without result, start the second round, increasing the lower bound.
                if (N <= 5) and first_round:
                    first_round = False
                    low_idx, up_idx = (
                        low_idx_start,
                        up_idx_start,
                    )  # Start searching from the start
                    N = up_idx - low_idx
        else:
            mid_idx = (up_idx - low_idx) // 2

            k1, dk1 = calculate_residual_slope(
                analog_segment[:mid_idx],
                photon_segment[:mid_idx],
                use_photon_as_reference,
            )
            k2, dk2 = calculate_residual_slope(
                analog_segment[mid_idx:],
                photon_segment[mid_idx:],
                use_photon_as_reference,
            )

            # print("Slope iteration: %s - %s. N > 30. K1: %s, DK1: %s, K2: %s, DK2: %s" % (low_idx, up_idx, k1, dk1, k2, dk2))

            if np.abs(k2 - k1) < slope_threshold * np.sqrt(dk1**2 + dk2**2):
                # print("Slope test passed!!!!")
                glue_found = True
            else:
                # print("Changing indices")
                # update indices for next loop
                if first_round:
                    up_idx -= step
                else:
                    low_idx += step

                N = up_idx - low_idx

                # If first round finished without result, start the second round, increasing the lower bound.
                if (N <= 5) and first_round:
                    first_round = False
                    low_idx, up_idx = (
                        low_idx_start,
                        up_idx_start,
                    )  # Start searching from the start
                    N = up_idx - low_idx

    if not glue_found:
        low_idx, up_idx = False, False
        print(
            "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(
                low_idx, up_idx
            )
        )

    return low_idx, up_idx, glue_found


def optimize_with_stability_test(
    analog_signal,
    photon_signal,
    low_idx_start,
    up_idx_start,
    stability_threshold,
    step,
    use_photon_as_reference,
):
    """ """

    # print('entering stability test')

    low_idx, up_idx = low_idx_start, up_idx_start

    glue_found = False
    N = up_idx - low_idx

    while not glue_found and (N > 5):

        analog_segment = analog_signal[low_idx:up_idx]
        photon_segment = photon_signal[low_idx:up_idx]

        mid_idx = (up_idx - low_idx) // 2

        k1, dk1 = calculate_slope(
            analog_segment[:mid_idx], photon_segment[:mid_idx], use_photon_as_reference
        )
        k2, dk2 = calculate_slope(
            analog_segment[mid_idx:], photon_segment[mid_idx:], use_photon_as_reference
        )

        # print("Stability iteration:  K1: %s, DK1: %s, K2: %s, DK2: %s" % (k1, dk1, k2, dk2))

        if np.abs(k2 - k1) < stability_threshold * np.sqrt(dk1**2 + dk2**2):
            glue_found = True
        else:
            # update
            low_idx += step
            up_idx -= step
            N = up_idx - low_idx

    if not glue_found:
        low_idx, up_idx = False, False
        print(
            "No suitable region found. Lower gluing idx: {0}. Upper gluing idx: {1}.".format(
                low_idx, up_idx
            )
        )

    return low_idx, up_idx, glue_found


def estimate_gluing_region(
    analog_signal,
    photon_signal,
    ranges,
    adc_range,
    adc_bits,
    n_res,
    photon_bg,
    photon_threshold,
    range_threshold=(1500, 6000),
    correlation_threshold=0.8,
    min_points=15,
    slope_test=True,
    slope_threshold=2,
    stability_test=True,
    stability_threshold=1,
    step=5,
    use_photon_as_reference=True,
):

    """_summary_

    Args:
        analog_signal (_type_): _description_
        photon_signal (_type_): _description_
        ranges (_type_): _description_
        adc_range (_type_): _description_
        adc_bits (_type_): _description_
        n_res (_type_): _description_
        photon_bg (_type_): _description_
        photon_threshold (_type_): _description_
        full_overlap_range (tuple, optional): _description_. Defaults to (1500, 15000).
        correlation_threshold (float, optional): _description_. Defaults to 0.8.
        min_points (int, optional): _description_. Defaults to 15.
        slope_test (bool, optional): _description_. Defaults to True.
        slope_threshold (int, optional): _description_. Defaults to 2.
        stability_test (bool, optional): _description_. Defaults to True.
        stability_threshold (int, optional): _description_. Defaults to 1.
        step (int, optional): _description_. Defaults to 5.
        use_photon_as_reference (bool, optional): _description_. Defaults to True.
    """

    # Initialize Output
    first_lower_idx, first_upper_idx, first_res = False, False, False
    slope_lower_idx, slope_upper_idx, slope_res = False, False, False
    stability_lower_idx, stability_upper_idx, stability_res = False, False, False
    lower_idx, upper_idx, glue = False, False, False

    # Stability Test is performed only after Slope Test
    if stability_test:
        slope_test = True
    set_trace()
    """ First Estimation of Gluing Range """
    first_lower_idx, first_upper_idx, first_res, _ = estimate_first_range( analog_signal, photon_signal, ranges, photon_bg, photon_threshold, adc_range, adc_bits, n_res, correlation_threshold, min_points, range_threshold, )
    set_trace()
    if first_res:
        """Slope Test"""
        if slope_test:
            slope_lower_idx, slope_upper_idx, slope_res = optimize_with_slope_test(
                analog_signal,
                photon_signal,
                first_lower_idx,
                first_upper_idx,
                slope_threshold,
                step,
                use_photon_as_reference,
            )
            if slope_res:
                """Stability Test"""
                if stability_test:
                    (
                        stability_lower_idx,
                        stability_upper_idx,
                        stability_res,
                    ) = optimize_with_stability_test(
                        analog_signal,
                        photon_signal,
                        slope_lower_idx,
                        slope_upper_idx,
                        stability_threshold,
                        step,
                        use_photon_as_reference,
                    )
                    if stability_res:
                        # Final Indices are from Stability Test
                        lower_idx, upper_idx = stability_lower_idx, stability_upper_idx
                    else:
                        # Final Indices are from Slope Test
                        lower_idx, upper_idx = slope_lower_idx, slope_upper_idx
                else:
                    # Final Indices are from Slope Test
                    lower_idx, upper_idx = slope_lower_idx, slope_upper_idx
        else:
            # Final Indices are from First Estimation
            lower_idx, upper_idx = first_lower_idx, first_upper_idx
    set_trace()
    if np.logical_and(lower_idx, upper_idx):
        glue = True

    # Save indices from different tests
    all_indices = {
        "region_range": {"status": glue, "indices": [lower_idx, upper_idx]},
        "first_range": {
            "status": first_res,
            "indices": [first_lower_idx, first_upper_idx],
        },
        "slope_range": {
            "status": slope_res,
            "indices": [slope_lower_idx, slope_upper_idx],
        },
        "stability_range": {
            "status": stability_res,
            "indices": [stability_lower_idx, stability_upper_idx],
        },
    }

    return lower_idx, upper_idx, glue, all_indices


def calculate_gluing_values(
    lower_gluing_region, upper_gluing_region, use_upper_as_reference
):
    """
    Calculate the multiplicative calibration constants for gluing the two signals.

    Parameters
    ----------
    lower_gluing_region: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_gluing_region: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    use_upper_as_reference: bool
       If True, the upper signal is used as reference. Else, the lower signal is used.

    Returns
    -------
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.
    """
    lower_gluing_region = (
        lower_gluing_region.ravel()
    )  # Ensure we have an 1D array using ravel
    upper_gluing_region = upper_gluing_region.ravel()

    # Find their linear relationship, using least squares
    slope_zero_intercept, _, _, _ = np.linalg.lstsq(
        lower_gluing_region[:, np.newaxis], upper_gluing_region
    )

    # Set the calibration constants
    if use_upper_as_reference:
        c_upper = 1
        c_lower = slope_zero_intercept
    else:
        c_upper = 1 / slope_zero_intercept
        c_lower = 1

    return c_lower, c_upper


def glue_signals_at_bins(
    lower_signal, upper_signal, min_bin, max_bin, c_lower, c_upper
):
    """
    Glue two signals at a given bin range.

    The signal can be either a 1D array or a 2D array with dimensions (time, range).

    Both signals are assumed to have the same altitude grid. The final glued signal is calculated
    performing a linear fade-in/fade-out operation in the glue region.

    Parameters
    ----------
    lower_signal: array
       The low-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    upper_signal: array
       The high-range signal to be used. Can be either 1D or 2D with dimensions (time, range).
    min_bin: int
       The lower bin to perform the gluing
    max_bin: int
       The upper bin to perform the gluing
    c_lower: float
       Calibration constant of the lower signal. It will be equal to 1, if `use_upper_as_reference` argument
       is False.
    c_upper: float
       Calibration constant of the upper signal. It will be equal to 1, if `use_upper_as_reference` argument
       is True.

    Returns
    -------
    glued_signal: array
       The glued signal array, same size as lower_signal and upper_signal.
    """
    # Ensure that data are 2D-like
    if lower_signal.ndim == 1:
        lower_signal = lower_signal[np.newaxis, :]  # Force 2D
        upper_signal = upper_signal[np.newaxis, :]  # Force 2D
        axis_added = True
    else:
        axis_added = False

    gluing_length = max_bin - min_bin

    lower_weights = np.zeros_like(lower_signal)

    lower_weights[:, :min_bin] = 1
    lower_weights[:, min_bin:max_bin] = 1 - np.arange(gluing_length) / float(
        gluing_length
    )

    upper_weights = 1 - lower_weights

    # Calculate the glued signal
    glued_signal = (
        c_lower * lower_weights * lower_signal + c_upper * upper_weights * upper_signal
    )

    # Remove dummy axis, if added
    if axis_added:
        glued_signal = glued_signal[0, :]
        lower_weights = lower_weights[0, :]
        upper_weights = upper_weights[0, :]

    return glued_signal, lower_weights, upper_weights


def gluing(
    analog_signal,
    photon_signal,
    ranges,
    photon_bg,
    adc_range,
    adc_bits,
    n_res=None,
    photon_threshold=None,
    range_threshold=None,
    correlation_threshold=None,
    min_points=None,
    slope_test=True,
    slope_threshold=2,
    stability_test=True,
    stability_threshold=1,
    step=5,
    use_photon_as_reference=True,
):
    """_summary_

    Args:
        analog_signal (_type_): analog signal dc-, bg-, bz- corrected
        photon_signal (_type_): photoncounting signal bg-, dt- corrected
        ranges (_type_): _description_
        photon_bg (_type_): _description_
        adc_range (_type_): _description_
        adc_bits (_type_): _description_
        n_res (_type_, optional): _description_. Defaults to None.
        pc_threshold (_type_, optional): _description_. Defaults to None.
        correlation_threshold (_type_, optional): _description_. Defaults to None.
        range_threshold (_type_, optional): _description_. Defaults to None.
        min_points (_type_, optional): _description_. Defaults to None.
        slope_test (bool, optional): _description_. Defaults to True
        slope_threshold (int, optional): _description_. Defaults to 2.
        stability_test (bool, optional): _description_. Defaults to True
        stability_threshold (int, optional): _description_. Defaults to 1.
        step (int, optional): _description_. Defaults to 5.
        use_photon_as_reference (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    # Defaults
    if n_res is None:
        # F = 5000
        # n_res = (2**adc_bits - 1)/F  # D'Amico et al., 2016, Eq.12
        n_res = 50  # --> Smin ~ 1e-3
    if photon_threshold is None:
        photon_threshold = 20
    if range_threshold is None:
        range_threshold = (1000, 6000)
    if correlation_threshold is None:
        correlation_threshold = 0.8
    if min_points is None:
        min_points = 15

    try:
        """Estimate Gluing Region"""
        lower_idx, upper_idx, glue, all_indices = estimate_gluing_region(
            analog_signal,
            photon_signal,
            ranges,
            adc_range,
            adc_bits,
            n_res,
            photon_bg,
            photon_threshold,
            range_threshold=range_threshold,
            correlation_threshold=correlation_threshold,
            min_points=min_points,
            slope_test=slope_test,
            slope_threshold=slope_threshold,
            stability_test=stability_test,
            stability_threshold=stability_threshold,
            step=step,
            use_photon_as_reference=use_photon_as_reference,
        )

        """ Calculate Glued Profile """
        if glue:
            c_analog, c_photon = calculate_gluing_values(
                analog_signal[lower_idx:upper_idx],
                photon_signal[lower_idx:upper_idx],
                use_photon_as_reference,
            )
            glued_signal, w_analog, w_photon = glue_signals_at_bins(
                analog_signal, photon_signal, lower_idx, upper_idx, c_analog, c_photon
            )
    except Exception as e:
        glue = False
        logger.warning("Signal not glued. %s" % str(e))
    if not glue:
        glued_signal, c_analog, c_photon, w_analog, w_photon, all_indices = (
            ranges * np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            {},
        )

    return glued_signal, c_analog, c_photon, w_analog, w_photon, all_indices, glue
