import numpy as np
import xarray as xr
import datetime as dt
import dask.array as da
from loguru import logger

from lidarpy.utils.file_manager import channel2info

""" AD-HOC CORRECTIONS
+ pc peak correction: MULHACEN
"""


def mulhacen_pc_peak_correction(signal):
    """
    Correction of the PC peaks in the PC channels caused by PMT degradation of MULHACEN
    Parameters
    ----------
    signal: array numpy
        lidar signal uncorrected from pc peaks 1D, 2D array (time, range)
    Returns
    -------
    signal_new: array numpy
        lidar signal pc peaks corrected. 2D array (time, range) [numpy.array]
    """

    print("INFO. Start PC Peak Correction")

    # force to 2D
    is_1d = False
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
        is_1d = True

    # threshold for delta bin:
    threshold = 1000

    new_signal = signal.copy()
    for i in range(signal.shape[0]):
        try:
            profile = np.squeeze(signal[i, :])
            # Obsolete
            # if (profile > 1000).any():
            #    indexes = np.arange(profile.size)
            #    idx_diff = indexes[profile > 200]
            #    for idx_ in idx_diff:
            #        if idx_ < 2:
            #            datacorr = 0.8 * np.median(profile[:10], axis=0)
            #        else:
            #            datacorr = 2 * profile[idx_ - 1] - profile[idx_ - 2]
            #        dif = profile[idx_] - datacorr
            #        profile[idx_] = datacorr
            #        if (idx_ + 1) in idx_diff:
            #            k = 1
            #            while (idx_ + k) in idx_diff:
            #                profile[idx_ + k] = profile[idx_ + k] - dif
            #                k += 1
            #        else:
            #            k = 0
            #        idx_ += k

            diff = np.diff(profile)
            if (diff > threshold).any():
                indexes = np.arange(diff.size)
                idx_diff = indexes[diff > threshold]  # With respect to diff array
                idx_profile = idx_diff + 1  # With respect to profile array
                for idx_ in idx_profile:
                    datacorr = np.mean(profile[idx_ - 3 : idx_ - 1])
                    profile[idx_] = datacorr
                    k = 1
                    while (idx_ + k) < profile.size and (
                        profile[idx_ + k] - profile[idx_]
                    ) > threshold:
                        datacorr = np.mean(profile[idx_ + k - 3 : idx_ + k - 1])
                        profile[idx_ + k] = datacorr
                        k += 1
            profile[
                0:3
            ] = 0  # TODO: Preguntar a JABA el beneficio de poner los primeros bines a 0
            new_signal[i, :] = profile
        except Exception:
            print("pc peak correction not performed for profile %d-th" % i)
    # if 1d:
    if is_1d:
        new_signal = new_signal.ravel()

    print("INFO. End PC Peak Correction")

    return new_signal


""" DARK SIGNAL
"""


def subtract_dark_current(signal, dc):
    """ """
    # set flag default:
    dark_corrected_flag = False
    # same length of range dimension
    do = False
    if dc is not None:
        # DC must be numpy
        dc = dc.astype(np.float64)
        """
        if isinstance(dc, int):
            dc = np.int64(dc)
        elif isinstance(dc, float):
            dc = np.float64(dc)
        """

        if not np.isnan(dc).all():
            if signal.ndim == 1:
                if dc.ndim == 1:
                    if len(signal) == len(dc):
                        do = True
                elif dc.ndim == 0:
                    do = True
                else:
                    print("ERROR. Wrong dimensions")
            elif signal.ndim == 2:
                if dc.ndim == 2:
                    if signal.shape == dc.shape:
                        do = True
                elif dc.ndim == 1:
                    if signal.shape[1] == len(dc):
                        do = True
                elif dc.ndim == 0:
                    do = True
                else:
                    print("ERROR. Wrong dimensions")
            elif signal.ndim == 0:
                if dc.ndim == 0:
                    do = True
            else:
                print("ERROR. Wrong dimensions")

        if do:
            signal = signal - dc
            dark_corrected_flag = True
        else:
            print("WARNING. DC not subtracted")

    return signal, dark_corrected_flag


""" BACKGROUND
"""


def estimate_background(rs, idx_min, idx_max, dc=None):
    """
    Background Signal is estimated by considering values in the very far range,

    Parameters
    ----------
    rs: numpy.array, xarray.Dataarray
        Signal: 1D, 2D array (time, range)
    idx_min: Index of Min Height for Background. int
    idx_max: Index of Max Height for Background. int
    dc: numpy.array, xarray.Dataarray
        DC Signal. 1D array (range)

    Returns
    -------
    bg: Background value. 1D array (time)
    """

    logger.info("Start Estimate Background")
    if rs.ndim == 1:
        rs = rs[np.newaxis, :]
    try:
        # Method: Average RS values over BG height range for each profile
        if dc is None:
            bg = np.nanmean(rs[:, idx_min : idx_max + 1], axis=1)
        else:
            bg = np.nanmean(
                rs[:, idx_min : idx_max + 1] - dc[idx_min : idx_max + 1], axis=1
            )
    except Exception as e:
        bg = None
        logger.error("background cannot be estimated")
        raise RuntimeError(e)
    logger.info("End Estimate Background")

    return bg


def subtract_background(signal, bg):
    """Subtract Background signal to Raw signal

    Parameters
    ----------
    signal : array
        raw signal
    bg : float
        background value

    Returns
    -------
    [type]
        [description]
    """

    do = False

    # BG must be numpy.float64 (it has ndim)
    if isinstance(bg, int):
        bg = np.int64(bg)
    elif isinstance(bg, float):
        bg = np.float64(bg)

    if signal.ndim == 1:
        if bg.ndim == 0:
            do = True
        else:
            print("ERROR. Wrong dimensions")
    elif signal.ndim == 2:
        if bg.ndim == 1:
            if signal.shape[0] == len(bg):
                do = True
        elif bg.ndim == 0:
            do = True
        else:
            print("ERROR. Wrong dimensions")
    elif signal.ndim == 0:
        if bg.ndim == 0:
            do = True
    else:
        print("ERROR. Wrong dimensions")

    if do:
        signal = (signal.T - bg).T
    else:
        print("ERROR. BG not subtracted")

    return signal


def apply_bin_zero_correction(sg, delay):
    """
    TODO: desacoplar dependencia con tipo de variable (numpy/dask array)

    Parameters
    ----------
    sg: array
        signal (range), (time, range)
    delay: float
        position of bin zero (>0, <0)

    Returns
    -------
    sg_c: array
        signal corrected from bin zero

    """

    if not isinstance(delay, int):
        delay = int(delay)

    if isinstance(sg, np.ndarray):
        is_1d = False
        if sg.ndim == 1:
            sg = sg[np.newaxis, :]
            is_1d = True

        sg_c = sg.copy()
        try:
            if delay > 0:
                sg_c[:, :-delay] = sg[:, delay:]
                sg_c[:, -delay:] = np.nan
            elif delay < 0:
                sg_c[:, -delay:] = sg[:, :delay]
                sg_c[:, :-delay] = np.nan
        except Exception as e:
            print("ERROR. In apply_bin_zero_correction. %s" % str(e))
        if is_1d:
            sg_c = sg_c.ravel()
    else:
        if delay > 0:
            sg_c = da.concatenate(
                [sg[:, delay:], da.zeros((sg.shape[0], delay)) * np.nan], axis=1
            )
        elif delay < 0:
            sg_c = da.concatenate(
                [da.zeros((sg.shape[0], abs(delay))) * np.nan, sg[:, :delay]], axis=1
            )
        else:
            sg_c = sg

    return sg_c


""" DEAD TIME
"""


def apply_dead_time_correction(pc, tau, system=0):
    """
    Application of DEAD TIME correction over PC signal

    Parameters
    ----------
    pc: array
        photoncounting signal in MHz
    tau: float
        dead time in ns
    system: int
        paralyzable (1), nonparalyzable (0)

    Returns
    -------
    c: array
        pc signal corrected from dead time. 1D (2D) array (range (time, range))

    """

    try:
        # tau from ns to us
        tau_us = tau * 1e-3
        if system == 0:  # NON-PARALYZABLE
            # Eq 4 [D'Amico et al., 2016]
            c = pc / (1 - pc * tau_us)
        elif system == 1:  # PARALYZABLE
            # To be derived from Eq (2) [D'Amico et al., 2016]. Non-analytic
            c = pc
            print("WARNING: PARALYZABLE NOT IMPLEMENTED. No correction is applied")
        else:
            c = pc
            print("WARNING: wrong system for dead time correction. None is applied")
        # No infinites nor negative values
        c = np.where(np.logical_or(np.isinf(c), c < 0), np.nan, c)

    except Exception as e:
        print("ERROR. In apply_dead_time_correction %s" % str(e))
        c = pc * np.nan

    return c


""" PREPROCESSING SIGNAL
"""

def preprocessing_analog_signal(
    signal,
    dc,
    bz,
    idx_min,
    idx_max,
    dc_flag=True,
    zerobin_flag=True,
    bg_flag=True,
    workflow=0,
):
    """

    Parameters
    ----------
    signal: Raw Signal. Measured. 1D, 2D array (time (rs), range)
    dc: Dark Current Signal. Measured. 1D, 2D array (time (dc), range)
    bz: Bin Zero. scalar.
    bg: Background. Pre processed. 1D array (time (rs))
    zerobin_flag: activate/desactivate zero-bin correction. bool
    bg_flag: activate/desactivate background correction. bool
    workflow: type of workflow. Scalar. 0: SCC; 1: BG before ZB

    Returns
    -------
    signal: Preprocessed Signal. 2D array (time (rs), range)

    """

    logger.info("Start Analog Preprocessing")
    try:
        # force to 2D
        is_1d = False
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
            is_1d = True

        # Type of workflow
        if np.logical_or(workflow < 0, workflow > 1):
            workflow = 0  # SCC

        # Workflow: ZERO_BIN[(RAW - DC)] -  BG
        # 0. Estimate Background
        bg = estimate_background(signal, idx_min, idx_max, dc=dc)

        # _, ax = plt.subplots()
        # ax.plot(signal.mean(axis=0), label='RAW')

        # 1. Subtract DC from AN and BG
        dark_corrected = False
        if dc_flag:
            signal, dark_corrected = subtract_dark_current(signal, dc)
            # dc_signal = np.array(signal)
            # ax.plot(dc_signal.mean(axis=0), label='DC')
        if workflow == 0:  # SCC
            if zerobin_flag:
                # 2. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)
                # # ax.plot(signal.mean(axis=0), label='BZ')
                # zb_signal = np.array(signal)
            if bg_flag:
                # 3. Subtract Background (which has been subtracted from DC)
                signal = subtract_background(signal, bg)
                # ax.plot(signal.mean(axis=0), label='BG')
                # bg_signal = np.array(signal)
        else:  # Indistinguible de SCC
            if bg_flag:
                # 2. Subtract Background (which has been subtracted from DC)
                signal = subtract_background(signal, bg)
            if zerobin_flag:
                # 3. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)
        # ax.legend()
        # plt.show()
        # pdb.set_trace()
    except Exception as e:
        logger.critical(str(e))
        logger.critical("Signal not pre-processed.")
        raise RuntimeError("Sinal not preprocessed")

    if is_1d:
        signal = signal.ravel()

    logger.info("End Analog Preprocessing")
    return signal, bg, dark_corrected


def preprocessing_photoncounting_signal(
    signal, tau, bz, idx_min, idx_max, deadtime_flag=True, zerobin_flag=True, workflow=0
):
    """

    Parameters
    ----------
    signal: Raw Signal. 2D array (time (rs), range)
    tau: Dead Time (ns). Scalar
    bz: zero bin for photoncounting (delay_an + bin_shift). Scalar.
    bg: Background. 1D array (time (rs))
    peak_correction: Apply peak correction. Bool
    workflow: type of workflow. Scalar. 0: SCC; 1: BG before ZB

    Returns
    -------
    signal: Preprocessed Signal. 2D array (time (rs), range)
    """

    try:
        # Force to 2D
        is_1d = False
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
            is_1d = True

        # Type of workflow
        if np.logical_or(workflow < 0, workflow > 1):
            workflow = 0  # SCC

        # Workflow SCC: ZERO_BIN[DT(PK(RAW))] -  BG
        # 0. Estimate Background
        bg = estimate_background(signal, idx_min, idx_max)

        if deadtime_flag:
            # Dead Time Correction
            # 1. Apply Dead Time Correction
            signal = apply_dead_time_correction(signal, tau)

        if workflow == 0:  # SCC
            if zerobin_flag:
                # 2. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)
            # 3. Subtract Background as estimated from Raw Signal
            signal = subtract_background(signal, bg)
        else:
            # 2. Subtract Background as estimated from Raw Signal
            signal = subtract_background(signal, bg)
            if zerobin_flag:
                # 3. Apply Trigger Delay
                signal = apply_bin_zero_correction(signal, bz)

    except Exception as e:
        print(str(e))
        print("signal not pre-processed.")

    if is_1d:
        signal = signal.ravel()

    return signal, bg


def ff_2D_overlap_from_channels(
    lidar_dataset: xr.Dataset,
    channel_ff: str,
    channel_nf: str,
    norm_range: tuple[float, float] = (2500, 3500),
    rel_dif_threshold: float = 2.5,
    force_to_one_when_full_overlap: bool = False,
) -> xr.DataArray:

    #Select window size
    if lidar_dataset.time.size <= 15:
        window_size = 3
    elif lidar_dataset.time.size > 15 and lidar_dataset.time.size <= 30:
        window_size = 5
    elif lidar_dataset.time.size > 30 and lidar_dataset.time.size <= 45:
        window_size = 7
    else:
        window_size = 11

    info_ff = channel2info(channel_ff)
    info_nf = channel2info(channel_nf)

    #Check if channels have the same wavelength
    if info_nf[0] != info_ff[0]:
        raise ValueError(
            f"Channels {channel_ff} and {channel_nf} must have the same wavelength"
        )

    # Select time, range to normalize and range to calculate overlap
    nf = lidar_dataset[f"signal_{channel_nf}"] / lidar_dataset[
        f"signal_{channel_nf}"
    ].sel(range=slice(*norm_range)).mean("range")
    ff = lidar_dataset[f"signal_{channel_ff}"] / lidar_dataset[
        f"signal_{channel_ff}"
    ].sel(range=slice(*norm_range)).mean("range")

    # Make the ratio avoiding division by zero
    overlap_raw = np.divide(ff.values, nf.values, out=np.ones_like(ff.values), where=np.logical_and(nf.values != 0, ff.values != 0))
    

    overlap_raw = xr.DataArray(
        overlap_raw, dims=("time", "range"), coords={"time": lidar_dataset.time, "range": lidar_dataset.range}
    )

    #Lower limit        
    rel_dif = 100 * (nf - ff / overlap_raw) / nf
    overlap_raw = overlap_raw.where(rel_dif < rel_dif_threshold, other=np.nan)
    
    overlap = overlap_raw.copy()
    
    # Set to 1.0 when full overlap is reached
    if force_to_one_when_full_overlap:
        #Find when full overlap is reached    
        full_overlap_reached_range = overlap.range[(overlap > 1.).argmax(dim="range")]
        overlap = overlap.where(overlap.range <= full_overlap_reached_range, other=1.0)
    
    # Assign attributes
    attrs = ["location", "system"]
    for attr_ in attrs:
        overlap.attrs[attr_] = lidar_dataset.attrs[attr_]

    overlap.attrs["history"] = dt.datetime.now().strftime(
        "Created %a %b %d %H:%M:%S %Y"
    )
    overlap.attrs["wavelength"] = info_nf[0]
    overlap.attrs["channel_ff"] = channel_ff
    overlap.attrs["channel_nf"] = channel_nf
    return overlap
