from pathlib import Path
from loguru import logger
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from lidarpy.general_utils.io import read_yaml_from_info
from lidarpy.utils.file_manager import (
    add_required_channels,
    filename2info,
    search_dc,
)
from lidarpy.general_utils import calibration
from lidarpy.preprocessing.gluing_de_la_rosa_slope import gluing
from lidarpy.general_utils.smoothing import (
    BIN_RES_M,
    bin_rescale,
    moving_average,
    sliding_average,
    estimate_snr,
    adaptive_sliding_average,
    smoothing_ATLAS,
)

from lidarpy.preprocessing.lidar_preprocessing_tools import (
    ff_2D_overlap_from_channels,
)
# NOTE: in your original snippet you had:
# from lidarpy.general_utils.smoothing import
# which was incomplete, so I removed it.


# ---------------------------------------------------------------------
# DEBUG HELPER
# ---------------------------------------------------------------------
def _debug_dump_signals(
    dataset: xr.Dataset,
    step: str,
    channels: list[str] | None = None,
    max_range: float = 2000.0,
    time_index: int = 0,
) -> None:
    """
    Compact debug print of the signals at a given pipeline step.

    What it shows:
    - 1 time index (default: 0)
    - range from 0 to `max_range` meters
    - for every channel: shape, min, max, mean

    This is intentionally light: you can enable it on every run.
    """
    if "channel" not in dataset:
        logger.debug(f"[{step}] dataset has no 'channel' coord.")
        return

    chan_list = channels or dataset.channel.values.tolist()
    rsel = dataset.range.sel(range=slice(0, max_range))

    logger.debug(f"[{step}] --- DEBUG SNAPSHOT ---")
    logger.debug(f"[{step}] time index: {time_index}, range: 0–{max_range} m")

    for ch in chan_list:
        sig_name = f"signal_{ch}"
        if sig_name not in dataset:
            logger.debug(f"[{step}] {sig_name} not in dataset → skipped.")
            continue
        da_1d = dataset[sig_name].isel(time=time_index).sel(range=rsel.range)
        arr = da_1d.values
        logger.debug(
            f"[{step}] {sig_name}: shape={arr.shape}, "
            f"min={np.nanmin(arr):.3e}, max={np.nanmax(arr):.3e}, mean={np.nanmean(arr):.3e}"
        )

    logger.debug(f"[{step}] ----------------------")

def _build_var_aggs_for_binning(dataset: xr.Dataset) -> dict:
    """
    Devuelve un dict {var_name: 'sum'|'mean'} para pasar a bin_rescale.
    - 'sum' en señales de canales photon-counting (*p) → cuentas extensivas.
    - 'mean' en el resto (intensivo por defecto).
    """
    var_aggs: dict[str, str] = {}

    # Si tienes 'stderr_*' y quieres rebinnearlos, lo más correcto sería
    # recalcularlos tras el binning. Aquí los dejamos fuera (no los tocamos).
    if "channel" in dataset.coords:
        chs = [str(c) for c in dataset.channel.values]
        for ch in chs:
            v = f"signal_{ch}"
            if v in dataset.data_vars:
                # photon counting si el canal termina en 'p'
                var_aggs[v] = "sum" if ch.endswith("p") else "mean"

    # Si tienes otras variables extensivas que quieras sumar explícitamente,
    # puedes añadirlas aquí:
    # for vname in ["photon_counts", "bg_counts"]:
    #     if vname in dataset.data_vars:
    #         var_aggs[vname] = "sum"

    return var_aggs

def preprocess(
    file_or_path: Path | str,
    channels: list[str] | None = None,
    crop_ranges: tuple[float, float] | None = (0, 20000),
    background_ranges: tuple[float, float] | None = None,

    # 1) FIRST: possible rebinning (range-resolution reduction)
    apply_bin: bool = False,
    bin_factor: int = 2,

    # 3) MAIN PHYSICAL/INSTRUMENTAL CORRECTIONS
    apply_dc: bool = True,
    apply_dt: bool = True,
    apply_bg: bool = True,
    apply_bz: bool = True,
    apply_ov: bool = False,
    save_dc: bool = False,
    save_bg: bool = False,
    gluing_products: bool = False,
    force_dc_in_session: bool = False,
    overlap_path: Path | str | None = None,

    # 5) LAST: smoothing
    apply_sm: bool = False,
    smooth_mode: str | list[str] | None = None,

    # DEBUG
    debug: bool = False,
    debug_max_range: float = 2000.0,

    **kwargs,
) -> xr.Dataset:
    """
    Lidar preprocessing pipeline – **ordered**.

    The idea of this pipeline is: do **geometry/resolution changes first**,
    then **select the channels** you really want to process, then **apply
    all instrumental/physical corrections in a strict order**, and only
    then do **post-processing** (height, cropping, gluing) and finally
    **smoothing** (because smoothing must see the final, corrected signals).

    Processing order
    ----------------
    1. (Optional) **Rebin / range rescale** (`apply_bin`, `bin_factor`)
       - This changes the vertical resolution and MUST be done before
         any correction that depends on the range grid (DC, BG, overlap).
       - If the main dataset is rebinned, every auxiliary dataset
         used later (DC, overlap) must be rebinned to the same resolution.

    2. **Channel selection + INFO update**
       - Decide which channels to keep.
       - Add any required/product channels defined in the system INFO.
       - Drop variables for channels you do not want to process.
       - Update dataset from metadata/header, if present.

    3. **Main corrections (strict order!)**
       - DC  (dark current)          → removes dark offset, usually analog
       - DT  (dead-time correction)  → for photon-counting channels
       - BG  (background subtraction)→ removes far-range night-sky background
       - BZ  (bin-zero correction)   → shifts profiles to correct laser–detector delay
       - OV  (overlap correction)    → corrects geometric overlap losses
       The order DC → DT → BG → BZ → OV is important because:
         * DC must be removed before everything else.
         * DT must act on the *raw photon count*.
         * BG must act on a *physically meaningful* profile.
         * BZ shifts the profile in range, so it must be applied
           when the signal is already corrected.
         * OV is usually the last purely range-dependent correction.

    4. **Post-processing**
       - Crop to the desired range (for plots/products).
       - Optionally glue analog & photon-counting channels.
       - Add height coordinate from range, taking zenith angle into account.

    5. (Optional) **Smoothing**
       - Apply only at the end, on the fully corrected product.
       - Several modes are available: gaussian, moving, sliding, ATLAS, adaptive_sliding.
       - Gaussian smoothing uses `scipy.ndimage.gaussian_filter1d` **along `range`**.

    Debugging
    ---------
    If `debug=True`, the function will print a small snapshot of the signals
    (time=0, range=0–`debug_max_range`) **after every step** above, in the
    same order that is documented here. That way you can verify that each
    correction is doing what you expect.

    Parameters
    ----------
    file_or_path : Path | str
        NetCDF file to be opened.
    channels : list[str] | None
        Channels to keep/process. If None, all channels in the file are used.
    crop_ranges : (float, float) | None
        Range interval to keep *after* corrections.
    background_ranges : (float, float) | None
        Altitude interval for background estimation; if None, the values
        from the dataset attributes are used.
    apply_* : bool
        Switches for every step in the pipeline.
    apply_sm : bool
        If True, smoothing is applied as the *last* step.
    smooth_mode : str | list[str] | None
        One or multiple smoothing modes to apply, in order.
        Passed to `apply_smooth`.
    debug : bool
        If True, show signal snapshots after every step.
    debug_max_range : float
        Upper limit (m) of range shown in debug snapshots.

    Returns
    -------
    xr.Dataset
        Fully preprocessed lidar dataset.
    """
    p = Path(file_or_path)
    if not p.exists():
        raise ValueError(f"{file_or_path} not found")

    # -------------------------------------------------------------
    # Read lidar info from filename and associated YAML
    # -------------------------------------------------------------
    lidar_nick, _, _, _, _, date = filename2info(p.name)
    global INFO
    INFO = read_yaml_from_info(lidar_nick, date)

    # open dataset (no lazy chunks here, we want to modify it later)
    # Load fully into memory to avoid keeping file handles open and
    # calling into netCDF/HDF5 libraries from multiple threads later.
    dataset = xr.open_dataset(p, chunks={}, engine="netcdf4").load()

    if debug:
        _debug_dump_signals(dataset, "0-open", channels, max_range=debug_max_range)

    # =============================================================
    # 1) REBIN (FIRST STEP)
    # =============================================================
    if apply_bin and bin_factor > 1:
        logger.info(
            f"Applying range bin rescale: factor={bin_factor} → "
            f"{bin_factor * BIN_RES_M:.2f} m"
        )
        # sum para cuentas (canales *p), mean para el resto
        var_aggs = _build_var_aggs_for_binning(dataset)

        dataset = bin_rescale(
            dataset,
            bin_factor,
            default_agg="mean",
            var_aggs=var_aggs,
            bin_coord="center",  # o "left"/"right" si prefieres ejes de borde
        )
    else:
        # guarda resolución actual aunque no se rebine
        dataset.attrs["rebin_factor"] = 1
        dataset.attrs["range_resolution_m"] = float(BIN_RES_M)

    if debug:
        _debug_dump_signals(dataset, "1-rebin", channels, max_range=debug_max_range)

    # =============================================================
    # 2) CHANNEL SELECTION / INFO UPDATE
    # =============================================================
    if channels is None:
        # process all channels found in the file
        channels = dataset.channel.values.tolist()
        raw_channels = channels
    else:
        # ensure we also keep the "required" channels defined in the info file
        channels = add_required_channels(lidar_nick, channels, date)
        # product channels must not be dropped, but raw signals might be
        raw_channels = [
            ch for ch in channels if ch not in INFO["product_channels"].keys()
        ]
        dataset = drop_unwanted_channels(dataset, keep_channels=raw_channels)

    # update variables/attrs from INFO->header, if present
    if "header" in INFO.keys():
        dataset = update_from_info(dataset, INFO["header"])

    if debug:
        _debug_dump_signals(dataset, "2-ch-selection", channels, max_range=debug_max_range)

    # =============================================================
    # 3) MAIN CORRECTIONS – in fixed order
    # =============================================================
    # 3.1 Dark current
    if apply_dc:
        dataset = apply_dark_current_correction(
            dataset,
            rs_path=p,
            save_dc=save_dc,
            force_dc_in_session=force_dc_in_session,
        )
    else:
        dataset.attrs["dc_corrected"] = str(False)

    if debug:
        _debug_dump_signals(dataset, "3.1-after-DC", channels, max_range=debug_max_range)

    # 3.2 Dead-time (photon-counting channels)
    if apply_dt:
        dataset = apply_dead_time_correction(dataset)
    else:
        dataset.attrs["dt_corrected"] = str(False)

    if debug:
        _debug_dump_signals(dataset, "3.2-after-DT", channels, max_range=debug_max_range)

    # 3.3 Background (far-range) subtraction
    if apply_bg:
        dataset = apply_background_correction(
            dataset,
            background_ranges=background_ranges,
            save_bg=save_bg,
        )
    else:
        dataset.attrs["bg_corrected"] = str(False)

    if debug:
        _debug_dump_signals(dataset, "3.3-after-BG", channels, max_range=debug_max_range)

    # 3.4 Bin-zero (range shift) correction
    if apply_bz:
        dataset = apply_bin_zero_correction(dataset, rs_path=p)
    else:
        dataset.attrs["bz_corrected"] = str(False)

    if debug:
        _debug_dump_signals(dataset, "3.4-after-BZ", channels, max_range=debug_max_range)

    # 3.5 Overlap correction (optional)
    if apply_ov:
        dataset = apply_overlap_correction(dataset, overlap_path)
        dataset.attrs["ov_corrected"] = "True"
    else:
        dataset.attrs["ov_corrected"] = "False"

    if debug:
        _debug_dump_signals(dataset, "3.5-after-OV", channels, max_range=debug_max_range)

    # =============================================================
    # 4) POST PROCESSING
    # =============================================================
    # 4.1 Crop to user-defined range
    dataset = apply_crop_ranges_correction(dataset, crop_ranges=crop_ranges)

    if debug:
        _debug_dump_signals(dataset, "4.1-after-crop", channels, max_range=debug_max_range)

    # 4.2 Optional: merge / glue detection modes (analog + pc → g)
    if gluing_products:
        dataset = apply_detection_mode_merge(dataset)
        if debug:
            # show also newly created glued channels
            _debug_dump_signals(dataset, "4.2-after-gluing", None, max_range=debug_max_range)

    # 4.3 Add height coordinate (range * cos(zenith))
    dataset = add_height(dataset)

    if debug:
        _debug_dump_signals(dataset, "4.3-after-height", channels, max_range=debug_max_range)

    # =============================================================
    # 5) SMOOTHING (LAST)
    # =============================================================
    if apply_sm:
        logger.info(f"Applying smoothing: mode={smooth_mode}")
        dataset = apply_smooth(
            dataset,
            smooth_mode=smooth_mode,
            debug=debug,
            debug_max_range=debug_max_range,
            **kwargs,
        )
    else:
        logger.info("No smoothing applied.")

    if debug:
        _debug_dump_signals(dataset, "5-after-smoothing", channels, max_range=debug_max_range)

    return dataset


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def update_from_info(dataset: xr.Dataset, header: dict) -> xr.Dataset:
    """
    Update dataset variables/attributes using the 'header' section from the
    instrument/info YAML.

    Rules:
    - If a key in header matches a dataset VARIABLE → overwrite that variable.
    - If a key in header matches a dataset ATTR → overwrite that attr.

    This allows keeping the NetCDF fairly generic and then injecting
    system-specific metadata later.
    """
    for key_ in header.keys():
        if key_ in dataset.variables:
            dataset[key_] = xr.DataArray(
                header[key_], dims=dataset[key_].dims, coords=dataset[key_].coords
            )
        if key_ in dataset.attrs:
            dataset.attrs[key_] = header[key_]
    return dataset


def drop_unwanted_channels(
    dataset: xr.Dataset, keep_channels: list[str] | None
) -> xr.Dataset:
    """
    Keep only the channels listed in `keep_channels` and drop:
    - their associated signal variables: `signal_{ch}`
    - their associated uncertainties:  `stderr_{ch}` if present
    - and finally select those channels in the coordinate.

    This is useful to reduce the dataset size *before* doing corrections.
    """
    if keep_channels is None:
        return dataset

    # channels to remove from coordinate
    remove_channels = [
        ch for ch in dataset.channel.values if ch not in keep_channels
    ]

    # drop their signals
    remove_signals = [
        f"signal_{ch}" for ch in remove_channels if f"signal_{ch}" in dataset
    ]
    dataset = dataset.drop_vars(remove_signals, errors="ignore")

    # drop their stderr
    remove_stderr = [
        f"stderr_{ch}" for ch in remove_channels if f"stderr_{ch}" in dataset
    ]
    dataset = dataset.drop_vars(remove_stderr, errors="ignore")

    # select only the ones we want
    dataset = dataset.sel(channel=keep_channels)
    return dataset


def add_height(dataset: xr.Dataset) -> xr.Dataset:
    """
    Add a 'height' coordinate computed from 'range' and a constant zenith angle.

    Requirements:
    - dataset must contain `zenithal_angle`
    - angle must be constant in time (e.g. vertical lidar)
    - if zenithal_angle == 0 → height = range
    """
    if "zenithal_angle" in dataset:
        zenithal_angles = np.asarray(dataset["zenithal_angle"].values)
        if not np.allclose(zenithal_angles, zenithal_angles.flat[0], equal_nan=False):
            raise RuntimeError(
                "Zenithal angle is not constant in time. Cannot add height coordinate."
            )
        zenithal_angle = np.deg2rad(zenithal_angles.flat[0])
        if zenithal_angle != 0:
            dataset["height"] = dataset["range"] * np.cos(zenithal_angle)
        else:
            dataset["height"] = dataset["range"].copy()
    return dataset


def apply_smooth(
    dataset: xr.Dataset,
    smooth_mode: str | list[str] | None = None,
    debug: bool = False,
    debug_max_range: float = 2000.0,
    **kwargs,
) -> xr.Dataset:
    """
    Apply one or multiple smoothing methods to variables named 'signal_{channel}'.

    Notes on smoothing:
    - Smoothing is intentionally placed at the **end** of the pipeline to avoid
      blurring instrumental signatures that the previous steps need to detect.
    - Each mode can have specific parameters, passed as:
        apply_smooth(..., smooth_mode="gaussian", gaussian={"sigma": 15.0})
      or
        apply_smooth(..., smooth_mode=["gaussian", "sliding"], gaussian={...}, sliding={...})

    Supported modes
    ---------------
    - "gaussian"        : 1D Gaussian convolution along 'range'
    - "moving"          : simple moving average
    - "sliding"         : sliding window with min/max ranges
    - "sliding_ATLAS"   : ATLAS-like 1D/2D smoothing
    - "adaptive_sliding": range-adaptive window based on SNR

    Axis choice for Gaussian
    ------------------------
    We explicitly find the index of the 'range' dimension in the DataArray
    and call `gaussian_filter1d(..., axis=that_index)`. This makes the code
    robust to dimension ordering like (time, range) or (range, time).
    """
    if smooth_mode is None:
        return dataset

    # normalize to list
    modes = [smooth_mode] if isinstance(smooth_mode, str) else smooth_mode

    for mode in modes:
        # parameters for this mode come in kwargs under that name
        params = kwargs.get(mode, {})

        if mode == "gaussian":
            logger.info("Applying 1D Gaussian smoothing along 'range'...")
            sigma = params.get("sigma", 15.0)
            for ch in dataset.channel.values:
                da = dataset[f"signal_{ch}"]
                axis_index = list(da.dims).index("range")
                smoothed = gaussian_filter1d(
                    da.values,
                    sigma=sigma,
                    axis=axis_index,
                )
                # write back to dataset
                dataset[f"signal_{ch}"].loc[:] = smoothed

            if debug:
                _debug_dump_signals(dataset, "smooth-gaussian", None, max_range=debug_max_range)

        elif mode == "moving":
            logger.info("Applying moving average smoothing...")
            window_sizes = params.get("window_range", 100.0)
            for ch in dataset.channel.values:
                da = dataset[f"signal_{ch}"]
                smoothed = moving_average(
                    data=da.values,
                    window_sizes=window_sizes,
                )
                dataset[f"signal_{ch}"].loc[:] = smoothed

            if debug:
                _debug_dump_signals(dataset, "smooth-moving", None, max_range=debug_max_range)

        elif mode == "sliding":
            logger.info("Applying sliding average smoothing...")
            for ch in dataset.channel.values:
                dataset[f"signal_{ch}"] = sliding_average(
                    dataset[f"signal_{ch}"],
                    maximum_range=params.get("sliding_maximum_range", 4000.0),
                    window_range=params.get("window_range", (50, 100)),
                )
            if debug:
                _debug_dump_signals(dataset, "smooth-sliding", None, max_range=debug_max_range)

        elif mode == "sliding_ATLAS":
            logger.info("Applying ATLAS sliding average smoothing (1D/2D)...")
            smoothing_window = params.get("smoothing_window", (50.0, 100.0))
            if isinstance(smoothing_window, (int, float)):
                smoothing_window = (smoothing_window, smoothing_window)
            err_type = params.get("err_type", "std")
            dataset = smoothing_ATLAS(
                dataset,
                smoothing_window=smoothing_window,
                err_type=err_type,
            )
            if debug:
                _debug_dump_signals(dataset, "smooth-ATLAS", None, max_range=debug_max_range)

        elif mode == "adaptive_sliding":
            logger.info("Applying adaptive sliding average smoothing based on SNR...")
            snr_target = params.get("snr_target", 5.0)
            L_min = params.get("L_min", 50)
            L_max = params.get("L_max", 100)
            for ch in dataset.channel.values:
                sig = dataset[f"signal_{ch}"]
                snr_name = f"snr_{ch}"
                if snr_name in dataset:
                    snr_da = dataset[snr_name]
                else:
                    logger.warning(f"SNR for {ch} not found. Estimating it from the signal...")
                    snr_da = estimate_snr(sig)
                    dataset[snr_name] = snr_da
                smoothed = adaptive_sliding_average(
                    sig,
                    snr=snr_da,
                    snr_target=snr_target,
                    L_min=L_min,
                    L_max=L_max,
                    dim="range",
                )
                dataset[f"signal_{ch}"] = smoothed
            if debug:
                _debug_dump_signals(dataset, "smooth-adaptive", None, max_range=debug_max_range)

        else:
            raise ValueError(
                "smooth_mode not recognized. Use: "
                "gaussian, moving, sliding, sliding_ATLAS, adaptive_sliding."
            )

    return dataset


def apply_dark_current_correction(
    dataset: xr.Dataset,
    rs_path: Path,
    save_dc: bool = False,
    force_dc_in_session: bool = False,
    **kwargs,
) -> xr.Dataset:
    """
    Apply dark-current correction to *dataset*.

    Workflow
    --------
    1. Get the **rebin factor** from the main dataset (if we rebinned earlier).
    2. Split the measurement into **time-continuous groups** (calibration helper).
    3. For each group:
       - Find the corresponding DC file on disk (same session or closest).
       - Keep only the **analog channels** (dc applies to analog).
       - If main dataset was rebinned → rebin DC with the *same* factor.
       - Compute mean DC profile over time.
       - Subtract it from all times in the group.

    Important
    ---------
    - We test DC stability: if the DC std in [0, 5000 m] is too large,
      we warn the user.
    - If `save_dc=True`, we store the DC profile in the main dataset.
    """
    # get rebin factor from the main dataset
    rebin_factor = int(dataset.attrs.get("rebin_factor", 1))

    # split measurement in time-continuous groups
    groups = calibration.split_continous_measurements(dataset.time.values)

    # only analog channels have DC
    channels = dataset.channel.values
    analog_channels: list[str] = [c for c in channels if c.endswith("a")]

    for group in groups:
        # find DC file for this period
        dc_path = search_dc(
            rs_path,
            session_period=group[[0, -1]],
            force_dc_in_session=force_dc_in_session,
        )
        dc = xr.open_dataset(dc_path)

        # keep only analog channels in DC
        dc = drop_unwanted_channels(dc, keep_channels=analog_channels)

        # if main dataset was rebinned, rebin DC too
        if rebin_factor > 1:
            logger.info(f"Rebinning dark-current dataset by factor={rebin_factor}")
            dc = bin_rescale(dc, rebin_factor)

        # index of this time group in main dataset
        lower_idx = np.where(dataset.time == group[0])[0][0]
        upper_idx = np.where(dataset.time == group[-1])[0][0] + 1

        for channel in analog_channels:
            signal_str = f"signal_{channel}"

            # quality check: DC should be fairly flat in range and time
            if dc[signal_str].sel(range=slice(0, 5000)).std("range").std("time") > 0.07:
                raise Warning(
                    f"Dark current std too high for channel {channel} in {dc_path}. "
                    f"Check DC measurement."
                )

            # mean dark current profile (time-averaged)
            dc_mean = dc[signal_str].mean(axis=0)  # (range,)

            # subtract to all times in group
            dataset[signal_str].loc[dict(time=group)] -= dc_mean.values[np.newaxis, :]

            # optionally store DC
            if save_dc:
                if f"dc_{channel}" not in dataset and lower_idx == 0:
                    dataset[f"dc_{channel}"] = dataset[signal_str] * np.nan
                dataset[f"dc_{channel}"][lower_idx:upper_idx] = dc_mean

    dataset.attrs["dc_corrected"] = str(True)
    return dataset


def apply_dead_time_correction(dataset: xr.Dataset) -> xr.Dataset:
    """
    Correct photon-counting channels for detector dead-time.

    Requirements
    ------------
    - The INFO YAML must define `dead_time_ns` per channel.
    - We apply the standard non-paralyzable correction:
          N_true = N_measured / (1 - N_measured * tau)
      with tau in microseconds.

    Only channels whose name ends with `p` are treated as photon-counting.
    """
    lidar_name = dataset.attrs["system"].lower()
    try:
        dt_dict = {
            key: value["dead_time_ns"]
            for (key, value) in INFO["channels"].items()
            if value.get("dead_time_ns", False)
        }
    except Exception:
        raise ValueError(f"No dead time value defined in INFO->{lidar_name}].")

    pc_channels: list[str] = [c for c in dataset.channel.values if c.endswith("p")]

    for ch in pc_channels:
        try:
            tau_us = dt_dict[ch] * 1e-3
        except KeyError:
            raise ValueError(f"No dead time value defined in INFO->{lidar_name}->{ch}.")
        signal_str = f"signal_{ch}"
        dataset[signal_str] = dataset[signal_str] / (1 - dataset[signal_str] * tau_us)
        dataset.attrs["dt_corrected"] = str(True)

    return dataset


def apply_background_correction(
    dataset: xr.Dataset,
    background_ranges: tuple[float, float] | None = None,
    save_bg: bool = False,
) -> xr.Dataset:
    """
    Subtract background (sky + dark residuals) using a far-range interval.

    If `background_ranges` is not provided, the interval is taken from the
    dataset attributes: (BCK_MIN_ALT, BCK_MAX_ALT).

    For every channel:
        background = mean(signal_{ch}(range in interval)) over range
        signal_{ch} -= background

    Optionally, the background time-series is stored as `bg_{ch}`.
    """
    if background_ranges is None:
        background_ranges = (
            dataset.attrs["BCK_MIN_ALT"],
            dataset.attrs["BCK_MAX_ALT"],
        )

    if background_ranges[1] <= background_ranges[0]:
        raise ValueError("background_ranges should be in order (min, max)")

    ranges_between = (background_ranges[0] < dataset.range) & (
        dataset.range < background_ranges[1]
    )
    for ch in dataset.channel.values:
        signal_str = f"signal_{ch}"
        background = dataset[signal_str].loc[:, ranges_between].mean(axis=1)
        dataset[signal_str] -= background
        if save_bg:
            dataset[f"bg_{ch}"] = background

    dataset.attrs["bg_corrected"] = str(True)
    return dataset


def apply_bin_zero_correction(dataset: xr.Dataset, rs_path: Path) -> xr.Dataset:
    """
    Shift signals in range to correct for bin-zero (channel-dependent delay).

    The INFO YAML must define `bin_zero` per channel, in native bins
    (e.g. 3.75 m). If the dataset was rebinned, we scale the bin_zero
    accordingly and shift by that many *rebinned* bins.
    """
    bz_dict = None
    if bz_dict is None:
        lidar_name = dataset.attrs["system"].lower()
        try:
            bz_dict = {
                key: value["bin_zero"] for (key, value) in INFO["channels"].items()
            }
        except Exception:
            raise ValueError(f"No bin zero value defined in INFO->{lidar_name}.")

    rebin_factor = int(dataset.attrs.get("rebin_factor", 1))

    for ch in dataset.channel.values:
        signal_str = f"signal_{ch}"
        bz_native = bz_dict[ch]  # in native 3.75 m bins
        # effective shift in *rebinned* bins
        bz_eff = int(np.floor(bz_native / rebin_factor))
        dataset[signal_str] = dataset[signal_str].shift(range=-bz_eff, fill_value=0.0)

    dataset.attrs["bz_corrected"] = str(True)
    return dataset


def apply_overlap_correction(
    dataset: xr.Dataset,
    ff_overlap_path: Path | str | None = None,
    nf_overlap_path: Path | str | None = None,
) -> xr.Dataset:
    """
    Correct signals for geometric overlap.

    Two options:
    1) If a 2D / per-channel overlap file is provided (ff_overlap_path),
       we read it and divide the signals by the corresponding overlap profile.
    2) Otherwise, we derive the near-field overlap from two channels
       using `ff_2D_overlap_from_channels(...)`.

    The INFO YAML must define:
    - overlap_channels
    - overlap_linked_channels

    For each linked channel we apply:
        signal_ch /= overlap_profile
    and we store `overlap_{ch}` in the dataset.
    """
    def _apply(dataset, overlap, channels2correct, overlap_channel):
        for ch in channels2correct:
            if ch in dataset.channel.values:
                sig = f"signal_{ch}"
                dataset[sig] = dataset[sig] / overlap
                dataset[sig].attrs["overlap_applied"] = overlap_channel
                dataset["overlap_corrected"][
                    dataset.channel.values == ch
                ] = 1
        return dataset

    lidar_name = dataset.attrs["system"].lower()
    overlap_channels = INFO["overlap_channels"]
    linked_channels = INFO["overlap_linked_channels"]

    overlap_corrected = xr.DataArray(
        np.zeros(dataset.channel.size),
        dims=("channel",),
        coords={"channel": dataset.channel},
    )
    dataset["overlap_corrected"] = overlap_corrected

    if ff_overlap_path is not None:
        with xr.open_dataarray(ff_overlap_path) as overlap_data:
            overlap = overlap_data.load()
    else:
        overlap = None

    for ch in linked_channels.keys():
        if ch not in dataset.channel.values:
            continue
        if overlap is not None:
            if ch in overlap.channel.values:
                overlap_ = overlap.sel(channel=ch)
            else:
                raise ValueError(f"Channel {ch} not found in overlap file")
        else:
            nf_ch = overlap_channels[ch]
            if nf_ch not in dataset.channel.values:
                raise ValueError(
                    f"Channel {nf_ch} not found for overlap derivation."
                )
            overlap_ = ff_2D_overlap_from_channels(
                dataset, channel_ff=ch, channel_nf=nf_ch
            )
        # make sure overlap and signal have the same range grid
        if dataset.range.size != overlap_.range.size:
            overlap_ = overlap_.interp(range=dataset.range, method="linear")

        dataset = _apply(dataset, overlap_, linked_channels[ch], overlap_channel=ch)
        dataset[f"overlap_{ch}"] = overlap_

    # near-field overlap could be added here
    return dataset


def apply_crop_ranges_correction(
    dataset: xr.Dataset, crop_ranges: tuple[float, float] | None = (0, 20000)
) -> xr.Dataset:
    """
    Crop the dataset to a given [min, max] range (in meters).

    Must be done AFTER all range-dependent corrections, or you risk
    cutting the part of the profile you need to compute them.
    """
    if crop_ranges is None:
        return dataset
    if crop_ranges[0] > crop_ranges[1]:
        raise ValueError("crop_ranges should be in order (min, max)")
    dataset = dataset.sel(range=slice(*crop_ranges))
    return dataset


def apply_detection_mode_merge(dataset: xr.Dataset) -> xr.Dataset:
    """
    Merge / glue analog (a) and photon-counting (p) detection modes
    into a single 'g' signal per wavelength.

    For every pc channel (*p):
      - find its analog counterpart (*a)
      - apply the gluing function
      - build a small dataset with the glued signal and meta
      - merge it back into the main dataset
    """
    channels_pc: list[str] = [
        c
        for c in dataset.channel.values
        if any(c.startswith(vc) and c.endswith("p") for vc in dataset.channel.values)
    ]

    range_m = dataset["range"].values
    glued_list: list[dict] = []

    for channel_pc in channels_pc:
        channel_an = f"{channel_pc[0:-1]}a"
        if f"signal_{channel_an}" not in dataset:
            continue
        signal_gl = gluing(
            dataset[f"signal_{channel_an}"],
            dataset[f"signal_{channel_pc}"],
        )

        polarization = channel_pc[-1] if channel_pc[-1] in ["p", "s"] else ""
        telescope = channel_pc[0]
        wavelength = channel_pc[1:-1]

        glued_list.append(
            {
                "name": f"{channel_pc[0:-1]}g",
                "signal": signal_gl,
                "polarization": polarization,
                "telescope": telescope,
                "wavelength": wavelength,
            }
        )

    glued_signal = {
        f"signal_{glued['name']}": (["time", "range"], glued["signal"])
        for glued in glued_list
    }

    other_var = {
        "wavelength": (["channel"], [g["wavelength"] for g in glued_list]),
        "polarization": (["channel"], [g["polarization"] for g in glued_list]),
        "telescope": (["channel"], [g["telescope"] for g in glued_list]),
        "bin_shift": (["channel"], [0 for _ in glued_list]),
    }

    glued_dataset = xr.Dataset(
        glued_signal | other_var,
        coords={
            "range": range_m,
            "time": dataset["time"].values,
            "channel": [g["name"] for g in glued_list],
        },
    )

    return xr.merge([dataset, glued_dataset], join="outer")
