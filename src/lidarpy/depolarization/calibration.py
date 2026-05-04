from pathlib import Path
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
import xarray as xr

from lidarpy.utils import file_manager
from lidarpy.preprocessing.lidar_preprocessing import preprocess
from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.general_utils.io import read_yaml_from_info


def retrieve_eta_star_profile(
    signal_T_plus45: xr.DataArray,
    signal_T_minus45: xr.DataArray,
    signal_R_plus45: xr.DataArray,
    signal_R_minus45: xr.DataArray,
    transmittance_extra_filter: float,
) -> xr.DataArray:
    """Calculate the calibration constant [eta_star] in a lidar system that is able to
    detect the T-to-R depolarization ratio.

    Args:
        signal_T_plus45 (xr.DataArray): The input vertical profile from the T channel. Calibrator angle phi=45.
        signal_T_minus45 (xr.DataArray): The input vertical profile from the R channel. Calibrator angle phi=45.
        signal_R_plus45 (xr.DataArray): The input vertical profile from the T channel. Calibrator angle phi=-45.
        signal_R_minus45 (xr.DataArray): The input vertical profile from the R channel. Calibrator angle phi=-45.
        kwargs (dict): transmittance_extra_filter [float] is the transmittance of the extra filter during depol. calibration.

    Returns:
        xr.DataArray: eta star profile.

    Notes:
        The calibration constant is calculated by the following formula:
    .. math::
        \\eta^* = \\left( \\frac{signal(T,+45)}{signal(R,+45)}\\frac{signal(T,-45)}{signal(R,-45)} \\right)^{0.5}

    References:
    Freudenthaler, V. 2016. https://doi.org/10.5194/amt-9-4181-2016
    """

    # Remove effect of the possible extra filter to avoid saturation of the s-channel.
    signal_T_plus45_with_extra_filter = signal_T_plus45
    signal_T_plus45 = signal_T_plus45_with_extra_filter / transmittance_extra_filter
    signal_T_minus45_with_extra_filter = signal_T_minus45
    signal_T_minus45 = signal_T_minus45_with_extra_filter / transmittance_extra_filter

    #   Calculate the signal ratio for the +45 position.
    delta_v45_plus = signal_R_plus45 / signal_T_plus45

    #   Calculate the signal ratio for the -45 position.
    delta_v45_minus = signal_R_minus45 / signal_T_minus45

    #   Calculate the calibration constant vertical profile.
    v_star_profile = (delta_v45_plus * delta_v45_minus) ** 0.5
    return v_star_profile


def eta_star_mean_std(
    eta_star_profile: xr.DataArray, average_range: tuple[float, float]
) -> tuple[float, float]:
    """Calculate the mean calibration constant and its standard error of the mean, from the calibration constant profile.

    Args:
    c_profile: vector
       The vertical profile of the calibration constant.
    kwargs (dict): min_range [float] is the lower vertical limit for the calculation in meters. max_range [float] is the upper vertical limit for the calculation in meters.

    Returns:
    c_mean: float
       Calibration constant's mean value (vertical axis).
    c_sem: float
       Calibration constant's standard error of the mean (vertical axis).
    """

    if "time" in eta_star_profile.dims:
        raise ValueError("eta_star_profile should have only `range` dimension.")

    #   Select the area of interest.
    eta_star_mean = (
        eta_star_profile.sel(range=slice(*average_range))
        .mean("range")
        .values.item()
    )
    eta_star_std = (
        eta_star_profile.sel(range=slice(*average_range))
        .std("range")
        .values.item()
    )

    #   Return the statistics.
    return eta_star_mean, eta_star_std


def calibration_factor_files(
    P45_fn: Path,
    N45_fn: Path,
    calib_dir: Path | str,    
    epsilon: float | None = None,
    an_calib_limits: tuple[float, float] = (750, 1250),
    pc_calib_limits: tuple[float, float] = (750, 1250),
    channel_T_transmittance: dict[str, float] | None = None,
) -> xr.Dataset:
    """It retrieves the calibration factor from two files (+45 and -45 degrees).
   
    Args:

        - P45_fn (Path): netcdf +45-degree filepath.
        - N45_fn (Path): netcdf +45-degree filepath.
        - calib_dir (Path | str): Directory to save the calibration file.
        - channel_T_transmittance dict[str, float] | None. Transmittance of the extra filter for each channel. Defaults means values taken from lidar info.
        - epsilon (float | None, optional): Misalignment angle of the calibrator. Defaults to None means zero.
        - an_calib_limits (tuple[float, float], optional): Mininum amd maximum range to performe the analaog calibration average. Defaults to (1500, 3000).
        - pc_calib_limits (tuple[float, float], optional): Mininum amd maximum range to performe the photoncounting(gluing) calibration average. Defaults to (2500, 4000).

    Raises:
        - FileNotFoundError: +45-degrees file not found.
        - FileNotFoundError: -45-degrees file not found.
        - FileNotFoundError: Calibration directory not found.
        - ValueError: detection mode not recognized.

    Returns:
        - xr.Dataset: calibration factor dataset.
    """    
    if not P45_fn.is_file():
        raise FileNotFoundError(f"{P45_fn} not found.")

    if not N45_fn.is_file():
        raise FileNotFoundError(f"{N45_fn} not found.")

    if isinstance(calib_dir, str):
        calib_dir = Path(calib_dir)

    if not calib_dir.is_dir():
        raise FileNotFoundError(f"{calib_dir} not found.")

    #Get correct lidar info using the filename    
    lidar_nick, _, _, _, _, calib_date  = file_manager.filename2info(P45_fn.name)

    global INFO 
    INFO = read_yaml_from_info(lidar_nick, calib_date)

    lidar_name: str = LIDAR_INFO["metadata"]["nick2name"][lidar_nick]

    P45 = preprocess(P45_fn)
    N45 = preprocess(N45_fn)

    calib_dict = {}
    for telescope_ in INFO[
        "polarized_channels"
    ].keys():

        calib_dict[telescope_] = {}

        for wavelength_ in INFO["polarized_channels"][
            telescope_
        ].keys():

            calib_dict[telescope_][wavelength_] = {}

            for mode_ in INFO["polarized_channels"][
                telescope_
            ][wavelength_].keys():
                if mode_ == 'a':
                    average_range = an_calib_limits
                else:
                    average_range = pc_calib_limits

                # Define the channel names
                channel_T = INFO["polarized_channels"][
                    telescope_
                ][wavelength_][mode_]["T"]

                channel_R = INFO["polarized_channels"][
                    telescope_
                ][wavelength_][mode_]["R"]

                # Check if the channel is available in both P45 and N45.
                if (
                    channel_T not in P45.channel.values
                    or channel_R not in P45.channel.values
                ):
                    continue

                if (
                    channel_T not in N45.channel.values
                    or channel_R not in N45.channel.values
                ):
                    continue

                calib_dict[telescope_][wavelength_][mode_] = {}

                # Apply smoothing filter
                signal_T_P45 = xr.apply_ufunc(
                    savgol_filter,
                    P45[f"signal_{channel_T}"].mean("time"),
                    11,
                    3,
                    dask="allowed",
                )
                signal_R_P45 = xr.apply_ufunc(
                    savgol_filter,
                    P45[f"signal_{channel_R}"].mean("time"),
                    11,
                    3,
                    dask="allowed",
                )
                signal_R_N45 = xr.apply_ufunc(
                    savgol_filter,
                    N45[f"signal_{channel_R}"].mean("time"),
                    11,
                    3,
                    dask="allowed",
                )
                signal_T_N45 = xr.apply_ufunc(
                    savgol_filter,
                    N45[f"signal_{channel_T}"].mean("time"),
                    11,
                    3,
                    dask="allowed",
                )
                
                if channel_T_transmittance is None:
                    transmittance_extra_filter = INFO["depolarization_calibration_transmittance"][channel_T[:-1]]
                else:
                    raise ValueError(f"channel_T_transmittance of channel {channel_T[:-1]} not found in lidar info config file. Dictionary: 'depolarization_calibration_transmittance'.")

                # Calculate the calibration constant vertical profile.
                eta_star_profile = retrieve_eta_star_profile(
                    signal_T_P45,
                    signal_T_N45,
                    signal_R_P45,
                    signal_R_N45,
                    transmittance_extra_filter=transmittance_extra_filter,
                )
                calib_dict[telescope_][wavelength_][mode_]["profile"] = eta_star_profile
                eta_star_mean, eta_star_std = eta_star_mean_std(
                    eta_star_profile, average_range=average_range
                )

                # Calibrator rotation, epsilon [Freudenthaler, V. (2016)., Eq. 194, 195]
                # average over calibration height interval
                gain_ratio_p45 = signal_T_P45 / signal_R_P45
                gain_ratio_n45 = signal_T_N45 / signal_R_N45

                ranges = P45.range.values

                match mode_:
                    case "a":
                        idx_avg_ranges = (ranges >= an_calib_limits[0]) & (
                            ranges <= an_calib_limits[1]
                        )
                    case "p":
                        idx_avg_ranges = (ranges >= pc_calib_limits[0]) & (
                            ranges <= pc_calib_limits[1]
                        )
                    # case "g":
                    #     ...
                    # TODO: Gluing gna be implemented in the future by receiving extra argument
                    case _:
                        raise ValueError(f"Mode {mode_} not recognized")

                Y = (gain_ratio_p45 - gain_ratio_n45) / (
                    gain_ratio_p45 + gain_ratio_n45
                )
                Y_avg = np.nanmean(Y[idx_avg_ranges])
                Y_std = np.nanstd(Y[idx_avg_ranges])
                if epsilon is None:
                    epsilon = (
                        (180 / np.pi) * 0.5 * np.arcsin(np.tan(0.5 * np.arcsin(Y_avg)))
                    )

                epsilon_err = (
                    (180 / np.pi)
                    * 0.5
                    * abs(
                        0.5 * np.arcsin(np.tan(0.5 * np.arcsin(Y_avg + Y_std)))
                        - 0.5 * np.arcsin(np.tan(0.5 * np.arcsin(Y_avg - Y_std)))
                    )
                )

                calib_dict[telescope_][wavelength_][mode_]["values"] = [
                    eta_star_mean,
                    eta_star_std,
                    Y_avg,
                    Y_std,
                    epsilon,
                    epsilon_err,
                ]

                calib_dict[telescope_][wavelength_][mode_]["signals"] = [
                    signal_T_N45,
                    signal_T_P45,
                    signal_R_P45,
                    signal_R_N45,
                    gain_ratio_p45,
                    gain_ratio_n45,
                ]

    # Create dataset
    calib_dataset = xr.Dataset()

    channels = []
    for telescope_ in calib_dict.keys():
        for wavelength_ in calib_dict[telescope_].keys():
            for mode_ in calib_dict[telescope_][wavelength_].keys():
                # Add the channel name to a list
                channels.append(f"{wavelength_}{telescope_[0]}{mode_}")

                # Define the key for the profile
                key_profile = f"eta_star_profile_{wavelength_}{telescope_[0]}{mode_}"
                calib_dataset[key_profile] = calib_dict[telescope_][wavelength_][mode_][
                    "profile"
                ]

                # Get the depolarization values
                (
                    eta_star_mean,
                    eta_star_std,
                    Y_avg,
                    Y_std,
                    epsilon,
                    epsilon_err,
                ) = calib_dict[telescope_][wavelength_][mode_]["values"]

                # Define xarray dataarrays
                calib_dataset[
                    f"eta_star_mean_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    eta_star_mean,
                    dims=[],
                    attrs={
                        "long_name": f"range-average of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1],
                    },
                )
                calib_dataset[
                    f"eta_star_standard_deviation_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    eta_star_std,
                    dims=[],
                    attrs={
                        "long_name": f"range standard-deviation of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1] 
                    },
                )

                calib_dataset[
                    f"Y_average_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    Y_avg,
                    dims=[],
                    attrs={
                        "long_name": f"range standard-deviation of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1]
                    },
                )
                calib_dataset[
                    f"Y_standard_deviation_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    Y_std,
                    dims=[],
                    attrs={
                        "long_name": f"range standard-deviation of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1]
                    },
                )

                calib_dataset[
                    f"epsilon_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    epsilon,
                    dims=[],
                    attrs={
                        "long_name": f"range standard-deviation of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1]
                    },
                )

                calib_dataset[
                    f"epsilon_error_{wavelength_}{telescope_[0]}{mode_}"
                ] = xr.DataArray(
                    epsilon_err,
                    dims=[],
                    attrs={
                        "long_name": f"range standard-deviation of {key_profile}",
                        "min_range_m": average_range[0],
                        "max_range_m": average_range[1]
                    },
                )

                signals_order = [
                    "signal_T_N45",
                    "signal_T_P45",
                    "signal_R_P45",
                    "signal_R_N45",
                    "gain_ratio_p45",
                    "gain_ratio_n45",
                ]

                for signal, signal_name in zip(calib_dict[telescope_][wavelength_][mode_]["signals"], signals_order):  # type: ignore
                    calib_dataset[
                        f"{signal_name}_{wavelength_}{telescope_[0]}{mode_}"
                    ] = signal

    # Channels is not a coordinate yet, just a utility to read the file
    calib_dataset["channels"] = channels

    # Add time
    calib_datetime_str = datetime.strftime(calib_date, "%Y%m%d_%H%M")
    calib_dataset.attrs["calibration_datetime"] = calib_datetime_str
    calib_dataset.attrs["system"] = lidar_name

    calib_filename = f"{lidar_nick}_eta-star_{calib_datetime_str}.nc"
    calib_dataset.to_netcdf(calib_dir / calib_filename)
    return calib_dataset



def eta_star_reader(filepath: Path) -> xr.Dataset:
    """It loads the eta_star data from files generated with `lidarpy.depolarization.calibration.calibration_factor_files`.

    Args:

        - filepath (Path): Filepath output from `lidarpy.depolarization.calibration.calibration_factor_files`.

    Raises:

        - FileNotFoundError: File not found.

    Returns:

        - xr.Dataset: eta_star in dataset. 
    """    
    if not filepath.is_file():
        raise FileNotFoundError(f"{filepath} not found.")

    eta_star_dataset = xr.open_dataset(filepath)
    return eta_star_dataset