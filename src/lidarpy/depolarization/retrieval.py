from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
from datetime import datetime

from lidarpy.utils import file_manager
from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.depolarization import io
from lidarpy.general_utils.io import find_nearest_filepath, read_yaml_from_info
from lidarpy.general_utils.dates import parse_datetime

CDIR = Path(__file__).parent.parent.parent.parent


def backscattering_ratio(
    molecular_backscatter: np.ndarray, particle_backscatter: np.ndarray
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Retrieves the backscattering ratio. Inputs must be in the same units.

    Args:
        molecular_backscatter (np.ndarray): Molecular backscatter coefficient.
        particle_backscatter (np.ndarray):  Particle backscatter coefficient.

    Returns:
        np.ndarray: backscattering ratio
    """

    return molecular_backscatter + particle_backscatter / molecular_backscatter


def linear_volume_depolarization_ratio(
    signal_R: np.ndarray,
    signal_T: np.ndarray,
    channel_R: str,
    channel_T: str,
    range: np.ndarray,
    time: np.ndarray,
    eta_star: float = 1,
    K: float = 1,
    GT: float = 1,
    HT: float = -1,
    GR: float = 1,
    HR: float = 1,
) -> xr.DataArray:
    """Calculate the linear volume depolarization ratio.

    Args:
        signal_R (np.ndarray): reflected signal in the polarizing beam splitter cube.
        signal_T (np.ndarray): transmitted signal in the polarizing beam splitter cube.
        channel_R (str): reflected channel in the polarizing beam splitter cube.
        channel_T (str): transmitted channel in the polarizing beam splitter cube.
        range (np.ndarray): range series of the signal.
        time (np.ndarray): time vector of the signal.
        eta_star (float, optional): calibration factor retrieved with Delta90 method. Defaults to 1.
        K (float, optional): K factor value simulated with Volker's algorithm. Defaults to 1.
        GT (float, optional): GT factor value simulated with Volker's algorithm. Defaults to 1.
        HT (float, optional): HT factor value simulated with Volker's algorithm. Defaults to -1.
        GR (float, optional): GR factor value simulated with Volker's algorithm. Defaults to 1.
        HR (float, optional): HR factor value simulated with Volker's algorithm. Defaults to 1.
        Y (float, optional): Optical system orientation (laser-beam Vs beam-splitter cube).

    Raises:
        ValueError: Wavelength, telescope or mode does not fit raises 'Polarized channels to be merged not appropiated'.
        ValueError: polarization codes do not fit raises 'Polarized channels to be merged not appropiated'.

    Returns:
        xr.DataArray: linear_volume_depolarization_ratio
    """

    wavelengthR, telescopeR, polR, modeR = file_manager.channel2info(channel_R)
    wavelengthT, telescopeT, polT, modeT = file_manager.channel2info(channel_T)

    if wavelengthR != wavelengthT or telescopeR != telescopeT or modeR != modeT:
        raise ValueError("Polarized channels to be merged not appropiated.")

    if polT not in ["p", "c", "s"] or polR not in ["p", "c", "s"] and polR != polT:
        raise ValueError("Polarized channels to be merged not appropiated.")

    # Remove zeros from signal_T to avoid division by zero
    signal_T[signal_T == 0] = np.nan
    eta = eta_star / K
    ratio = (signal_R / signal_T) / eta

    # Remove zeros from ratio to avoid division by zero
    ratio[ratio == 0] = np.nan
    lvdr_ = (((GT + HT) * ratio - (GR + HR)) / ((GR - HR) - (GT - HT) * ratio)).astype(
        float
    )
    # # Add new axis to lvdr_
    if lvdr_.ndim == 1:
        lvdr_ = np.expand_dims(lvdr_, axis=0)


    # Create DataArray
    lvdr = xr.DataArray(
        lvdr_,
        coords={"time": time, "range": range},
        dims=["time", "range"],
        attrs={
            "long_name": "Linear Volume Depolarization Ratio",
            "detection_mode": LIDAR_INFO["metadata"]["code_mode_str2number"][modeR],
            "wavelength": wavelengthR,
            "units": "$\\#$",
        },
    )
    return lvdr


def add_depolarization_products_of_channel(
    dataset: xr.Dataset,
    channel_depo: str,
    ghk: Dict[str, Union[float, None]],
    calibration: xr.Dataset,
) -> xr.Dataset:
    """Add total (parallel + cross) signal and the linear volume depolarization ratio for the polarizing channels to the `lidar dataset`.

    Args:
        dataset (xr.Dataset): Lidar dataset from `lidar.preprocess`.
        channel_depo (str): Depolarization channel [e.g., '532x', '355n'].
        ghk (Dict[str, Union[float, None]]): `channel_depo` GHK values from info lidar config (*.yml).
        calibration (xr.Dataset): Depolarization calibration dataset from `lidarpy.depolarization.calibration.eta_star_reader`.

    Raises:
        ValueError: At least one GHK value is None.
        KeyError: `channel_depo` eta_star_mean is not in the calibration dataset.

    Returns:
        xr.Dataset: Given lidar dataset with the depolarization products.
    """

    # Check GHK values are not None
    if "K" in ghk.keys():
        if ghk["K"] is not None:
            K = ghk["K"]
        else:
            raise ValueError("K is None.")
    else:
        raise KeyError("K is not in GHK values.")
    if "GT" in ghk.keys():
        if ghk["GT"] is not None:
            GT = ghk["GT"]
        else:
            raise ValueError("GT is None.")
    else:
        raise KeyError("GT is not in GHK values.")
    if "HT" in ghk.keys():
        if ghk["HT"] is not None:
            HT = ghk["HT"]
        else:
            raise ValueError("HT is None.")
    else:
        raise KeyError("HT is not in GHK values.")
    if "GR" in ghk.keys():
        if ghk["GR"] is not None:
            GR = ghk["GR"]
        else:
            raise ValueError("GR is None.")
    else:
        raise KeyError("GR is not in GHK values.")
    if "HR" in ghk.keys():
        if ghk["HR"] is not None:
            HR = ghk["HR"]
        else:
            raise ValueError("HR is None.")
    else:
        raise KeyError("HR is not in GHK values.")

    # Get lidar name
    lidar_name: str = dataset.attrs["system"].lower()

    #Get lidar nick
    lidar_nick: str = LIDAR_INFO["metadata"]["name2nick"][lidar_name]

    # Get INFO
    INFO = read_yaml_from_info(lidar_nick, dataset.time.values[0])

    # Get channel info
    wavelength_, telescope_ = int(channel_depo[0:-1]), channel_depo[-1]

    # Get polarization channels to be processed
    polarized_channels = INFO["polarized_channels"]

    for mode_ in polarized_channels[f"{telescope_}f"][wavelength_].keys():
        # Data
        channel_R = polarized_channels[f"{telescope_}f"][wavelength_][mode_]["R"]
        channel_T = polarized_channels[f"{telescope_}f"][wavelength_][mode_]["T"]

        if (
            dataset.get(f"signal_{channel_R}") is not None
            and dataset.get(f"signal_{channel_T}") is not None
        ):
            try:
                eta_star = calibration[
                    f"eta_star_mean_{wavelength_}{telescope_}{mode_}"
                ].item()
            except KeyError:
                raise KeyError(
                    f"eta_star_mean_{wavelength_}{telescope_}{mode_} must be a key in the calibration xr.dataset."
                )

            # Sum of parallel and cross polarized channels
            dataset[f"signal_{wavelength_}{telescope_}t{mode_}"] = (
                eta_star * HR * dataset[f"signal_{channel_T}"]
                - HT * dataset[f"signal_{channel_R}"]
            )

            # Retrieve LVDR
            signal_R = dataset[f"signal_{channel_R}"].values
            signal_T = dataset[f"signal_{channel_T}"].values
            ranges = dataset.range.values
            times = dataset.time.values

            lvdr = linear_volume_depolarization_ratio(
                signal_R,
                signal_T,
                channel_R,
                channel_T,
                ranges,
                times,
                eta_star,
                K,
                GT,
                HT,
                GR,
                HR,
            )

            # Save lvdr in dataset
            key_channel = channel_R.replace("pa", "a").replace("pp", "p")
            dataset[f"linear_volume_depolarization_ratio_{key_channel}"] = lvdr
            dataset[f"linear_volume_depolarization_ratio_{key_channel}"].attrs[
                "calibration_datetime"
            ] = calibration.attrs["calibration_datetime"]
    return dataset


def add_depolarization_products(
    dataset: xr.Dataset, depo_calib_dir: Path
) -> xr.Dataset:

    # Check directories exist
    if not depo_calib_dir.exists() and not depo_calib_dir.exists():
        raise FileNotFoundError(f"{depo_calib_dir} does not exist.")

    # Get lidar name
    lidar_name: str = dataset.attrs["system"].lower()
    lidar_nick: str = LIDAR_INFO["metadata"]["name2nick"][lidar_name]

    # Get INFO
    INFO = read_yaml_from_info(lidar_nick, dataset.time.values[0])

    # Get datetime from dataset    
    target_date = parse_datetime(dataset.time.values[0])
    
    # Get polarization channels to be processed
    polarized_channels = INFO["polarized_channels"]

    # Get depolarization calibration file
    depo_calib_file = find_nearest_filepath(
        depo_calib_dir, f"*{lidar_nick}*eta-star*.nc", 2, target_date, and_previous=True
    )

    depo_calib = xr.open_dataset(depo_calib_file)

    for telescope_ in polarized_channels.keys():
        for wavelength_ in polarized_channels[telescope_].keys():
            # Channel depo
            depo_channel = f"{wavelength_}{telescope_[0]}"

            if not depo_channel in [
                channel_.item()[:-2] for channel_ in dataset.channel
            ]:
                continue

            ghk = INFO["GHK"][depo_channel]

            # Add depolarization products
            dataset = add_depolarization_products_of_channel(
                dataset, depo_channel, ghk, depo_calib
            )
    return dataset

def particle_depolarization(
    linear_volume_depolarization_ratio: np.ndarray,
    backscattering_ratio: np.ndarray,
    molecular_depolarization: float,
    time: np.ndarray,
    range: np.ndarray,
) -> xr.DataArray:  # FIXME: this function may realy direction to the lvdr xr.DataArray inside the lidar dataset once executed lvdr_from_dataset.
    """Calculate the linear particle depolarization ratio.

    Args:
        linear_volume_depolarization_ratio (np.ndarray): linear volume depolarization ratio
        backscattering_ratio (np.ndarray): _description_
        molecular_depolarization (float): molecular linear volume depolarization ratio
        time (np.ndarray): time vector of the signal.
        range (np.ndarray): range series of the signal.

    Returns:
        xr.DataArray: linear particle depolarization ratio.

    Notes:
    The linear particle depolarization ratio is calculated by the formula:

    .. math::
       \\delta^p = \\frac{(1 + \\delta^m)\\delta^V \\mathbf{R} - (1 + \\delta^V)\\delta^m}
       {(1 + \\delta^m)\\mathbf{R} - (1 + \\delta^V)}


    References:
    Freudenthaler, V. et al. Depolarization ratio profiling at several wavelengths in pure
    Saharan dust during SAMUM 2006. Tellus, 61B, 165-179 (2008)
    """

    delta_p = (
        (1 + molecular_depolarization)
        * linear_volume_depolarization_ratio
        * backscattering_ratio
        - (1 + linear_volume_depolarization_ratio) * molecular_depolarization
    ) / (
        (1 + molecular_depolarization) * backscattering_ratio
        - (1 + linear_volume_depolarization_ratio)
    )

    #Add new axis to delta_p
    if delta_p.ndim == 1:
        delta_p = np.expand_dims(delta_p, axis=0)
    
    # Create DataArray
    delta_p_xarray = xr.DataArray(
        delta_p,
        coords={"time": time, "range": range},
        dims=["time", "range"],
        attrs={
            "long_name": "Linear Volume Depolarization Ratio",
            "units": "$\\#$",
        },  # TODO: decide where the attributes should be included.
    )
    return delta_p_xarray
