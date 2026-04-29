from __future__ import annotations

import os
import re
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import overload, Any

import xarray as xr
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid as cumtrapz
from .types import LidarInfoType, Telescope

from lidarpy.general_utils.io import read_yaml

""" MODULE For General Lidar Utilities
"""

# LIDAR SYSTEM INFO
INFO_FILE = Path(__file__).parent.parent.absolute() / "info" / "info_lidars.yml"
LIDAR_INFO: LidarInfoType = read_yaml(INFO_FILE)

INFO_PLOT_FILE = Path(__file__).parent.parent.absolute() / "plot" / "info.yml"
LIDAR_PLOT_INFO = read_yaml(INFO_PLOT_FILE)


@overload
def signal_to_rcs(signal: xr.DataArray, ranges: xr.DataArray) -> xr.DataArray:
    ...


@overload
def signal_to_rcs(
    signal: np.ndarray[Any, np.dtype[np.float64]],
    ranges: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ...


def signal_to_rcs(signal, ranges) -> xr.DataArray | np.ndarray[Any, np.dtype[np.float64]]:
    """Convert Lidar Signal to range-corrected signal

    Args:
        signal (np.ndarray[Any, np.dtype[np.float64]] | xr.DataArray): Lidar signal
        ranges (np.ndarray[Any, np.dtype[np.float64]] | xr.DataArray): Lidar ranges of signal

    Returns:
         xr.DataArray | np.ndarray[Any, np.dtype[np.float64]]: Range-corrected signal

    """
    return signal * ranges**2


@overload
def rcs_to_signal(rcs: xr.DataArray, ranges: xr.DataArray) -> xr.DataArray:
    ...


@overload
def rcs_to_signal(rcs: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    ...


def rcs_to_signal(rcs, ranges):
    return rcs / ranges**2


def smooth_signal(signal, method="savgol", savgol_kwargs: dict | None = None):
    """Smooth Lidar Signal

    Args:
        signal ([type]): [description]
        method (str, optional): [description]. Defaults to 'savgol'.
    """

    if method == "savgol":
        if savgol_kwargs is None:
            savgol_kwargs = {"window_length": 21, "polyorder": 2}
        smoothed_signal = savgol_filter(signal, **savgol_kwargs)
    else:
        raise NotImplementedError(f"{method} has not been implemented yet")

    return smoothed_signal


def estimate_snr(signal, window=5):
    """[summary]

    Args:
        signal ([type]): [description]
    """

    # ventana: numero impar
    if window % 2 == 0:
        window += 1
    subw = window // 2

    n = len(signal)
    avg = np.zeros(n) * np.nan
    std = np.zeros(n) * np.nan

    for i in range(n):
        ind_delta_min = i - subw if i - subw >= 0 else 0
        ind_delta_max = i + subw if i + subw < n else n - 1

        si = signal[ind_delta_min : (ind_delta_max + 1)]
        avg[i] = np.nanmean(si)
        std[i] = np.nanstd(si)

        # print("%i, %i, %i" % (i, ind_delta_min, ind_delta_max + 1))
        # print(signal[ind_delta_min:(ind_delta_max+1)])
    snr = avg / std

    return snr, avg, std


def get_lidar_name_from_filename(fn):
    """Get Lidar System Name from L1a File Name
    Args:
        fn (function): [description]
    """
    lidar_nick = os.path.basename(fn).split("_")[0]
    if lidar_nick in LIDAR_INFO["metadata"]["nick2name"].keys():
        lidar = lidar_nick
    else:
        lidar = None
    return lidar


def to_licel_date_str(date: datetime) -> str:
    # Convert month number into hex capital letter
    month_hex = f"{date.month:x}".upper()
    return f'{date.strftime(r"%y")}{month_hex}{date.strftime(r"%d")}'


def licel_to_datetime(licel_name: str) -> datetime:
    """Convert Licel Date String to datetime
    Args:
        licel_name (str): Licel Date String
    """
    try:
        name, extension = licel_name.split(".")
    except ValueError:
        raise ValueError(f"File name {licel_name} does not have an extension")
    # Convert hexadecimal 'm' to decimal
    month_decimal = int(name[-5], 16)
    return datetime.strptime(
        f"{name[-7:-5]}{month_decimal:02d}{name[-4:-2]}T{name[-2:]}{extension[:4]}",
        r"%y%m%dT%H%M%S",
    )


def filter_wildcard(
    directory: Path | TemporaryDirectory, pattern_or_list: str | list[str] = r'\.\d+$', recursive_search=True
) -> list[Path]:
    """Filter files by wildcard

    Args:
        directory (Path): Directory to search.
        wildcard (str, optional): Wildcard. Defaults to r'\.\d+$'.
        recursive_search (bool, optional): Recursive search. Defaults to True means recursive search includes subdirectories.

    Raises:
        Exception: Directory does not exist.
        Exception: Directory is not a directory.

    Returns:
        list[Path]: List of files.
    """
    if isinstance(directory, TemporaryDirectory):
        directory = Path(directory.name)
        
    if not directory.exists():
        raise Exception(f"Directory {directory} does not exist")
    if not directory.is_dir():
        raise Exception(f"{directory} is not a directory")
    
    if isinstance(pattern_or_list, str):
        if recursive_search:            
            return [file for file in directory.rglob("*") if re.search(pattern_or_list, file.absolute().as_posix())]
        else:
            return [file for file in directory.glob("*") if re.search(pattern_or_list, file.absolute().as_posix())]
    else: 
        if recursive_search:
            return [file for file in directory.rglob("*") if file.name in pattern_or_list]
        else:
            return [file for file in directory.glob("*") if file.name in pattern_or_list]


def is_within_datetime_slice(filename: str, datetime_slice: slice) -> str:
    return datetime_slice.start <= licel_to_datetime(filename) < datetime_slice.stop

def get_532_from_telescope(telescope: Telescope = Telescope.xf) -> str:
    if telescope == Telescope.xf:
        return "532xpa"
    elif telescope == Telescope.ff:
        return "532fpa"
    elif telescope == Telescope.nf:
        return "532npa"

    raise ValueError("Telescope type not recognized. Options are xf, ff, nf")


def sigmoid(x, x0, k, coeff: float = 1, offset: float = 0):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return (coeff * y) + offset


def extrapolate_beta_with_angstrom(
    beta_ref: np.ndarray,
    wavelength_ref: float,
    wavelength_target: float,
    angstrom_exponent: float | np.ndarray,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return beta_ref * (wavelength_target / wavelength_ref) ** -angstrom_exponent


def integrate_from_reference(integrand, x, reference_index):
    """

    at x[ref_index], the integral equals = 0
    """
    # integrate above reference
    int_above_ref = cumtrapz(integrand[reference_index:], x=x[reference_index:])

    # integrate below reference
    int_below_ref = cumtrapz(
        integrand[: reference_index + 1][::-1], x=x[: reference_index + 1][::-1]
    )[::-1]

    return np.concatenate([int_below_ref, np.zeros(1), int_above_ref])


def optical_depth(extinction, height, ref_index=0):
    """
    Integrate extinction profile along height: r'$\tau(z) = \int_0^z d\dseta \alpha(\dseta)$'
    """

    return integrate_from_reference(extinction, height, reference_index=ref_index)


def refill_overlap(
    atmospheric_profile: np.ndarray[Any, np.dtype[np.float64]],
    height: np.ndarray[Any, np.dtype[np.float64]] | xr.DataArray,
    fulloverlap_height: float = 600,
    fill_with: float | None = None,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Fill overlap region [0-`fulloverlap_height`] of the profile `atmospheric_profile` with the value `fill_with` provided by the user. If None, fill with the value at `fulloverlap_height`.

    Args:
        atmospheric_profile (np.ndarray): Atmospheric profile
        height (np.ndarray): Range profile in meters
        fulloverlap_height (float, optional): Fulloverlap height in meters. Defaults to 600 m.
        fill_with (float, optional): Value to fill the overlap region. Defaults to None.

    Returns:
        np.ndarray: Profile `atmospheric_profile` with the overlap region filled.
    """
    if isinstance(atmospheric_profile, xr.DataArray):
        atmospheric_profile = atmospheric_profile.values

    if fulloverlap_height < height[0] or fulloverlap_height > height[-1]:
        raise ValueError(
            "The fulloverlap_height is outside the range of height values."
        )

    idx_overlap = np.abs(height - fulloverlap_height).argmin()

    if fill_with is None:
        fill_with = atmospheric_profile[idx_overlap]

    new_profile = np.copy(atmospheric_profile)
    new_profile[:idx_overlap] = fill_with

    return new_profile
