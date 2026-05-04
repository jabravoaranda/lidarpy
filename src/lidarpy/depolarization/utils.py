from pathlib import Path

import numpy as np
import xarray as xr
from datetime import datetime

from lidarpy.utils import file_manager

from lidarpy.general_utils.io import find_nearest_filepath


def search_nereast_calib(
    depoCalib: dict, channel: str, current_date: np.datetime64
) -> dict:
    """It searches the nearest calibration to `current_date`. 

    Args:
        depoCalib (dict): Dictionary with depolarization calibrations.
        channel (str): Target channel.
        current_date (np.datetime64): Datetime.

    Returns:
        dict: Dictionary with depolarization calibration at target datetime. 
    """
    wavelength_, telescope_, *_ = file_manager.channel2info(channel)

    # Search the last calibration performed the current measurement
    idx = depoCalib[telescope_]["%0d" % wavelength_].index.get_indexer(
        [current_date], method="pad"
    )  # 'pad': search the nearest lower; 'nearest': search the absolute nearest.

    calib = depoCalib[telescope_]["%0d" % wavelength_].iloc[idx].to_dict("records")[0]

    return calib


def search_nearest_eta_star_from_file(
    channel: str, target_date: datetime, calib_dir: Path | str | None = None
) -> tuple[float, float]:
    """It searches the nearest calibration factor file to a given date (`target_date`).

    Args:
        channel (str): target channel.
        target_date (datetime): target datetime.
        calib_dir (Path | str | None, optional): Directory of calibrations. Defaults to None.

    Raises:
        NotADirectoryError: `calib_dir` is not a directory.

    Returns:
        tuple[float, float]: depolarization calibration average and standard deviation.
    """    
    if calib_dir is None:
        calib_dir = Path.cwd()
    elif isinstance(calib_dir, str):
        calib_dir = Path(calib_dir)

    if not calib_dir.exists() or not calib_dir.is_dir():
        raise NotADirectoryError(f"{calib_dir} not found.")

    calib_path = find_nearest_filepath(calib_dir, "*eta-star*", 2, target_date, and_previous=True)

    calib = xr.open_dataset(calib_path)

    eta_star_mean = f"eta_star_mean_{channel}"
    et_star = calib[eta_star_mean].values.item()
    std_et_star = calib[f"std_{eta_star_mean}"].values.item()

    return et_star, std_et_star
