from pathlib import Path


from datetime import datetime, date
import numpy as np

from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.general_utils.io import read_yaml_from_info
from lidarpy.general_utils.dates import parse_datetime


def channel2info(channel: str) -> tuple[int, str, str | None, str]:
    """`channel_str2info` extracts the information concerning the channel configuration from the string code.

    Args:
        channel (str): Chanel string code (e.g., '532xta'.)

    Returns:
        tuple[str]: Tuple with following elements: wavelength [int], telescope Options['xf', 'ff', 'nf'], polarization Options['p', 's', 't'], detection Options['a', 'p', 'g'].
    """

    channel = channel.split("_")[-1]
    if len(channel) > 5:
        wave = int(channel[0:-3], base=10)
        telescope = f"{channel[-3]}f"
        polarization = channel[-2]
        detection = channel[-1]
    elif len(channel) == 5:
        wave = int(channel[0:-2], base=10)
        telescope = f"{channel[-2]}f"
        polarization = None
        detection = channel[-1]
    else:
        raise ValueError("Channel format not recognized.")

    return wave, telescope, polarization, detection


def filename2info(filename: str) -> tuple[str, str, str, str, str, datetime]:
    parts = filename.split(".")[0].split("_")

    lidar_nick = parts[0]
    data_level = parts[1]
    measurement_type = parts[2][1:3]
    signal_type = parts[3]
    telescope = parts[4]
    if measurement_type in ["rs", "hf"]:
        date_ = parts[5]
        hour_ = None
    else:
        date_ = parts[5]
        hour_ = parts[6]
    if hour_ is not None:
        date_dt = datetime.strptime(date_ + hour_, "%Y%m%d%H%M")

    else:  # has not hour
        date_dt = datetime.strptime(date_, "%Y%m%d")
    return lidar_nick, data_level, measurement_type, signal_type, telescope, date_dt


def info2filename(
    date: datetime | date,
    telescope: str = "xf",
    lidar_nick: str | None = None,
    data_level: str = "1a",
    measurement_type: str = "RS",
    signal_type: str = "rs",
    lidar_name: str | None = None,
    subdir: str | None = '',
    add_hour: bool = False,
) -> str:
    """Lidar information to filename.

    Args:
        channel (str): Channel string code (e.g., '532xpa')
        date (datetime.datetime | datetime.date): It must be datetime.datetime if `measurement_type` is `DC`.
        lidar_nick (str | None): Lidar nick (e.g., 'mhc'0')
        measurement_type (str, optional): Measurment lidar type ['RS', 'DP', 'TC', DC', 'OT']. Defaults to 'RS'.
        signal_type (str, optional): Signal type  ['rs', 'sd']. Defaults to 'rs'.
        lidar_name (str | None, optional): If `lidar_nick` is not provided, `lidar_name` is mandatory. Defaults to None.
        subdir (str | None, optional): Subdirectory where the file is located (e.g., -45, +45, North, South). Defaults to None.

    Returns:
        str: filename
    """

    if lidar_nick is None and lidar_name is None:
        raise NameError("lidar_nick or lidar_name must be provided.")

    if lidar_nick is None and lidar_name is not None:
        if lidar_name in LIDAR_INFO["metadata"]["name2nick"].keys():
            lidar_nick = LIDAR_INFO["metadata"]["name2nick"][lidar_name]

    if measurement_type in ["RS", "HF"]:
        if add_hour:
            date_str = date.strftime("%Y%m%d_%H%M")
        else:
            date_str = date.strftime("%Y%m%d")
        filename_ = f"{lidar_nick}_{data_level}_P{measurement_type.lower()}_{signal_type.lower()}_{telescope}_{date_str}"
    else:
        date_str = date.strftime("%Y%m%d_%H%M")
        filename_ = f"{lidar_nick}_{data_level}_P{measurement_type.lower()}{subdir}_{signal_type.lower()}_{telescope}_{date_str}"

    filename = filename_ + ".nc"

    return filename


def filename2path(
    filename: str,
    dir: Path,
    data_level: str | None = None,
    check_exists: bool = True,
) -> Path:
    """Create the path where is located the `filename`.

    Args:
        filename (str): lidar file name (e.g, 'mhc_1a_Prs_rs_xf_20220808_1700.nc')
        dir (Path): Root directory where lidar data are located.
        check_exist (bool, optional): Force to check if the file exists in that path. If False, it raises an error. Defaults to True.

    Returns:
        filepath (Path): lidar file path.
    """

    if not dir.exists() or not dir.is_dir():
        raise ValueError("Path must be provided")

    lidar_nick, _, *_, date = filename2info(filename)

    lidar_name: str = LIDAR_INFO["metadata"]["nick2name"][lidar_nick]

    if data_level is None:
        filepath = (
            dir
            / lidar_name
            / f"{date.year}"
            / f"{date.month:02d}"
            / f"{date.day:02d}"
            / filename
        )
    else:
        filepath = (
            dir
            / lidar_name
            / data_level
            / f"{date.year}"
            / f"{date.month:02d}"
            / f"{date.day:02d}"
            / filename
        )

    if check_exists and not filepath.is_file():
        raise FileNotFoundError(f"{filepath} does not exist.")

    return filepath


def info2path(
    lidar_name: str,
    date: datetime | date,
    dir: Path,
    data_level: str = "1a",
    measurement_type: str = "RS",
    signal_type: str = "rs",
    telescope: str = "xf",
    subdir: str | None = None,
    check_exist: bool = False,
    add_hour: bool = False,
) -> Path:
    """`directory_from_info` provides the directory of 1a-lidar files.

    Args:
        lidar_name (str): Lidar name. Options in `LIDAR_INFO`.
        channel (str): channel code. Options in `LIDAR_INFO`.
        date (datetime.date): Date of measurement.
        dir (Path): Root where lidar data are storaged.
        data_level (str, optional): Data level. Defaults to '1a'.

    Raises:
        NotFoundErr: Directory created does not exist.

    Returns:
        Path: Directory with 1a-lidar files.
    """

    if not dir.exists() or not dir.is_dir():
        raise NotADirectoryError("dir not found.")

    filename = info2filename(
        date,
        data_level=data_level,
        lidar_name=lidar_name,
        measurement_type=measurement_type,
        signal_type=signal_type,
        telescope=telescope,
        subdir=subdir,
        add_hour=add_hour,
    )

    filepath = filename2path(
        filename, dir, data_level=data_level, check_exists=check_exist
    )

    return filepath


def search_dc(
    rs_path: Path,
    session_period: tuple[np.datetime64, np.datetime64] | np.ndarray,
    force_dc_in_session: bool = False,
) -> Path:
    # TODO: Change this documentation
    # TODO: Is fine only searching in the same day?
    """`find_dc` searches a DC file path near to the `date` provided that can be used to preprocess the `lidar_data` data.

    Args:
        lidar_nick (str): Nick of lidar. Options in `LIDAR_INFO`.
        telescope (str, optional): Telescope type code Options['xf', 'ff', 'nf']
        session_period (tuple[datetime.datetime, datetime.datetime]): Period where the DC is searched. If not found, the nearest is provided.
        force_dc_in_session (bool, optional): Force the DC measurment to be in the period, otherwise it raises an error. Defaults to False.
    Returns:
        Path: Path of the DC measurement.
    """
    telescope = filename2info(rs_path.name)[4]
    candidates = list(rs_path.parent.glob(f"*1a_Pdc_*{telescope}*"))

    for dc_path in candidates:
        dc_npdate = np.datetime64(filename2info(dc_path.name)[-1])
        if (session_period[0] <= dc_npdate) and (dc_npdate <= session_period[1]):
            return dc_path

    if force_dc_in_session:
        raise FileNotFoundError(f"Cannot find a dc for times {session_period}")

    candidates, candidate_dates = extract_filenames_dates_from_wildcard(
        rs_path.parent.parent, "*1a_Pdc_*"
    )
    idx = np.abs(
        candidate_dates.astype("M8[ns]").astype(float)
        - np.array(session_period).astype("M8[ns]").astype(float).mean()
    ).argmin()
    dc_path = candidates[idx]

    return dc_path


def extract_filenames_dates_from_wildcard(
    directory: Path, file_wildcard: str
) -> tuple[list[Path], np.ndarray]:
    """Provides file list and its dates from file wildcard. WARNING: slow function.

    Args:
        directory (Path): Directory where files will be searched, including subdirectories.
        file_wildcard (str): string file wildcard (e.g., '*Prs*.nc')

    Raises:
        FileNotFoundError: Any file found.

    Returns:

        np.ndarray: array of dates.
    """

    candidates = list(filter(lambda p: p.is_file(), directory.rglob(file_wildcard)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"File not found with description {file_wildcard} in {directory}."
        )

    dates = np.array(
        list(
            map(
                lambda p: np.datetime64(p.name.split(".")[0].split("_")[-1]), candidates
            )
        )
    )
    return candidates, dates


def info2general_path(
    lidar_name: str,
    date: datetime | date | str,
    data_dir: Path,
    data_level: str | None = None,
) -> Path:
    """Similar to info2path but can take only lidar name, data level and dates as only args

    Args:
        lidar_name (str): _description_
        date (datetime | str): _description_
        data_dir (Path): Path up to lidar folders.
        data_level (str): _description_

    Returns:
        Path: _description_
    """

    date = parse_datetime(date)

    if not data_dir.exists() or not data_dir.is_dir():
        raise NotADirectoryError("data_dir not found.")

    if (data_level is not None) and (data_level != "0a"):
        return (
            data_dir
            / lidar_name
            / data_level
            / f"{date.year}"
            / f"{date.month:02}"
            / f"{date.day:02}"
        )
    else:
        return (
            data_dir
            / lidar_name
            / f"{date.year}"
            / f"{date.month:02}"
            / f"{date.day:02}"
        )


def add_required_channels(
    lidar_nick: str,
    channels: list | np.ndarray,
    target_datetime: datetime
) -> list:

    INFO = read_yaml_from_info(lidar_nick, target_datetime)
    if isinstance(channels, np.ndarray):
        channels = channels.tolist()

    required_channels = channels.copy()
    for channel_ in channels:
        if channel_ in INFO["product_channels"].keys():
            required_channels = required_channels + [
                *INFO["product_channels"][channel_]
            ]
    unique_channels = np.unique(np.array(required_channels)).tolist()
    return unique_channels
    
