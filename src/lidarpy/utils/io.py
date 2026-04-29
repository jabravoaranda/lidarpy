from pdb import set_trace
import re
from loguru import logger
import yaml
import zipfile
import tempfile
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime

from lidarpy.general_utils.utils import datetime_np2dt


def read_yaml(path: Path | str) -> Any:

    """
    Reads a YAML file and returns the data as a dictionary.

    Args:
        path (Path | str): The path to the YAML file.

    Returns:
        Any: The data contained in the YAML file, typically as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def read_yaml_from_info(lidar_name: str, target_date: datetime) -> Any:
    """
    Reads a YAML file containing information for a specific lidar and date.
    This function searches for the nearest YAML file in the "lidar/info" directory
    that matches the given lidar name and target date. The filename should follow
    the pattern "info_{lidar_name}*.yml" where the date is located at the second
    position in the filename.
    Args:
        lidar_name (str): The name of the lidar.
        target_date (datetime): The target date for which to find the nearest YAML file.
    Returns:
        Any: The contents of the YAML file.
    """

    file = find_nearest_filepath(
        dir=Path(__file__).parent.parent.absolute() / "info",
        wildcard_filename=f"info_{lidar_name}*.yml",
        date_location_in_filename=2,
        date=target_date,
        and_previous=True,
    )
    return read_yaml(file)


def find_nearest_filepath(
    dir: Path,
    wildcard_filename: str,
    date_location_in_filename: int,
    date: datetime,
    and_previous: bool = False,
) -> Path:
    """Finds the nearest file path in the specified directory based on a wildcard filename pattern and a target date.

    Args:
        dir (Path): The directory to search for files.
        wildcard_filename (str): The wildcard filename pattern to match files.
        date_location_in_filename (int): The index of the date in the filename split by underscores.
        date (datetime): The target date to find the nearest file.
        and_previous (bool, optional): If True, then find the nearest file before the target date. Defaults to False.

    Returns:
        Path: The path to the nearest file.

    Raises:
        ValueError: If no file is found in the directory.

    """
    # Get all candidate file paths
    candidates = np.array([*dir.rglob(wildcard_filename)])
    # Raise an error if no candidate files are found
    if len(candidates) == 0:
        raise ValueError(
            f"No files found in {dir} with the pattern {wildcard_filename}."
        )
    # Extract the dates from candidate file names
    dates_string = [
        p.name.split(".")[0].split("_")[date_location_in_filename] for p in candidates
    ]

    # Convert dates_string to numpy datetime64 objects
    dates = np.array(
        [datetime.strptime(date_str, "%Y%m%d") for date_str in dates_string]
    )

    # if and_previous is True, then find the nearest date before date_target
    if and_previous:
        candidates = candidates[dates < datetime_np2dt(date)]
        dates = dates[dates < datetime_np2dt(date)]

    # After filtering (e.g., when and_previous=True) there may be no candidates left.
    # Check and raise a clear error instead of letting np.argmin fail on an empty array.
    if len(candidates) == 0:
        raise ValueError(
            f"No files found in {dir} before {date.strftime('%Y-%m-%d')} with the pattern {wildcard_filename}."
        )

    # Calculate absolute differences between each date and date_target
    date_diffs = np.abs(dates - datetime_np2dt(date))

    # Find the index of the minimum absolute difference
    try:
        nearest_date_index = np.argmin(date_diffs)
    except ValueError:
        raise ValueError("No file found.")

    # Get the path of the nearest file
    path = candidates[nearest_date_index]

    # Raise an error if the path does not exist
    if not path.exists():
        raise ValueError("No file found.")

    return path


def unzip_file(
    file: Path,
    pattern_or_list: str | list[str] = "*.*",
    destination: Path | None = None,
) -> tempfile.TemporaryDirectory | None:
    """Unzips a zip file to a temporary directory.

    Args:
        file (Path): The path to the zip file.
        pattern (str, optional): The pattern to match files to extract. Defaults to '*.*'.
        destination (Path | None, optional): The destination directory to extract the files. Defaults to None creating a temporary directory.

    Raises:
        ValueError: file is not a zip file.
        ValueError: file is not a valid zip file.
        ValueError: An error occurred.

    Returns:
        Path: The path to the unzipped directory.
    """
    if not file.suffix.endswith("zip"):
        raise ValueError(f"The file {file} is not a zip file.")

    # Create a temporary directory
    if destination is None:        
        unzipped_dir = tempfile.TemporaryDirectory(prefix="tmp_unzipped_", dir=Path.cwd().absolute(), ignore_cleanup_errors=False)        
    else:
        unzipped_dir = tempfile.TemporaryDirectory(prefix="tmp_unzipped_", dir=destination, ignore_cleanup_errors=False)
    
    with zipfile.ZipFile(file, "r") as zip_ref:
        if isinstance(pattern_or_list, list):
            files_to_extract = [
                file
                for file in zip_ref.namelist()
                if any(pattern in file for pattern in set(pattern_or_list))
            ]
        elif isinstance(pattern_or_list, str):
            if pattern_or_list == "*.*":
                files_to_extract = zip_ref.namelist()
            else:
                pattern = re.compile(pattern_or_list)
                files_to_extract = [
                    file_info.filename
                    for file_info in zip_ref.infolist()
                    if bool(re.search(pattern, file_info.filename))
                ]        
        if len(files_to_extract) == 0:
            logger.warning(f"No files found in {file} with the pattern {pattern_or_list}.")
            return None            
        for file_ in files_to_extract:
            zip_ref.extract(file_, path=Path(unzipped_dir.name))   
        
    return unzipped_dir
