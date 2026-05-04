import os
from pathlib import Path

from datetime import datetime

from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.general_utils.io import find_nearest_filepath, read_yaml_from_info


def GHK_simulator(
    lidar_nick: str,
    calibrator: str,
    target_datetime: datetime,
    output_dir: Path,
    channel: str | None = None,
) -> list[Path]:
    """Run GHK simulator for a given lidar, calibrator and channel.

    Args:

        - lidar_nick (str): lidar nickname
        - calibrator (str): calibrator nickname (eg., "hwp", "rot", "pol")
        - target_datetime (datetime): Target datetime
        - output_dir (Path): Output directory
        - channel (str | None, optional): Lidar channel in format dddss (e.g., 532n). Defaults to None (all channels).

    Raises:

        - NotADirectoryError: Directory not found
        - FileNotFoundError: File not found

    Returns:
    
        - list[Path]: List of output paths
    """

    if not output_dir.exists() or not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} not found.")

    if channel is None:
        INFO = read_yaml_from_info(lidar_nick, target_datetime)
        channels = INFO["GHK_channels"]
    else:
        channels = [channel]
    output_paths = []
    for channel_ in channels:
        # Ini file for each channel
        ini_ghk_dir = Path(__file__).parent.absolute() / "GHK" / "system_settings"
        ini_filepath = find_nearest_filepath(
            ini_ghk_dir,
            f"optic_input_{lidar_nick}_{calibrator}_{channel_}*.py",
            5,
            target_datetime,
            and_previous=True,
        )

        if not ini_filepath.exists():
            raise FileNotFoundError(f"Ini file not found: {ini_filepath}.")

        output_path_ = run_GHK_simulator(ini_filepath, output_dir)
        if isinstance(output_path_, Path):
            output_paths.append(output_path_)
    return output_paths


def run_GHK_simulator(ini_path: Path, output_dir: Path) -> Path:
    """Run GHK simulator for a given ini file.

    Args:

        - ini_path (Path): Input ini file
        - output_dir (Path): Output directory

    Raises:

        - FileNotFoundError: File not found

    Returns:

        - Path: Output path
    """
    depo_path = Path(__file__).parent.absolute()

    GHK_program = depo_path / "GHK" / "GHK_0.9.8h_Py3.7.py"

    output_dir.mkdir(parents=True, exist_ok=True)

    os.system(f"python {GHK_program} {ini_path} {output_dir.absolute()}")

    output_path = [*output_dir.rglob(f"*{ini_path.name.split('.')[0]}*.dat")]

    if isinstance(output_path, list) and len(output_path) == 1:
        output_path = output_path[0]
        return output_path
    else:
        raise FileNotFoundError(f"Output file not found in {output_dir}.")
