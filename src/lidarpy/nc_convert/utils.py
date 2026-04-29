import re
import sys
import shutil
import psutil
from pathlib import Path
from loguru import logger
from datetime import datetime

from lidarpy.utils.types import LidarName
from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.general_utils.io import find_nearest_filepath

RAW_FIRST_LETTER = LIDAR_INFO["metadata"]["licel_file_wildcard"]

logger.add(sys.stdout, level="INFO")


def search_config_file(
    lidar_name: LidarName,
    target_datetime: datetime,
    opt_config: Path | str | None,
) -> Path:
    """Searches for a configuration file.

    Args:
        lidar_name (LidarName): Name of the lidar.
        opt_config (Path | str | None, optional): Path to the configuration file. Defaults to None.

    Raises:
        FileNotFoundError: No configution file found in opt_config.

    Returns:
        Config: Configuration object.
    """
    if opt_config is not None:
        if isinstance(opt_config, Path):
            config_path = opt_config
        elif isinstance(opt_config, str):
            config_path = Path(opt_config)

        if not config_path.exists():
            raise FileNotFoundError(f"Configution file {opt_config} not found.")
    else:
        config_dir = Path(__file__).parent / "configs"
        config_path = find_nearest_filepath(
            config_dir,
            f"{lidar_name.value.upper()}*.toml",
            1,
            target_datetime,
            and_previous=True,
        )
        if not config_path.exists():
            raise FileNotFoundError(f"No configution file found in {config_dir}")
    return config_path


def cleanup_tmp_folders():
    """Delete temporary folders created during the process.

    Raises:

        - OSError: If there is an error deleting the temporary folder.
    """    
    pattern = r"tmp_unzipped_[a-zA-Z0-9_]+$"
    for dir_ in Path(__file__).parent.parent.parent.parent.glob("tmp_unzipped_*"):
        if re.search(pattern, dir_.name):
            try:
                # Ensure all files are closed
                for proc in psutil.process_iter(['pid', 'open_files']):
                    for file in proc.info['open_files'] or []:
                        if dir_ in Path(file.path).parents:
                            proc.terminate()
                            proc.wait()
                
                shutil.rmtree(dir_)
                print(f"Temporary folder deleted: {dir_}")
            except psutil.NoSuchProcess:
                print(f"Process already terminated.")
            except psutil.AccessDenied:
                print(f"Access denied to terminate process.")
            except RuntimeError as e:
                print(f"Error terminating process: {e}")