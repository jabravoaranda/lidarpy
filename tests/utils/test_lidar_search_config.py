from datetime import datetime, time
from pathlib import Path

from linc import get_config

from lidarpy.utils.types import LidarName
from lidarpy.nc_convert.utils import search_config_file


def test_search_config_alh_with_configfile(alhambra_rs_measurement):
    target_datetime = datetime.combine(alhambra_rs_measurement.unique_dates[0], time(0, 0))
    config_filepath = search_config_file(
        LidarName.alh,
        target_datetime,
        Path(r".\src\lidarpy\nc_convert\configs\ALHAMBRA_20180101.toml"),
    )
    config = get_config(config_filepath)
    assert config.lidar.attrs["system"] == "ALHAMBRA"


def test_search_config_alh_with_datetime(alhambra_rs_measurement):
    target_datetime = datetime.combine(alhambra_rs_measurement.unique_dates[0], time(0, 0))
    config_filepath = search_config_file(
        LidarName.alh,
        target_datetime,
        None,
    )
    config = get_config(config_filepath)
    assert config.lidar.attrs["system"] == "ALHAMBRA"
