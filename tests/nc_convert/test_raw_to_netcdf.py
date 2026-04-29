from datetime import date, datetime
from pathlib import Path

import xarray as xr

from lidarpy.nc_convert.measurement import Measurement
from lidarpy.utils.types import MeasurementType


RAW_DIR = Path("tests/data/RAW")
TARGET_DATE = date(2023, 8, 30)
RS_FILENAME = "RS_20230830_0315.zip"


def test_find_alhambra_rs_measurement(alhambra_rs_measurement: Measurement):
    assert alhambra_rs_measurement.path.exists()
    assert alhambra_rs_measurement.path == (
        RAW_DIR / "alhambra" / "2023" / "08" / "30" / RS_FILENAME
    )
    assert alhambra_rs_measurement.type == MeasurementType.RS
    assert alhambra_rs_measurement.session_datetime == datetime(2023, 8, 30, 3, 15)
    assert len(alhambra_rs_measurement.filenames) == 60
    assert alhambra_rs_measurement.has_linked_dc is True


def test_convert_alhambra_rs_to_netcdf(alhambra_rs_nc: Path, tmp_path: Path):
    assert alhambra_rs_nc.exists()
    assert alhambra_rs_nc.suffix == ".nc"
    assert alhambra_rs_nc.is_relative_to(tmp_path)


def test_alhambra_rs_netcdf_contract(alhambra_rs_nc: Path):
    with xr.open_dataset(alhambra_rs_nc) as dataset:
        assert "time" in dataset.dims
        assert "range" in dataset.dims
        assert "channel" in dataset.dims
        assert dataset.sizes["time"] == 60
        assert dataset.sizes["range"] > 0
        assert dataset.sizes["channel"] > 0
        assert dataset.attrs["system"] == "ALHAMBRA"
        assert dataset["time"].notnull().all()
        assert dataset["range"].notnull().all()
