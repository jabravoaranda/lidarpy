from datetime import datetime
from pathlib import Path

import xarray as xr

from lidarpy.nc_convert.measurement import Measurement
from lidarpy.utils.types import MeasurementType


RAW_DIR = Path("tests/data/RAW")
DC_FILENAME = "DC_20230830_0315.zip"


def test_find_alhambra_dc_measurement(alhambra_dc_measurement: Measurement):
    assert alhambra_dc_measurement.path.exists()
    assert alhambra_dc_measurement.path == (
        RAW_DIR / "alhambra" / "2023" / "08" / "30" / DC_FILENAME
    )
    assert alhambra_dc_measurement.type == MeasurementType.DC
    assert alhambra_dc_measurement.session_datetime == datetime(2023, 8, 30, 3, 15)
    assert len(alhambra_dc_measurement.filenames) > 0
    assert alhambra_dc_measurement.has_linked_dc is False


def test_convert_alhambra_dc_to_netcdf(alhambra_dc_nc: Path, tmp_path: Path):
    assert alhambra_dc_nc.exists()
    assert alhambra_dc_nc.suffix == ".nc"
    assert alhambra_dc_nc.is_relative_to(tmp_path)


def test_alhambra_dc_netcdf_contract(alhambra_dc_nc: Path):
    with xr.open_dataset(alhambra_dc_nc) as dataset:
        assert "time" in dataset.dims
        assert "range" in dataset.dims
        assert "channel" in dataset.dims
        assert dataset.sizes["time"] > 0
        assert dataset.sizes["range"] > 0
        assert dataset.sizes["channel"] > 0
        assert dataset.attrs["system"] == "ALHAMBRA"
        assert dataset["time"].notnull().all()
        assert dataset["range"].notnull().all()
