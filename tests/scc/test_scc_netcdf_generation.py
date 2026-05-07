from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from lidarpy.nc_convert.measurement import Measurement, info2measurements
from lidarpy.scc.licel2scc import licel2scc
from lidarpy.utils.types import MeasurementType


RAW_DIR = Path("tests/data/RAW")
SCC_CONFIG = Path(
    "src/lidarpy/scc/scc_configFiles/alh_parameters_scc_781_20230713.py"
)
SCC_ID = 781
TARGET_DATE = datetime(2023, 8, 30)
TARGET_PERIOD = slice(
    datetime.combine(TARGET_DATE, time(3, 15)),
    datetime.combine(TARGET_DATE, time(3, 45)),
)
MEASUREMENT_ID = "20230830gra0315"


@pytest.mark.integration
@pytest.mark.slow
def test_generate_30_minute_alhambra_scc_netcdf(tmp_path):
    unzip_dir = tmp_path / "unzipped"
    output_dir = (
        tmp_path
        / "scc_input"
        / "alhambra"
        / "scc"
        / f"scc{SCC_ID}"
        / "2023"
        / "08"
        / "30"
    )
    unzip_dir.mkdir()
    output_dir.mkdir(parents=True)

    measurements = info2measurements(
        lidar_name="alhambra",
        target_date=TARGET_DATE,
        raw_dir=RAW_DIR,
        measurement_type=MeasurementType.RS,
    )
    assert measurements is not None

    measurement = _measurement_with_files_in_period(measurements, TARGET_PERIOD)
    assert measurement is not None
    assert measurement.dc is not None

    files_in_period = measurement.get_filenames_within_datetime_slice(TARGET_PERIOD)
    rs_files = sorted(
        path.as_posix()
        for path in measurement.get_filepaths(
            pattern_or_list=files_in_period, destination=unzip_dir
        )
    )
    assert len(rs_files) == 30

    dc_dir = measurement.dc.unzip(destination=unzip_dir)
    assert dc_dir is not None
    dc_files_pattern = (dc_dir / "20230830" / "*.[0-9]*").as_posix()

    measurement_class = licel2scc.create_custom_class(
        SCC_CONFIG.absolute().as_posix(),
        use_id_as_name=True,
        temperature=20,
        pressure=1013.25,
    )
    try:
        licel2scc.convert_to_scc(
            measurement_class,
            rs_files,
            dc_files_pattern,
            MEASUREMENT_ID,
            output_dir=output_dir,
        )

        output_file = output_dir / f"{MEASUREMENT_ID}.nc"
        assert output_file.exists()

        with xr.open_dataset(output_file) as dataset:
            assert dataset.attrs["Measurement_ID"] == MEASUREMENT_ID
            assert dataset.attrs["RawData_Start_Date"] == "20230830"
            assert dataset.attrs["RawData_Start_Time_UT"] == "031417"
            assert dataset.attrs["RawData_Stop_Time_UT"] == "034453"
            assert dataset.sizes["time"] == 30
            assert "Background_Profile" in dataset

            dataset = dataset.assign_coords(channels=dataset["channel_ID"].values)
            background_profile_mean = (
                dataset["Background_Profile"]
                .sel(channels=2204)
                .mean(dim="time_bck")
                .sel(points=slice(0, 1000))
                .mean(dim="points")
                .item()
            )
            assert np.isclose(background_profile_mean, 5.2114, atol=0.01)
    finally:
        measurement.remove_tmp_unzipped_dir()
        if isinstance(measurement.dc, Measurement):
            measurement.dc.remove_tmp_unzipped_dir()


def _measurement_with_files_in_period(
    measurements: list[Measurement],
    target_period: slice,
) -> Measurement | None:
    for measurement in measurements:
        if measurement.get_filenames_within_datetime_slice(target_period):
            return measurement
    return None
