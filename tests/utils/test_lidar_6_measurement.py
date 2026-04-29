from datetime import date, datetime
from pathlib import Path

from lidarpy.nc_convert.measurement import Measurement
from lidarpy.utils.types import MeasurementType


RAW_DIR = Path("tests/data/RAW")
TARGET_DATE = datetime(2023, 8, 30)


def test_measurement_class_properties(alhambra_rs_measurement: Measurement, tmp_path: Path):
    data_dir = RAW_DIR / "alhambra" / "2023" / "08" / "30"

    assert type(alhambra_rs_measurement) is Measurement
    assert alhambra_rs_measurement.path == data_dir / "RS_20230830_0315.zip"
    assert alhambra_rs_measurement.type == MeasurementType.RS
    assert alhambra_rs_measurement.session_datetime == datetime(2023, 8, 30, 3, 15)
    assert len(alhambra_rs_measurement.filenames) == 60
    assert alhambra_rs_measurement.lidar_name.value == "alhambra"
    assert alhambra_rs_measurement.telescope == "xf"
    assert alhambra_rs_measurement.is_zip is True
    assert len(alhambra_rs_measurement.datetimes) == 60
    assert sorted(alhambra_rs_measurement.datetimes)[0] == datetime(2023, 8, 30, 3, 15, 18)
    assert sorted(alhambra_rs_measurement.datetimes)[-1] == datetime(2023, 8, 30, 4, 15, 29)
    assert sorted(alhambra_rs_measurement.unique_dates)[0] == date(2023, 8, 30)
    assert alhambra_rs_measurement.sub_dirs[0] == TARGET_DATE.strftime("%Y%m%d")
    assert alhambra_rs_measurement.has_linked_dc is True
    assert type(alhambra_rs_measurement.dc) is Measurement

    file_set = alhambra_rs_measurement.get_filepaths(destination=tmp_path)
    assert file_set is not None
    assert isinstance(file_set, set)
    assert len(file_set) == 60
    assert all(tmp_path in filepath.parents for filepath in file_set)

    alhambra_rs_measurement.remove_tmp_unzipped_dir()


def test_get_files_within_period(alhambra_rs_measurement: Measurement, tmp_path: Path):
    date_ini = datetime(2023, 8, 30, 3, 15)
    date_end = datetime(2023, 8, 30, 3, 20)
    files = alhambra_rs_measurement.get_filenames_within_datetime_slice(slice(date_ini, date_end))

    assert len(files) == 5

    file_set = alhambra_rs_measurement.get_filepaths(
        pattern_or_list=files,
        destination=tmp_path,
    )
    assert file_set is not None
    assert len(file_set) == 5
    assert all(tmp_path in filepath.parents for filepath in file_set)

    alhambra_rs_measurement.remove_tmp_unzipped_dir()


def test_remove_unzipped_path(alhambra_rs_measurement: Measurement, tmp_path: Path):
    filepath = alhambra_rs_measurement.unzip(destination=tmp_path)

    assert filepath is not None
    assert filepath.exists()
    assert tmp_path in filepath.parents

    alhambra_rs_measurement.remove_tmp_unzipped_dir()
    assert alhambra_rs_measurement._unzipped_dir is None
