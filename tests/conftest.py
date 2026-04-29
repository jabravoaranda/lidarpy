import shutil
import pytest
import matplotlib
from pathlib import Path
from datetime import date

from lidarpy.nc_convert.measurement import (
    Measurement,
    info2measurements,    
)
from lidarpy.utils.types import LidarName, MeasurementType

matplotlib.use("Agg")

RAW_DIR = Path(r"./tests/data/RAW")
OUTPUT_DIR = Path(r"./tests/data/PRODUCTS")
ALHAMBRA_TEST_DATE = date(2023, 8, 30)
ALHAMBRA_RS_FILENAME = "RS_20230830_0315.zip"
ALHAMBRA_DC_FILENAME = "DC_20230830_0315.zip"


def pytest_addoption(parser):
    parser.addoption("--runmhc", action="store_true", default=False)
    parser.addoption("--runalh", action="store_true", default=False)
    parser.addoption("--runlinc", action="store_true", default=False)
    parser.addoption("--cleanup", action="store_true", default=False)


@pytest.fixture(scope="session")
def run_linc(pytestconfig):
    return pytestconfig.getoption("runlinc")


@pytest.fixture(scope="session")
def run_linc_mhc(pytestconfig):
    return pytestconfig.getoption("runmhc")


@pytest.fixture(scope="session")
def run_linc_alh(pytestconfig):
    return pytestconfig.getoption("runalh")


@pytest.fixture(scope="session")
def cleanup(pytestconfig):
    return pytestconfig.getoption("cleanup")


@pytest.fixture(scope="session")
def linc_files(run_linc, run_linc_mhc, run_linc_alh, cleanup):
    if run_linc:
        run_linc_mhc = True
        run_linc_alh = True

    if run_linc_mhc:
        measurements = info2measurements(
            lidar_name="mulhacen", target_date=date(2022, 8, 8), raw_dir=RAW_DIR
        )
        if measurements is None:
            raise Exception("No measurements found")
        for measurement in measurements:
            measurement.to_nc(by_dates=False, output_dir=OUTPUT_DIR, overwrite=True)
            measurement.remove_tmp_unzipped_dir()

        measurements = info2measurements(
            lidar_name="mulhacen", target_date=date(2022, 6, 17), raw_dir=RAW_DIR
        )
        if measurements is None:
            raise Exception("No measurements found")
        for measurement in measurements:
            measurement.to_nc(by_dates=False, output_dir=OUTPUT_DIR, overwrite=True)
            measurement.remove_tmp_unzipped_dir()

    if run_linc_alh:
        measurements = info2measurements(
            lidar_name="alhambra", target_date=date(2023, 2, 22), raw_dir=RAW_DIR
        )
        if measurements is None:
            raise Exception("No measurements found")
        for measurement in measurements:
            measurement.to_nc(by_dates=False, output_dir=OUTPUT_DIR, overwrite=True)
            measurement.remove_tmp_unzipped_dir()

        measurements = info2measurements(
            lidar_name="alhambra", target_date=date(2023, 8, 30), raw_dir=RAW_DIR
        )
        if measurements is None:
            raise Exception("No measurements found")
        for measurement in measurements:
            measurement.to_nc(by_dates=False, output_dir=OUTPUT_DIR, overwrite=True)
            measurement.remove_tmp_unzipped_dir()

    yield  # This yield waits until fixture is out of scope (after running all tests in this case)  # noqa: E501
    if cleanup:
        if run_linc_mhc:
            mhc_rm_path = OUTPUT_DIR / "mulhacen" / "1a" / "2022"
            if mhc_rm_path.exists():
                shutil.rmtree(mhc_rm_path)
        if run_linc_alh:
            alh_rm_path = OUTPUT_DIR / "alhambra" / "1a" / "2023"
            if alh_rm_path.exists():
                shutil.rmtree(alh_rm_path)


@pytest.fixture
def raw_dir() -> Path:
    return RAW_DIR


@pytest.fixture
def alhambra_rs_measurement(raw_dir: Path) -> Measurement:
    measurements = info2measurements(
        lidar_name=LidarName.alh,
        target_date=ALHAMBRA_TEST_DATE,
        raw_dir=raw_dir,
        measurement_type=MeasurementType.RS,
    )
    assert measurements is not None
    return next(m for m in measurements if m.path.name == ALHAMBRA_RS_FILENAME)


@pytest.fixture
def alhambra_dc_measurement(raw_dir: Path) -> Measurement:
    measurements = info2measurements(
        lidar_name=LidarName.alh,
        target_date=ALHAMBRA_TEST_DATE,
        raw_dir=raw_dir,
        measurement_type=MeasurementType.DC,
    )
    assert measurements is not None
    return next(m for m in measurements if m.path.name == ALHAMBRA_DC_FILENAME)


@pytest.fixture
def alhambra_rs_nc(alhambra_rs_measurement: Measurement, tmp_path: Path) -> Path:
    output_paths = alhambra_rs_measurement.to_nc(output_dir=tmp_path, overwrite=True)
    assert output_paths is not None
    assert len(output_paths) == 1
    yield output_paths[0]
    alhambra_rs_measurement.remove_tmp_unzipped_dir()


@pytest.fixture
def alhambra_dc_nc(alhambra_dc_measurement: Measurement, tmp_path: Path) -> Path:
    output_paths = alhambra_dc_measurement.to_nc(output_dir=tmp_path, overwrite=True)
    assert output_paths is not None
    assert len(output_paths) == 1
    yield output_paths[0]
    alhambra_dc_measurement.remove_tmp_unzipped_dir()


@pytest.fixture
def alhambra_rs_dc_nc(
    alhambra_rs_measurement: Measurement,
    alhambra_dc_measurement: Measurement,
    tmp_path: Path,
) -> tuple[Path, Path]:
    dc_output_paths = alhambra_dc_measurement.to_nc(
        output_dir=tmp_path, overwrite=True
    )
    rs_output_paths = alhambra_rs_measurement.to_nc(
        output_dir=tmp_path, overwrite=True
    )

    assert dc_output_paths is not None
    assert rs_output_paths is not None
    assert len(dc_output_paths) == 1
    assert len(rs_output_paths) == 1

    yield rs_output_paths[0], dc_output_paths[0]

    alhambra_rs_measurement.remove_tmp_unzipped_dir()
    alhambra_dc_measurement.remove_tmp_unzipped_dir()


@pytest.fixture(scope="session")
def clean_depo_calibrations():
    calib_dir = {}
    calib_dir["alh"] = Path(
        r"tests\data\PRODUCTS\alhambra\QA\depolarization_calibration"
    )
    calib_dir["mhc"] = Path(
        r"tests\data\PRODUCTS\alhambra\QA\depolarization_calibration"
    )
    for calib_dir in calib_dir.values():
        if calib_dir.exists():
            shutil.rmtree(calib_dir)
        calib_dir.mkdir(parents=True)

