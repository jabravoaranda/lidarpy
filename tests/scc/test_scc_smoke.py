from __future__ import annotations

import importlib
import importlib.resources as resources
from datetime import datetime

from lidarpy.scc.io import move2odir
from lidarpy.scc.utils import date_from_filename, getTP


def test_scc_core_modules_import():
    for module_name in (
        "lidarpy.scc",
        "lidarpy.scc.io",
        "lidarpy.scc.scc",
        "lidarpy.scc.scc_access",
        "lidarpy.scc.transfer",
        "lidarpy.scc.utils",
        "lidarpy.scc.licel2scc.licel2scc",
        "lidarpy.scc.plot.scc_zip",
    ):
        assert importlib.import_module(module_name).__name__ == module_name


def test_scc_package_data_is_available():
    campaign = (
        resources.files("lidarpy.scc.scc_campaigns")
        / "scc_campaign_sample.json"
    )
    config = (
        resources.files("lidarpy.scc.scc_configFiles")
        / "alh_parameters_scc_729.py"
    )
    plot_info = resources.files("lidarpy.scc.plot") / "info.yml"

    assert campaign.is_file()
    assert config.is_file()
    assert plot_info.is_file()


def test_date_from_filename_parses_licel_month_codes():
    dates = date_from_filename(
        [
            "R2410100.123456",
            "R24A0100.123456",
            "R24B0100.123456",
            "R24C0100.123456",
        ]
    )

    assert dates == [
        datetime(2024, 1, 1, 0, 12),
        datetime(2024, 10, 1, 0, 12),
        datetime(2024, 11, 1, 0, 12),
        datetime(2024, 12, 1, 0, 12),
    ]


def test_getTP_reads_temperature_and_pressure_from_licel_header(tmp_path):
    licel_file = tmp_path / "R2401010.123456"
    licel_file.write_bytes(
        b"R2401010.123456\n"
        b"1 2 3 4 5 6 7 8 9 10 11 12 18.5 930.25\n"
    )

    temperature, pressure = getTP(str(licel_file))

    assert temperature == 18.5
    assert pressure == 930.25


def test_move2odir_moves_file_to_destination(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "out"
    source.write_text("scc payload", encoding="utf-8")

    moved = move2odir(str(source), str(destination))

    assert moved is True
    assert not source.exists()
    assert (destination / source.name).read_text(encoding="utf-8") == "scc payload"
