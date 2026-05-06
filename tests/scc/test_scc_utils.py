from __future__ import annotations

from datetime import datetime
import json

from lidarpy.scc import utils


def test_date_from_filename_parses_numeric_and_licel_month_codes():
    dates = utils.date_from_filename(
        [
            "RM2350303.151800",
            "RM23A0303.151800",
            "RM23B0303.151800",
            "RM23C0303.151800",
        ]
    )

    assert dates == [
        datetime(2023, 5, 3, 3, 15),
        datetime(2023, 10, 3, 3, 15),
        datetime(2023, 11, 3, 3, 15),
        datetime(2023, 12, 3, 3, 15),
    ]


def test_get_tp_reads_temperature_and_pressure_from_licel_header(tmp_path):
    raw_file = tmp_path / "RM230503.151800"
    raw_file.write_bytes(
        b"Station header\n"
        b"0 1 2 3 4 5 6 7 8 9 10 11 22.5 1008.0\n"
    )

    temperature, pressure = utils.getTP(str(raw_file))

    assert temperature == 22.5
    assert pressure == 1008.0


def test_get_tp_returns_none_values_for_missing_or_short_header(tmp_path):
    short_header = tmp_path / "RM230503.151800"
    short_header.write_bytes(b"Station header\n0 1 2\n")

    assert utils.getTP(str(short_header)) == (None, None)
    assert utils.getTP(str(tmp_path / "missing")) == (None, None)


def test_get_campaign_config_builds_default_operational_config():
    campaign = utils.get_campaign_config(
        scc_config_id=729,
        hour_ini=0.0,
        hour_end=24.0,
        hour_resolution=1.0,
        timestamp=1,
        slot_name_type=2,
    )

    assert campaign == {
        "name": "operational",
        "lidar_config": {
            "operational": {
                "scc": 729,
                "hour_ini": 0.0,
                "hour_end": 24.0,
                "hour_res": 1.0,
                "timestamp": 1,
                "slot_name_type": 2,
            }
        },
    }


def test_get_campaign_config_loads_json_file(tmp_path):
    campaign_file = tmp_path / "campaign.json"
    expected = {"name": "campaign", "lidar_config": {"slot": {"scc": 729}}}
    campaign_file.write_text(json.dumps(expected), encoding="utf-8")

    assert utils.get_campaign_config(str(campaign_file)) == expected


def test_get_scc_config_imports_parameter_module(tmp_path):
    config_file = tmp_path / "scc_parameters_for_test.py"
    config_file.write_text(
        "general_parameters = {'system': 'ALHAMBRA'}\n"
        "channel_parameters = {729: {'name': '532fta'}, 730: {'name': '532ftp'}}\n",
        encoding="utf-8",
    )

    config = utils.get_scc_config(str(config_file))

    assert config == {
        "general_parameters": {"system": "ALHAMBRA"},
        "channel_parameters": {
            729: {"name": "532fta"},
            730: {"name": "532ftp"},
        },
        "channels": [729, 730],
    }


def test_check_scc_output_inlocal_requires_expected_output_directories(tmp_path):
    assert utils.check_scc_output_inlocal(str(tmp_path)) is False

    for name in (
        "hirelpp",
        "cloudmask",
        "scc_preprocessed",
        "scc_optical",
        "scc_plots",
    ):
        (tmp_path / name).mkdir()

    assert utils.check_scc_output_inlocal(str(tmp_path)) is True
