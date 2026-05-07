from __future__ import annotations

from datetime import datetime
import json

from lidarpy.scc import utils
from lidarpy.utils import utils as lidar_utils


def _write_licel_header_file(path, channel_ids):
    channel_lines = [
        f"1 0 1 1 1 0 7.5 532.0 0 0 0 0 12 600 100 {channel_id}"
        for channel_id in channel_ids
    ]
    path.write_bytes(
        "\n".join(
            [
                path.name,
                "Granada 03/05/2023 03:15:00 03/05/2023 03:16:00 660 -3.6 37.16 3",
                f"0 0 0 0 {len(channel_ids)}",
                *channel_lines,
            ]
        ).encode("utf-8")
    )


def _write_scc_config(path, channel_ids, channel_string_ids=None):
    if channel_string_ids is None:
        channel_string_ids = channel_ids
    channel_parameters = {
        channel_id: {"channel_ID": idx, "channel_string_ID": channel_string_id}
        for idx, (channel_id, channel_string_id) in enumerate(
            zip(channel_ids, channel_string_ids), start=1
        )
    }
    path.write_text(
        "general_parameters = {'System': 'TEST'}\n"
        f"channel_parameters = {channel_parameters!r}\n",
        encoding="utf-8",
    )


def test_date_from_filename_parses_numeric_and_licel_month_codes():
    dates = lidar_utils.date_from_filename(
        [
            "RM2350303.151800",
            "RM23A0303.151800",
            "RM23B0303.151800",
            "RM23C0303.151800",
        ]
    )

    assert dates == [
        datetime(2023, 5, 3, 3, 15, 18),
        datetime(2023, 10, 3, 3, 15, 18),
        datetime(2023, 11, 3, 3, 15, 18),
        datetime(2023, 12, 3, 3, 15, 18),
    ]


def test_get_tp_reads_temperature_and_pressure_from_licel_header(tmp_path):
    raw_file = tmp_path / "RM230503.151800"
    raw_file.write_bytes(
        b"Station header\n"
        b"0 1 2 3 4 5 6 7 8 9 10 11 22.5 1008.0\n"
    )

    temperature, pressure = lidar_utils.getTP(str(raw_file))

    assert temperature == 22.5
    assert pressure == 1008.0


def test_get_tp_returns_none_values_for_missing_or_short_header(tmp_path):
    short_header = tmp_path / "RM230503.151800"
    short_header.write_bytes(b"Station header\n0 1 2\n")

    assert lidar_utils.getTP(str(short_header)) == (None, None)
    assert lidar_utils.getTP(str(tmp_path / "missing")) == (None, None)


def test_scc_utils_reexports_licel_helpers_for_legacy_callers():
    assert utils.date_from_filename is lidar_utils.date_from_filename
    assert utils.getTP is lidar_utils.getTP


def test_get_scc_config_id_from_binary_uses_parameter_files(tmp_path):
    raw_file = tmp_path / "RM230503.151800"
    _write_licel_header_file(raw_file, ["BT0", "BT10", "BT15"])
    _write_scc_config(
        tmp_path / "tst_parameters_scc_783_20200101.py", ["BT0", "BT10"]
    )
    _write_scc_config(
        tmp_path / "tst_parameters_scc_781_20200101.py", ["BT0", "BT10", "BT15"]
    )

    assert (
        utils.get_scc_config_id_from_binary(
            raw_file,
            lidar_prefix="tst",
            scc_config_directory=tmp_path,
            target_datetime=datetime(2023, 5, 3),
        )
        == 781
    )


def test_get_scc_config_id_from_binary_uses_actris_rules_for_ties(tmp_path):
    raw_file = tmp_path / "RM230503.151800"
    _write_licel_header_file(raw_file, ["BT0", "BT10", "BT15"])
    _write_scc_config(
        tmp_path / "tst_parameters_scc_781_20200101.py",
        ["BT0", "BT10", "BT15"],
        ["1064fta", "532npa", "531fta"],
    )
    _write_scc_config(
        tmp_path / "tst_parameters_scc_999_20200101.py",
        ["BT0", "BT10", "BT15"],
        ["1064fta", "532npa", "531fta"],
    )
    actris_config = tmp_path / "actris_config.yml"
    actris_config.write_text(
        "\n".join(
            [
                "systems:",
                "  test:",
                "    parameter_prefix: tst",
                "    channel_string_id_pattern: '^(?P<wavelength>\\d+)(?P<telescope>[fn])'",
                '    far_telescope_tokens: ["f"]',
                '    near_telescope_tokens: ["n"]',
                "    elastic_wavelengths: [355, 532, 1064]",
                "    scc_config_rules:",
                "      - has_far: true",
                "        has_near: true",
                "        has_raman: true",
                "        scc_config_id: 781",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        utils.get_scc_config_id_from_binary(
            raw_file,
            lidar_prefix="tst",
            scc_config_directory=tmp_path,
            target_datetime=datetime(2023, 5, 3),
            actris_config_file=actris_config,
        )
        == 781
    )


def test_get_scc_config_id_from_binary_rejects_missing_channels(tmp_path):
    raw_file = tmp_path / "RM230503.151800"
    _write_licel_header_file(raw_file, ["BT0", "BT10"])
    _write_scc_config(
        tmp_path / "tst_parameters_scc_781_20200101.py", ["BT0", "BT10", "BT15"]
    )
    _write_scc_config(tmp_path / "tst_parameters_scc_783_20200101.py", ["BT0", "BT10"])

    assert (
        utils.get_scc_config_id_from_binary(
            raw_file,
            lidar_prefix="tst",
            scc_config_directory=tmp_path,
            target_datetime=datetime(2023, 5, 3),
        )
        == 783
    )


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
