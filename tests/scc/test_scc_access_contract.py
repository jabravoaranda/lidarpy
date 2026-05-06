from __future__ import annotations

from datetime import datetime

from lidarpy.scc.scc_access import SCC, settings_from_path


def _measurement_payload(measurement_id: str, **overrides):
    payload = {
        "id": measurement_id,
        "upload": 127,
        "hirelpp": 127,
        "cloudmask": 127,
        "elpp": 127,
        "elda": 127,
        "elquick": 0,
        "elic": 0,
        "eldec": 0,
        "is_being_processed": False,
        "is_delayed": False,
        "is_queued": False,
        "elpp_exit_code": {"exit_code": 0, "description": "ok"},
        "elda_exit_code": {"exit_code": 0, "description": "ok"},
        "elquick_exit_code": {"exit_code": 0, "description": "ok"},
        "elic_exit_code": {"exit_code": 0, "description": "ok"},
        "eldec_exit_code": {"exit_code": 0, "description": "ok"},
        "hirelpp_exit_code": {"exit_code": 0, "description": "ok"},
        "cloudmask_exit_code": {"exit_code": 0, "description": "ok"},
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(
        self,
        *,
        ok=True,
        status_code=200,
        json_data=None,
        text="",
        url="https://scc.example.test/",
        chunks=None,
        cookies=None,
    ):
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.url = url
        self._chunks = chunks or []
        self.cookies = cookies or {"csrftoken": "csrf-token"}

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=1024):
        yield from self._chunks


class FakeSession:
    def __init__(self, get_responses=None, post_responses=None):
        self.auth = None
        self.verify = True
        self.get_responses = get_responses or {}
        self.post_responses = post_responses or {}
        self.get_calls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        response = self.get_responses[url]
        if isinstance(response, list):
            return response.pop(0)
        return response

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        response = self.post_responses[url]
        if isinstance(response, list):
            return response.pop(0)
        return response


def test_scc_client_urls_and_output_dir_are_normalized(tmp_path):
    scc = SCC(("basic-user", "basic-pass"), str(tmp_path), "https://scc.example.test/")

    assert scc.output_dir == tmp_path
    assert scc.session.auth == ("basic-user", "basic-pass")
    assert scc.session.verify is False
    assert scc.login_url == "https://scc.example.test/accounts/login/"
    assert (
        scc.download_preprocessed_pattern
        == "https://scc.example.test/data_processing/measurements/{0}/download-preprocessed/"
    )
    assert (
        scc.api_measurement_pattern
        == "https://scc.example.test/api/v1/measurements/{0}/"
    )


def test_settings_from_path_loads_yaml_and_converts_credentials_to_tuples(tmp_path):
    config_path = tmp_path / "scc.yml"
    config_path.write_text(
        "\n".join(
            [
                "basic_credentials:",
                "  - basic-user",
                "  - basic-pass",
                "website_credentials:",
                "  - web-user",
                "  - web-pass",
                "base_url: https://scc.example.test/",
                f"output_dir: {tmp_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    settings = settings_from_path(config_path)

    assert settings["basic_credentials"] == ("basic-user", "basic-pass")
    assert settings["website_credentials"] == ("web-user", "web-pass")
    assert settings["base_url"] == "https://scc.example.test/"


def test_get_measurement_and_status_helpers_parse_api_response(tmp_path):
    measurement_id = "20240506gr0001"
    scc = SCC(("basic", "secret"), tmp_path, "https://scc.example.test/")
    measurement_url = scc.api_measurement_pattern.format(measurement_id)
    payload = _measurement_payload(
        measurement_id,
        is_being_processed=True,
        is_delayed=False,
        is_queued=True,
    )
    scc.session = FakeSession(
        get_responses={
            measurement_url: [
                FakeResponse(json_data=payload),
                FakeResponse(json_data=payload),
                FakeResponse(json_data=payload),
                FakeResponse(json_data=payload),
            ]
        }
    )

    measurement, status = scc.get_measurement(measurement_id)

    assert status == 200
    assert measurement is not None
    assert measurement.id == measurement_id
    assert measurement.elpp == 127
    assert measurement.rerun_all_url.endswith(
        f"/data_processing/measurements/{measurement_id}/rerun-all/"
    )
    assert scc.is_being_processed(measurement_id) is True
    assert scc.is_delayed(measurement_id) is False
    assert scc.is_queued(measurement_id) is True


def test_measurement_id_for_date_returns_first_free_sequence(tmp_path):
    scc = SCC(("basic", "secret"), tmp_path, "https://scc.example.test/")
    search_url = (
        "https://scc.example.test/api/v1/measurements/"
        "?id__startswith=20240506gr"
    )
    scc.session = FakeSession(
        get_responses={
            search_url: FakeResponse(
                json_data={
                    "objects": [
                        {"id": "20240506gr00"},
                        {"id": "20240506gr01"},
                    ]
                }
            )
        }
    )

    measurement_id = scc.measurement_id_for_date(datetime(2024, 5, 6), "gr")

    assert measurement_id == "20240506gr02"


def test_download_files_writes_streamed_response_to_expected_zip(tmp_path):
    measurement_id = "20240506gr0001"
    scc = SCC(("basic", "secret"), tmp_path, "https://scc.example.test/")
    download_url = scc.download_preprocessed_pattern.format(measurement_id)
    scc.session = FakeSession(
        get_responses={
            download_url: FakeResponse(chunks=[b"zip-", b"payload", b""])
        }
    )

    scc.download_files(measurement_id, "elpp", download_url)

    output_file = tmp_path / f"preprocessed_{measurement_id}.zip"
    assert output_file.read_bytes() == b"zip-payload"


def test_upload_file_posts_payload_and_extracts_measurement_id(tmp_path):
    measurement_id = "20240506gr0001"
    scc = SCC(("basic", "secret"), tmp_path, "https://scc.example.test/")
    measurement_file = tmp_path / "measurement.nc"
    measurement_file.write_bytes(b"netcdf payload")

    scc.session = FakeSession(
        get_responses={
            scc.upload_url: FakeResponse(cookies={"csrftoken": "upload-csrf"})
        },
        post_responses={
            scc.upload_url: FakeResponse(
                url="https://scc.example.test/data_processing/measurements/created/",
                text=f"<h3>Measurement {measurement_id} <small>",
            )
        },
    )

    uploaded_id = scc.upload_file(measurement_file, system_id=729)

    assert uploaded_id == measurement_id
    post_url, post_kwargs = scc.session.post_calls[0]
    assert post_url == scc.upload_url
    assert post_kwargs["data"] == {"system": 729}
    assert post_kwargs["headers"]["X-CSRFToken"] == "upload-csrf"
    assert "data" in post_kwargs["files"]
    post_kwargs["files"]["data"].close()
