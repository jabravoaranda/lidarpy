from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def built_distributions():
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build distribution artifacts")

    root = Path(__file__).resolve().parents[2]
    build_dir = root / "artifacts" / "packaging_tests"
    dist_dir = build_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    for artifact in dist_dir.glob("lidarpy-*"):
        artifact.unlink()

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(root / ".uv-python"))
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(root / ".venv311"))

    try:
        subprocess.run(
            [uv, "build", "--out-dir", str(dist_dir)],
            cwd=root,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.lower()
        if "socket" in error or "network" in error or "forbidden" in error:
            pytest.skip("uv build requires network/cache access that is unavailable")
        raise

    sdist = next(dist_dir.glob("lidarpy-*.tar.gz"))
    wheel = next(dist_dir.glob("lidarpy-*.whl"))
    return sdist, wheel


def _sdist_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        return set(archive.getnames())


def _wheel_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _contains(names: set[str], suffix: str) -> bool:
    return any(name.endswith(suffix) for name in names)


def _contains_part(names: set[str], part: str) -> bool:
    return any(part in name for name in names)


def test_built_distributions_exclude_coordination_files(built_distributions):
    sdist, wheel = built_distributions

    names = _sdist_names(sdist) | _wheel_names(wheel)

    assert not _contains(names, "AGENTS.md")
    assert not _contains(names, "ROADMAP.md")


def test_built_distributions_exclude_raw_fixtures(built_distributions):
    sdist, wheel = built_distributions

    names = _sdist_names(sdist) | _wheel_names(wheel)

    assert not _contains_part(names, "tests/data/RAW")
    assert not _contains_part(names, "tests\\data\\RAW")


def test_built_wheel_includes_runtime_package_data(built_distributions):
    _, wheel = built_distributions

    names = _wheel_names(wheel)

    assert _contains(names, "lidarpy/info/info_lidars.yml")
    assert _contains(names, "lidarpy/plot/info.yml")
    assert _contains(names, "lidarpy/nc_convert/configs/ALHAMBRA_20231121.toml")
    assert _contains(names, "lidarpy/assets/LOGO_GFAT_150pp")
