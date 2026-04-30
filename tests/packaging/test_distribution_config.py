from __future__ import annotations

import tomllib
from pathlib import Path


def test_coordination_files_are_not_in_sdist():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert "/AGENTS.md" not in sdist["include"]
    assert "/ROADMAP.md" not in sdist["include"]
    assert "/AGENTS.md" in sdist["exclude"]
    assert "/ROADMAP.md" in sdist["exclude"]


def test_wheel_only_packages_lidarpy_source_tree():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert wheel["packages"] == ["src/lidarpy"]
    assert "src/lidarpy/assets/*" in wheel["artifacts"]


def test_runtime_dependencies_do_not_include_removed_notebook_stack():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependency_names = {
        dependency.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
        for dependency in pyproject["project"]["dependencies"]
    }

    assert {"psutil", "pytz"}.issubset(dependency_names)
    assert {
        "notebook",
        "ipywidgets",
        "seaborn",
        "pyarrow",
        "distributed",
        "atmospheric_lidar",
        "typer",
    }.isdisjoint(dependency_names)
