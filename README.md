# atmolidarpy

`atmolidarpy` is the PyPI distribution for the standalone atmospheric lidar
processing package imported as `lidarpy`.

The project was migrated out of the former `gfatpy.lidar` submodule so that the
lidar processing code can be installed, tested, documented and released on its
own. It includes RAW Licel conversion, preprocessing, quicklooks, synthetic
signal generation, retrieval routines, SCC support modules and package data for
known lidar systems.

Current public documentation:

- [Documentation site](https://jabravoaranda.github.io/lidarpy/)
- [Module reference index](https://jabravoaranda.github.io/lidarpy/references.html)

## Documentation Map

- [Overview](docs/index.html): package scope and repository map.
- [Getting started](docs/getting-started.html): install, first import, first
  validation and expected results.
- [Processing workflow](docs/processing-workflow.html): operational data flow
  from RAW files to NetCDF, preprocessing, quicklooks and retrieval checks.
- [Examples](docs/examples.html): runnable snippets for synthetic signals,
  preprocessing and SCC client contracts.
- [Operational guide](docs/operations.html): local, CI, release, rollback,
  troubleshooting and runtime boundaries.
- [References](docs/references.html): scientific references and public module
  index.
- [Contributing](docs/contributing.html): development workflow and documentation
  maintenance.

## Install

For users:

```powershell
python -m pip install atmolidarpy
python -c "import lidarpy; print(lidarpy.__version__)"
```

Expected result: Python imports `lidarpy` without needing the old `gfatpy`
package. The distribution name and import name are intentionally different:
install `atmolidarpy`, import `lidarpy`.

For local development:

```powershell
uv sync --group dev
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python -m pytest tests -q -m "not slow"
```

The full suite includes RAW fixture conversion and can create large temporary
NetCDF files. On Windows, run large groups in chunks and keep pytest temporary
directories inside the repository.

## Repository Shape

- `src/lidarpy/nc_convert`: RAW/Licel discovery and NetCDF conversion.
- `src/lidarpy/preprocessing`: correction pipeline, overlap and gluing support.
- `src/lidarpy/plot`: quicklook plotting.
- `src/lidarpy/retrieval`: Klett, Raman, overlap, calibration and synthetic
  validation helpers.
- `src/lidarpy/atmo`: molecular atmosphere and Rayleigh utilities.
- `src/lidarpy/depolarization`: calibration and retrieval helpers.
- `src/lidarpy/scc`: SCC conversion, client, resources and plotting helpers.
- `tests`: focused unit, integration, packaging, SCC, docs and synthetic tests.
- `docs`: static GitHub Pages source.
- `scripts`: documentation build and figure generation.

## Release

Current package version: `0.1.3`.

The package is published as `atmolidarpy` on PyPI through GitHub Actions Trusted
Publishing. Releases are tag-driven:

```powershell
git tag v0.1.3
git push origin v0.1.3
```

The `Publish Package` workflow builds sdist and wheel artifacts, checks them
with Twine and publishes to PyPI from the GitHub environment named `pypi`.

Versioning policy while the project remains alpha:

- Bug fixes that keep the public API compatible use `0.1.x`.
- Public API changes use `0.2.0`, `0.3.0`, etc.
- `1.0.0` is reserved for the first stable API intended for production
  dependants such as `gfat-worker`.

Rollback is normally operational rather than destructive: publish a corrected
new version, or pin downstream environments to the last known good version. Do
not delete release tags or PyPI files unless there is a severe security or
legal reason.
