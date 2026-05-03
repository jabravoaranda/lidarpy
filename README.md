# atmolidarpy

`atmolidarpy` is a standalone distribution for atmospheric lidar data
processing, migrated from the former `gfatpy.lidar` submodule. The importable
Python package remains `lidarpy`.

Current public documentation:

- [Documentation site](docs/index.html)
- [Module reference index](docs/references.html)

Install the package from PyPI with:

```powershell
python -m pip install atmolidarpy
```

## Development

Run the local test suite with:

```powershell
$env:PYTHONPATH='src'; $env:MPLBACKEND='Agg'; .\.venv\Scripts\python -m pytest tests -q
```

For a faster development loop that skips RAW conversion and larger integration
checks:

```powershell
$env:PYTHONPATH='src'; $env:MPLBACKEND='Agg'; .\.venv\Scripts\python -m pytest tests -q -m "not slow"
```

GitHub Actions runs the same pytest suite on pushes and pull requests targeting
`main` and `develop`.

## Release

Tag releases as `v*`. The publish workflow builds the source distribution and
wheel, checks them with Twine, and publishes through PyPI Trusted Publishing.
Before the first real release, configure the PyPI project to trust the GitHub
repository environment named `pypi`.
