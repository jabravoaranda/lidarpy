# ROADMAP

## Current Goal

Migrate the lidar processing workflow from `gfatpy/lidar` to standalone
`lidarpy`.

## Done

- Created standalone `src/lidarpy` package.
- Added `pyproject.toml` for `lidarpy`.
- Configured pytest with `--basetemp=.pytest_tmp`.
- Added `.gitignore` entries for local runtimes, pytest temp folders, unzip
  folders, and generated artifacts.
- Kept only the current minimal ALHAMBRA RAW fixtures:
  - `tests/data/RAW/alhambra/2023/08/30/RS_20230830_0315.zip`
  - `tests/data/RAW/alhambra/2023/08/30/DC_20230830_0315.zip`
- Removed legacy modules/tests outside the current migration focus:
  `scc`, `quality_assurance`, `depolarization`, `mulhacen`, `synthetic`, and
  old retrieval tests.
- Kept `retrieval` for now because `apply_ov=True` may need overlap retrieval.
- Added tests for:
  - RS binary to NetCDF conversion.
  - DC binary to NetCDF conversion.
  - `Measurement` behavior.
  - lidar config/search helpers.
  - SNR helper.
  - file manager helper.
  - ALHAMBRA quicklook generation.
  - ALHAMBRA basic preprocessing.
- Basic preprocessing coverage includes:
  - `apply_bg=True`
  - `apply_bz=True`
  - `apply_dc=True`
  - `apply_dt=True`
  - `apply_bin=True`
  - `apply_sm=True` with `smooth_mode="gaussian"`
- Added lightweight contract tests for synthetic signal generation:
  - `generate_particle_properties`
  - 1D elastic synthetic signals.
  - 1D elastic/Raman synthetic signals.
  - depolarization synthetic signals.
  - 2D elastic synthetic signals.
  - 2D Raman synthetic signals.
- Added a synthetic quicklook smoke test using `synthetic_signals_2D` and
  `quicklook_xarray`.
- Added synthetic Raman and LPDR quicklook smoke tests.
- Made quicklook color normalization ignore NaNs for `"auto"` and `"limits"`
  scale bounds.
- Fixed `bin_rescale()` to coarsen each `DataArray` instead of indexing a
  `DatasetCoarsen`.
- Fixed root package imports after removing `quality_assurance`.
- Fixed `add_height()` when zenith angle is zero and added unit coverage for
  constant and time-varying zenith angles.
- Added packaging config tests to keep coordination files out of sdist/wheel.
- Added synthetic edge-case tests for Raman activation and current
  `force_zero_aer_after_bin` behavior.

## In Progress

- None.

## Next Tasks

1. Add a migrated `apply_ov=True` preprocessing test.
2. Add a migrated `gluing_products=True` preprocessing test.
3. Decide whether `retrieval` remains in the package after overlap migration.
4. Review package-data rules for YAML, TOML, and assets.
5. Run the migrated test suite on a machine with enough disk space.
6. Review external dependencies in `pyproject.toml` and remove unused ones.
7. Decide whether `utils` and `general_utils` should both remain or whether one
   can become a compatibility layer.

## Known Risks

- Long test runs can fill disk because RAW fixtures are unzipped and converted
  to large NetCDF files.
- `apply_ov=True` may require a generated or fixture overlap file.
- `gluing_products=True` may expose assumptions inherited from old fixed-product
  tests.
- Synthetic signal annotations and implementation contracts do not always
  match. Current tests capture the implementation behavior before refactoring.
- Synthetic quicklooks currently adapt the generated signal to the plotting
  contract by assigning datetime coordinates and renaming it to `signal_*`.
- These coordination files are tracked for development only and must not be
  included in PyPI distributions.

## Commit Landmarks

- `3c95e48 Build standalone lidarpy package`
- `a92910d Add migrated lidar workflow tests`
