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
- Fixed `bin_rescale()` to coarsen each `DataArray` instead of indexing a
  `DatasetCoarsen`.
- Fixed root package imports after removing `quality_assurance`.

## In Progress

- None.

## Next Tasks

1. Add a migrated `apply_ov=True` preprocessing test.
2. Add a migrated `gluing_products=True` preprocessing test.
3. Fix `add_height()` when zenith angle is zero.
4. Decide whether `retrieval` remains in the package after overlap migration.
5. Review package-data rules for YAML, TOML, and assets.
6. Run the migrated test suite on a machine with enough disk space.
7. Review external dependencies in `pyproject.toml` and remove unused ones.
8. Decide whether `utils` and `general_utils` should both remain or whether one
   can become a compatibility layer.

## Known Risks

- Long test runs can fill disk because RAW fixtures are unzipped and converted
  to large NetCDF files.
- `add_height()` currently skips height creation when zenith angle is zero
  because it checks `zenithal_angle.values.all()`.
- `apply_ov=True` may require a generated or fixture overlap file.
- `gluing_products=True` may expose assumptions inherited from old fixed-product
  tests.
- Synthetic signal annotations and implementation contracts do not always
  match. Current tests capture the implementation behavior before refactoring.
- These coordination files are tracked for development only and must not be
  included in PyPI distributions.

## Commit Landmarks

- `3c95e48 Build standalone lidarpy package`
- `a92910d Add migrated lidar workflow tests`
