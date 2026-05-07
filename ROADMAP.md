# ROADMAP

## Current Status

Migration closed. The lidar processing workflow has been migrated from
`gfatpy/lidar` to the standalone PyPI distribution `atmolidarpy`; the importable
Python package remains `lidarpy`. Remaining work is maintenance and release
hardening. The `atmolidarpy` project is published on PyPI and Trusted
Publishing is configured for releases from this repository.

## Done

- Created standalone `src/lidarpy` package.
- Added `pyproject.toml` for `lidarpy`.
- Renamed the PyPI distribution to `atmolidarpy` because `lidarpy` is already
  taken on PyPI; imports remain `import lidarpy`.
- Configured pytest with `--basetemp=.pytest_tmp`.
- Added `.gitignore` entries for local runtimes, pytest temp folders, unzip
  folders, and generated artifacts.
- Kept only the current minimal ALHAMBRA RAW fixtures:
  - `tests/data/RAW/alhambra/2023/08/30/RS_20230830_0315.zip`
  - `tests/data/RAW/alhambra/2023/08/30/DC_20230830_0315.zip`
- Removed legacy modules/tests outside the current migration focus:
  `quality_assurance`, `mulhacen`, `synthetic`, and old retrieval tests.
- Restored `scc` and `depolarization` as crucial standalone package modules
  after discovering they had been incorrectly removed from the migration scope.
- Added SCC smoke coverage for restored standalone modules: core imports,
  packaged campaign/config/plot resources, Licel filename date parsing,
  temperature/pressure header parsing, and local file movement.
- Added offline SCC access contract coverage with fake HTTP sessions: SCC URL
  construction, YAML settings loading, API measurement parsing/status helpers,
  measurement-id sequence selection, streamed ZIP downloads, and upload response
  parsing without contacting a real SCC server. Validated on 2026-05-06 with
  `$env:PYTHONPATH='src'; $env:MPLBACKEND='Agg'; .\.venv\Scripts\python -m pytest tests\scc -q`
  passing with `11 passed in 0.40s`.
- Expanded user-facing documentation into an operational guide covering package
  scope, installation, first validation, preprocessing lifecycle, examples,
  local/CI/Docker/production boundaries, troubleshooting, PyPI release and
  rollback practices.
- Kept `retrieval` as part of the standalone package scope; retrieval was in
  the migration roadmap and is covered by synthetic Klett, iterative elastic,
  Raman extinction, Raman backscatter, and overlap-related tests/docs.
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
- Added retrieval smoke tests that feed synthetic elastic/Raman signals into
  Klett, iterative elastic, Raman extinction, and Raman backscatter retrievals.
- Added numerical retrieval tests against synthetic truth for Klett
  backscatter, Raman extinction, and Raman backscatter in a useful range
  outside near-field/boundary regions.
- Added numerical synthetic-truth coverage for bottom-up
  `iterative_beta_forward`; `quasi_beta` remains a single-iteration
  approximation and is tested as such, not as an exact inversion.
- Added `initial_particle_optical_depth` to `iterative_beta_forward` so
  bottom-up retrieval can start above the first range bin when the particle AOD
  below `start_height` is known.
- Added physical-property tests for synthetic signals covering lidar-ratio
  consistency, Angstrom wavelength scaling, elastic/Raman lidar equations,
  bounded monotonic transmittance, and depolarization component ratios.
- Refined synthetic Raman activation so scalar `wavelengths` can use an
  explicit `wavelength_raman`, while the default remains elastic-only.
- Fixed 2D synthetic `force_zero_aer_after_bin` to target the range axis.
- Removed unused notebook/UI/heavy runtime dependencies from `pyproject.toml`
  and added direct `psutil`/`pytz` dependencies for imported modules.
- Made `general_utils` the canonical location for generic utilities and
  removed duplicated `utils` wrappers; `utils` now keeps only lidar-specific
  helpers.
- Validated the package build artifacts with `uv build`: `AGENTS.md` and
  `ROADMAP.md` are absent from sdist/wheel, while lidar package data such as
  assets, YAML, and TOML config files are present.
- Added build-artifact packaging tests that inspect real wheel/sdist contents
  when `uv build` can run in the local environment.
- Removed clear unused imports from active modules/tests and dropped unused
  runtime dependency `pytz`.
- Added minimal migrated-flow documentation and retrieval edge-case tests for
  out-of-range reference intervals.
- Removed the transitional migrated-flow document from public docs; migration
  status remains in `ROADMAP.md`, while `docs/` keeps user-facing references.
- Added a static GitHub Pages documentation site under `docs/` with overview,
  getting started, workflow, examples, references, and contributing pages.
- Configured the `docs` GitHub Actions workflow to build `site/`, generate
  pdoc API HTML, and publish the generated Pages artifact.
- Added documentation figures generated from synthetic elastic, Raman, and LPDR
  quicklooks so users can inspect expected visual outputs.
- Added a documentation figure comparing synthetic aerosol truth with Klett,
  iterative elastic, Raman extinction, and Raman backscatter retrieval outputs.
- Extended the retrieval-validation documentation figure with LPDR derived from
  polarized synthetic components.
- Integrated documentation figure generation into `scripts/build_docs.py` so
  GitHub Actions regenerates figures before publishing the Pages artifact.
- Added and expanded a scientific-references section to the public docs linking
  the migrated retrieval, molecular-atmosphere, depolarization, overlap, and
  preprocessing routines to their source papers.
- Added `.github/workflows/publish-package.yml` to build distributions, check
  them with Twine, and publish to PyPI through Trusted Publishing.
- Added `.github/workflows/tests.yml` so pushes and pull requests against
  `main` and `develop` run the pytest suite on GitHub Actions.
- Published `atmolidarpy` on PyPI through Trusted Publishing:
  - `0.1.0` on 2026-05-04.
  - `0.1.1` on 2026-05-04.
  The latest release artifacts were uploaded from `publish-package.yml` at tag
  `v0.1.1`.
- Updated package metadata URLs to point to the standalone `lidarpy`
  repository, documentation site, and issue tracker.
- Removed active `pdb.set_trace()` calls and obvious diagnostic print output
  from active utility/preprocessing paths.
- Added pytest markers for documentation, packaging, integration, and slow
  tests so the full suite can be split cleanly during development without
  changing the default CI behavior.
- Replaced remaining noisy `print()` output in active retrieval/reference-range
  paths with `loguru` logging or existing debug gates.
- Added a migrated `apply_ov=True` preprocessing test using a generated
  per-channel overlap NetCDF profile.
- Added a migrated `gluing_products=True` preprocessing test for ALHAMBRA
  near-field 532 nm analog/photon-counting products.
- Confirmed that `retrieval` remains in `lidarpy` as part of the standalone
  package migration scope.
- Reviewed the remaining `utils` modules. `file_manager`, `types`, `snr`, and
  `get_reference_range` are lidar-specific; the remaining helpers in
  `utils.py` are tied to lidar/Licel naming, signal/RCS handling, overlap, or
  retrieval/synthetic workflows, so no additional move to `general_utils` is
  needed for migration closure.
- Split general utility helpers into clearer `general_utils.fitting`,
  `general_utils.numerics`, and `general_utils.dates` modules. Removed the old
  `optimized` and `utils` modules, moved `check_dir` into `general_utils.io`,
  and deleted unused calendar/legacy miscellaneous helpers.
- Added coverage for derived `apply_ov=True` overlap retrieval using the
  ALHAMBRA full-field/near-field channel pair `1064fta`/`1064nta`, so migrated
  preprocessing now covers both generated overlap files and overlap derived
  from channels. Validated on 2026-05-05 with the focused overlap tests:
  `tests/preprocessing/test_preprocessing_basic_alh.py::test_preprocess_alhambra_with_overlap_file`
  and
  `tests/preprocessing/test_preprocessing_basic_alh.py::test_preprocess_alhambra_with_derived_overlap`
  passed with `2 passed in 122.29s`.
- Validated the migrated test suite on 2026-05-04:
  `$env:PYTHONPATH='src'; $env:MPLBACKEND='Agg'; .\.venv\Scripts\python -m pytest tests -q`
  passed with `78 passed in 499.29s`.
- Added initial SCC utility unit tests covering Licel filename date parsing,
  header temperature/pressure extraction, campaign config loading/building, SCC
  parameter module import, and local SCC output directory checks.
- Moved reusable Licel helpers for filename datetime parsing and header
  temperature/pressure extraction from `scc.utils` to the lidar-specific
  `utils.utils` module, keeping SCC compatibility imports for legacy callers.

## In Progress

- He movido funciones de "fechas" de lidar.utils.utils a general_utils.dates, hay que revisar dependencias e importaciones. 

- Generar un método en lidarpy que identifica qué tipo de configuración SCC (ver lista) tiene la medida utilizando un fichero binario:
- 781: far y near de noche -> si hay canales f y n con raman
- 783: far y near de día -> si hay canales f y n sin raman
- 1038: near de noche -> si hay canales sólo n con raman
- 1040: near de día -> si hay canales sólo n sin raman
- 1039: far de noche -> si hay canales sólo f con raman
- 1041: far de día -> si hay canales sólo f sin raman
- Hacer un test de esto. 

- 

## Next Tasks

- None.

## Post-Migration Maintenance

- Long test runs can fill disk because RAW fixtures are unzipped and converted
  to large NetCDF files.
- Synthetic quicklooks currently adapt the generated signal to the plotting
  contract by assigning datetime coordinates and renaming it to `signal_*`.
- `iterative_beta_forward` started above the first range bin requires a valid
  `initial_particle_optical_depth`; otherwise the retrieval is missing its
  lower-boundary particle transmittance.
- These coordination files are tracked for development only and must not be
  included in PyPI distributions.
- SCC tests currently cover local/import/package-data behavior and the SCC
  access client contract with mocked HTTP sessions. End-to-end SCC
  submission/download workflows still require SCC server access and credentials
  and should stay out of the default offline suite.

## Commit Landmarks

- `3c95e48 Build standalone lidarpy package`
- `a92910d Add migrated lidar workflow tests`
