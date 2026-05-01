# References

This page is the current reference index for the migrated `lidarpy` package.
It is intentionally module-oriented so it can be kept stable while the
standalone package is still being cleaned up.

## Main Processing Modules

- `lidarpy.nc_convert.measurement`
  - Binary Licel measurement discovery, parsing, and NetCDF writing.
- `lidarpy.preprocessing.lidar_preprocessing`
  - Main preprocessing entry point and workflow orchestration.
- `lidarpy.plot.quicklook`
  - Quicklook generation for lidar products and synthetic signals.

## Synthetic Signals

- `lidarpy.retrieval.synthetic.generator`
  - `generate_particle_properties`
  - `synthetic_signals`
  - `synthetic_signals_despo`
  - `synthetic_signals_2D`
  - `synthetic_raman_signals_2D`

The synthetic generator is used as the package-internal reference for physics
checks and retrieval tests. See [migrated_lidar_flow.md](migrated_lidar_flow.md)
for the tested physical contracts.

## Retrieval

- `lidarpy.retrieval.klett`
  - `klett_rcs`
  - `quasi_beta`
  - `iterative_beta`
  - `iterative_beta_forward`
  - `find_lidar_ratio`
- `lidarpy.retrieval.raman`
  - `retrieve_extinction`
  - `retrieve_backscatter`
- `lidarpy.retrieval.overlap`
  - overlap-related retrieval helpers.
- `lidarpy.retrieval.calibration`
  - calibration-factor retrieval helpers.

Important current boundary: `quasi_beta` is a single-iteration approximation,
not an exact inversion. `iterative_beta_forward` needs
`initial_particle_optical_depth` when starting above the first range bin.

## Atmosphere

- `lidarpy.atmo.atmo`
  - standard atmosphere, transmittance, attenuated backscatter, and related
    atmospheric helpers.
- `lidarpy.atmo.rayleigh`
  - molecular extinction/backscatter and molecular properties.
- `lidarpy.atmo.ecmwf`
  - ECMWF profile access and interpolation helpers.
- `lidarpy.atmo.solar`
  - solar-position helpers.

## Lidar Utilities

- `lidarpy.utils.file_manager`
  - lidar filename, path, channel, and measurement search helpers.
- `lidarpy.utils.snr`
  - analog and photon-counting signal-to-noise helpers.
- `lidarpy.utils.utils`
  - lidar-specific signal utilities such as range correction and overlap
    refilling.
- `lidarpy.utils.types`
  - shared lidar typing definitions.

Generic utilities live under `lidarpy.general_utils`. Duplicated wrappers under
`lidarpy.utils` were removed during migration; only lidar-specific helpers
should remain there.

## Generating API HTML Locally

The project includes `pdoc` in the `docs` optional dependency group. After
installing documentation dependencies, generate API HTML with:

```powershell
.\.venv311\Scripts\python -m pdoc lidarpy -o site\api
```

The generated `site/` directory is a local build artifact and is not part of the
runtime wheel.
