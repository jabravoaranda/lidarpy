# Migrated Lidar Flow

This document summarizes the current standalone `lidarpy` workflow during the
migration from `gfatpy.lidar`.

## Processing Flow

The migrated chain currently covered by tests is:

1. Binary Licel RAW files to NetCDF:
   - RS measurements.
   - DC measurements.
   - ALHAMBRA fixture coverage.
2. Basic preprocessing:
   - background correction.
   - bin-zero handling.
   - dark-current correction.
   - dead-time correction.
   - binning.
   - Gaussian smoothing.
3. Quicklooks:
   - ALHAMBRA measurement quicklook.
   - synthetic elastic quicklook.
   - synthetic Raman quicklook.
   - synthetic LPDR quicklook.
4. Synthetic signals:
   - 1D elastic.
   - 1D elastic plus Raman.
   - 1D depolarization.
   - 2D elastic.
   - 2D Raman.
5. Retrieval validation:
   - Klett backscatter.
   - Raman extinction.
   - Raman backscatter.
   - `quasi_beta` as a single-iteration approximation.
   - `iterative_beta_forward` as a bottom-up retrieval with an explicit lower
     boundary condition when starting above the first range bin.

## Synthetic Truth Guarantees

Synthetic tests currently verify these physical relationships:

- particle extinction follows `alpha = lidar_ratio * beta`.
- Angstrom scaling is preserved between wavelengths.
- elastic and Raman generated signals match their lidar equations when overlap
  is disabled.
- transmittance is bounded between 0 and 1 and decreases with range.
- depolarization component ratios are internally consistent.
- 2D products preserve the same lidar-ratio and wavelength-scaling contracts.

## Retrieval Boundaries

Retrieval tests compare against synthetic truth only where assumptions are
explicit:

- the useful comparison range excludes near-field and boundary regions.
- Klett requires a physically meaningful reference and optional aerosol
  backscatter at reference.
- Raman backscatter uses the same reference interval and particle Angstrom
  exponent used to generate the synthetic signal.
- `quasi_beta` is intentionally not treated as an exact inversion. It is a
  one-step approximation.
- `iterative_beta_forward` can start above the first range bin only if
  `initial_particle_optical_depth` is supplied. Without that boundary condition,
  the particle transmittance below `start_height` is unknown.

## Open Migration Items

These areas are intentionally left for separate migration steps:

- advanced preprocessing with overlap correction.
- gluing products if not already integrated from another branch.
- final decision on whether all retrieval modules remain in `lidarpy`.
- broader full-suite runs on a machine with enough disk space for generated
  NetCDF files.
