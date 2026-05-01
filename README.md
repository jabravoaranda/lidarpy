# lidarpy

`lidarpy` is a standalone package for lidar data processing, migrated from the
former `gfatpy.lidar` submodule.

Current migrated workflow:

1. Convert Licel binary measurements to NetCDF.
2. Apply basic lidar preprocessing.
3. Generate quicklooks.
4. Generate synthetic elastic, Raman, and depolarization signals for validation.
5. Validate retrieval algorithms against synthetic truth where the physical
   assumptions are explicit.

See [docs/migrated_lidar_flow.md](docs/migrated_lidar_flow.md) for the current
migration flow, tested guarantees, and known boundaries.

The public module reference index is in [docs/references.md](docs/references.md).
