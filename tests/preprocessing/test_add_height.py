from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from lidarpy.preprocessing.lidar_preprocessing import add_height


def _dataset_with_zenithal_angle(angles: list[float]) -> xr.Dataset:
    ranges = np.array([0.0, 1000.0, 2000.0])
    time = np.arange(len(angles))
    return xr.Dataset(
        data_vars={
            "zenithal_angle": ("time", np.array(angles, dtype=float)),
        },
        coords={
            "range": ranges,
            "time": time,
        },
    )


def test_add_height_uses_range_when_zenithal_angle_is_zero():
    dataset = _dataset_with_zenithal_angle([0.0, 0.0])

    result = add_height(dataset)

    assert "height" in result
    assert np.allclose(result["height"].values, result["range"].values)


def test_add_height_projects_range_for_constant_nonzero_zenithal_angle():
    dataset = _dataset_with_zenithal_angle([30.0, 30.0])

    result = add_height(dataset)

    assert "height" in result
    assert np.allclose(result["height"].values, result["range"].values * np.cos(np.deg2rad(30.0)))


def test_add_height_rejects_time_varying_zenithal_angle():
    dataset = _dataset_with_zenithal_angle([0.0, 1.0])

    with pytest.raises(RuntimeError, match="Zenithal angle is not constant"):
        add_height(dataset)
