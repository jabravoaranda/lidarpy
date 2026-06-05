import numpy as np
import pytest
import xarray as xr

from lidarpy.retrieval.ablh import detect_ablh, prepare_ablh_input
from lidarpy.retrieval.synthetic.generator import synthetic_signals_2D
from lidarpy.utils.utils import signal_to_rcs


def _synthetic_2d_case() -> tuple[xr.DataArray, np.ndarray]:
    ranges = np.arange(30.0, 4000.0, 30.0)
    time = np.arange(0.0, 240.0, 5.0)
    base_ablh = 1400.0
    amplitude = 250.0

    signal, _ = synthetic_signals_2D(
        ranges,
        time,
        apply_overlap=False,
        number_of_initial_nan_values=3,
        sigmoid_edge=(base_ablh, 3000.0),
        apply_entrainment_zone=True,
        amplitude_entrainment_zone=(amplitude, amplitude),
        period_entrainment_zone=60.0,
        variable_intensity=True,
    )
    rcs = signal_to_rcs(signal, signal["range"]).rename("synthetic_rcs")
    truth = base_ablh + amplitude * np.cos(2 * np.pi / 60.0 * time)
    return rcs, truth


@pytest.mark.parametrize(
    ("method", "threshold", "max_median_error"),
    [
        ("wct", 0.02, 80.0),
        ("temporal_variance", 1e-5, 80.0),
    ],
)
def test_ablh_methods_detect_layer_on_synthetic_2d_signal(
    method,
    threshold,
    max_median_error,
):
    rcs, truth = _synthetic_2d_case()

    result = detect_ablh(
        rcs,
        method=method,
        min_range=700.0,
        max_range=2200.0,
        threshold=threshold,
        time_window_minutes=10.0,
        wct_width=300.0,
    )

    assert result["ablh"].dims == ("time",)
    assert result.attrs["method"] == method
    finite = np.isfinite(result["ablh"].values)
    assert finite.mean() > 0.9
    assert np.nanmedian(np.abs(result["ablh"].values - truth)) < max_median_error


def test_ablh_input_adapters_read_lidarpy_and_cloudnet_contracts():
    rcs, _ = _synthetic_2d_case()
    ranges = rcs["range"].values
    lidar_signal = (rcs / rcs["range"] ** 2).rename("signal_532fta")
    lidarpy_dataset = xr.Dataset({"signal_532fta": lidar_signal})
    cloudnet_dataset = xr.Dataset(
        {
            "beta_smooth": rcs.rename("beta_smooth"),
            "height": (("range",), ranges + 680.0),
        }
    )

    (
        lidar_signal_prepared,
        lidar_ranges,
        lidar_heights,
        lidar_source,
    ) = prepare_ablh_input(
        lidarpy_dataset,
        input_kind="lidarpy",
        channel="532fta",
    )
    (
        cloudnet_signal,
        cloudnet_ranges,
        cloudnet_heights,
        cloudnet_source,
    ) = prepare_ablh_input(
        cloudnet_dataset,
        input_kind="cloudnet",
        variable="beta_smooth",
    )

    assert lidar_signal_prepared.dims == ("time", "range")
    assert cloudnet_signal.dims == ("time", "range")
    assert lidar_source == "signal_532fta"
    assert cloudnet_source == "beta_smooth"
    np.testing.assert_allclose(lidar_ranges, ranges)
    np.testing.assert_allclose(cloudnet_ranges, ranges)
    np.testing.assert_allclose(lidar_signal_prepared.values, rcs.values)
    np.testing.assert_allclose(cloudnet_signal.values, rcs.values)
    assert lidar_heights is None
    np.testing.assert_allclose(cloudnet_heights, ranges + 680.0)


def test_cloudnet_ablh_height_is_reported_above_sea_level():
    rcs, _ = _synthetic_2d_case()
    ranges = rcs["range"].values
    cloudnet_dataset = xr.Dataset(
        {
            "beta_smooth": rcs.rename("beta_smooth"),
            "height": (("range",), ranges + 680.0),
        }
    )

    result = detect_ablh(
        cloudnet_dataset,
        input_kind="cloudnet",
        variable="beta_smooth",
        method="wct",
        min_range=700.0,
        max_range=2200.0,
        threshold=0.02,
    )

    finite = np.isfinite(result["ablh_range"].values)
    assert finite.any()
    np.testing.assert_allclose(
        result["ablh_height"].values[finite] - result["ablh_range"].values[finite],
        680.0,
        atol=1e-6,
    )
