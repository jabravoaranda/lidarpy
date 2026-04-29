from __future__ import annotations

import numpy as np
import xarray as xr

from lidarpy.preprocessing import preprocess


def _has_finite_values(data_array: xr.DataArray) -> bool:
    values = np.asarray(data_array.values)
    return np.isfinite(values).any() and not np.all(np.isnan(values))


def test_preprocess_alhambra_basic_from_generated_netcdf(alhambra_rs_nc):
    dataset = preprocess(
        alhambra_rs_nc,
        channels=["1064fta"],
        crop_ranges=(0.0, 15000.0),
        apply_dc=False,
        apply_dt=False,
        apply_bg=True,
        apply_bz=True,
        apply_ov=False,
        gluing_products=False,
        apply_sm=False,
    )

    try:
        assert isinstance(dataset, xr.Dataset)
        assert {"time", "range"}.issubset(dataset.dims)
        assert dataset.sizes["time"] > 0
        assert dataset.sizes["range"] > 0

        assert dataset.attrs["system"] == "ALHAMBRA"
        assert dataset.attrs["dc_corrected"] == "False"
        assert dataset.attrs["dt_corrected"] == "False"
        assert dataset.attrs["bg_corrected"] == "True"
        assert dataset.attrs["bz_corrected"] == "True"
        assert dataset.attrs["ov_corrected"] == "False"

        assert float(dataset["range"].min()) >= 0.0
        assert float(dataset["range"].max()) <= 15000.0

        # TODO: Riesgo detectado: add_height() no añade height en este caso
        # porque el ángulo cenital parece ser cero y la lógica actual solo entra
        # si zenithal_angle.values.all() es verdadero. No lo he corregido todavía
        # porque este paso era para preprocessing básico y no quería mezclar
        # cambio funcional de producción.
        assert "1064fta" in {str(channel) for channel in dataset.channel.values}
        assert "signal_1064fta" in dataset
        assert dataset["signal_1064fta"].dims == ("time", "range")
        assert _has_finite_values(dataset["signal_1064fta"])
    finally:
        dataset.close()


def test_preprocess_alhambra_with_dark_current_and_dead_time(alhambra_rs_dc_nc):
    rs_path, dc_path = alhambra_rs_dc_nc
    assert rs_path.parent == dc_path.parent

    dataset = preprocess(
        rs_path,
        channels=["532fta", "532ftp"],
        crop_ranges=(0.0, 15000.0),
        apply_dc=True,
        apply_dt=True,
        apply_bg=True,
        apply_bz=True,
        apply_ov=False,
        gluing_products=False,
        apply_sm=False,
    )

    try:
        assert isinstance(dataset, xr.Dataset)
        assert {"time", "range"}.issubset(dataset.dims)
        assert dataset.attrs["system"] == "ALHAMBRA"
        assert dataset.attrs["dc_corrected"] == "True"
        assert dataset.attrs["dt_corrected"] == "True"
        assert dataset.attrs["bg_corrected"] == "True"
        assert dataset.attrs["bz_corrected"] == "True"
        assert dataset.attrs["ov_corrected"] == "False"

        available_channels = {str(channel) for channel in dataset.channel.values}
        assert {"532fta", "532ftp"}.issubset(available_channels)

        for channel in ("532fta", "532ftp"):
            signal_name = f"signal_{channel}"
            assert signal_name in dataset
            assert dataset[signal_name].dims == ("time", "range")
            assert _has_finite_values(dataset[signal_name])
    finally:
        dataset.close()


def test_preprocess_alhambra_with_range_binning(alhambra_rs_nc):
    with xr.open_dataset(alhambra_rs_nc) as raw_dataset:
        raw_range_size = raw_dataset.sizes["range"]

    dataset = preprocess(
        alhambra_rs_nc,
        channels=["1064fta"],
        crop_ranges=(0.0, 15000.0),
        apply_bin=True,
        bin_factor=4,
        apply_dc=False,
        apply_dt=False,
        apply_bg=True,
        apply_bz=True,
        apply_ov=False,
        gluing_products=False,
        apply_sm=False,
    )

    try:
        assert isinstance(dataset, xr.Dataset)
        assert dataset.attrs["rebin_factor"] == 4
        assert dataset.attrs["range_resolution_m"] == 15.0
        assert dataset.sizes["range"] < raw_range_size

        assert "signal_1064fta" in dataset
        assert dataset["signal_1064fta"].dims == ("time", "range")
        assert _has_finite_values(dataset["signal_1064fta"])
    finally:
        dataset.close()


def test_preprocess_alhambra_with_gaussian_smoothing(alhambra_rs_nc):
    dataset = preprocess(
        alhambra_rs_nc,
        channels=["1064fta"],
        crop_ranges=(0.0, 15000.0),
        apply_dc=False,
        apply_dt=False,
        apply_bg=True,
        apply_bz=True,
        apply_ov=False,
        gluing_products=False,
        apply_sm=True,
        smooth_mode="gaussian",
        gaussian={"sigma": 2.0},
    )

    try:
        assert isinstance(dataset, xr.Dataset)
        assert {"time", "range"}.issubset(dataset.dims)
        assert dataset.attrs["dc_corrected"] == "False"
        assert dataset.attrs["dt_corrected"] == "False"
        assert dataset.attrs["bg_corrected"] == "True"
        assert dataset.attrs["bz_corrected"] == "True"
        assert dataset.attrs["ov_corrected"] == "False"

        assert "signal_1064fta" in dataset
        assert dataset["signal_1064fta"].dims == ("time", "range")
        assert _has_finite_values(dataset["signal_1064fta"])
    finally:
        dataset.close()
