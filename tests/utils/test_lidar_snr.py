import numpy as np
import xarray as xr
from math import isclose

from lidarpy.utils.snr import snr_analog, snr_photocounting


def _signal_dataarray(dataset: xr.Dataset) -> xr.DataArray:
    signal_vars = [name for name in dataset.data_vars if name.startswith("signal_")]
    assert signal_vars
    return dataset[signal_vars[0]]


def test_snr_analog_alhambra(alhambra_rs_nc):
    ds = xr.open_dataset(alhambra_rs_nc)
    # Use one channel as signal and background for demonstration
    signal = _signal_dataarray(ds)
    background = signal
    snr = snr_analog(signal, background, dim="time")
    assert isinstance(snr, xr.DataArray)
    assert snr.shape == signal.shape
    assert not np.isnan(snr.values).any()
    # Optional: check mean SNR is close to 0 (since signal == background)
    assert isclose(snr.values.mean(), 0.0, abs_tol=1e-6)
    ds.close()


def test_snr_photocounting_alhambra(alhambra_rs_nc):
    ds = xr.open_dataset(alhambra_rs_nc)
    signal = _signal_dataarray(ds)
    background = signal
    snr = snr_photocounting(signal.values, background.values)
    assert isinstance(snr, np.ndarray)
    assert snr.shape == signal.shape
    assert not np.isnan(snr).any()
    ds.close()
