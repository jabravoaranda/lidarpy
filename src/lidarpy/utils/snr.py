import numpy as np
import xarray as xr

def snr_analog(signal, background, dim="time"):
    """
    Calculate SNR for analog signals (Baars et al. 2016).
    
    Parameters:
        signal: array-like or xarray.DataArray
        background: array-like or xarray.DataArray (same dims as signal)
        dim: dimension over which to average the background (default: "time")
    
    Returns:
        snr: array-like of SNR values
    """
    if isinstance(signal, xr.DataArray) and isinstance(background, xr.DataArray):
        background_mean = background.mean(dim=dim)
        background_std = background.std(dim=dim)
        snr = (signal - background_mean) / background_std
    else:
        background_mean = np.mean(background, axis=0)
        background_std = np.std(background, axis=0)
        snr = (signal - background_mean) / background_std
    return snr


def snr_photocounting(signal, background, dark_noise_var=0):
    """
    Calculate SNR for photon counting signals (Baars et al. 2016 Eq A10).
    
    Parameters:
        signal: array-like of counts (S)
        background: array-like of background counts (B)
        dark_noise_var: variance of dark noise (optional, default 0)
    
    Returns:
        snr: array-like of SNR values
    """
    snr_photocounting = signal / np.sqrt(signal + background + dark_noise_var)
    return snr_photocounting

