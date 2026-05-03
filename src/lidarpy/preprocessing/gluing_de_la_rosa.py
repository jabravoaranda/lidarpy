import numpy as np
import xarray as xr
from numpy.polynomial.polynomial import polyfit, polyval

from lidarpy.general_utils.fitting import windowed_corrcoefs


def gluing(
    signal_an: xr.DataArray,
    signal_pc: xr.DataArray,
    range_min: float = 1200,
    range_max: float = 5000,
    window_meters: float = 1200,
) -> np.ndarray:
    sel_signal_an = signal_an.sel(
        range=slice(range_min - window_meters / 2, range_max + window_meters / 2)
    )
    sel_signal_pc = signal_pc.sel(
        range=slice(range_min - window_meters / 2, range_max + window_meters / 2)
    )

    ranges = sel_signal_an.range.values
    window_bins = int(window_meters // (ranges[1] - ranges[0]))

    rcs_an = (sel_signal_an * ranges**2).values
    rcs_pc = (sel_signal_pc * ranges**2).values

    corr_coeffs = windowed_corrcoefs(rcs_an, rcs_pc, window_bins)
    gluing_window_bins = np.apply_along_axis(lambda x: x.argmax(), 1, corr_coeffs)
    gluing_bins = gluing_window_bins + int(range_min // (ranges[1] - ranges[0])) + 1

    rcs_an_whole = (signal_an * signal_an.range**2).values
    rcs_pc_whole = (signal_pc * signal_pc.range**2).values
    rcs_gl_whole = rcs_pc_whole.copy()

    for idx in np.arange(rcs_an_whole.shape[0]):
        hw = window_bins // 2  # Half window
        gl_bin = gluing_bins[idx]

        win_an = rcs_an_whole[idx, gl_bin - hw : gl_bin + hw]
        win_pc = rcs_pc_whole[idx, gl_bin - hw : gl_bin + hw]

        reg = polyfit(win_an, win_pc, 1)

        cov = np.corrcoef(win_an, win_pc)[1, 0]
        # print(f"Covariance: {cov}")
        # print(f"Original cov: {corr_coeffs[idx].max()}")
        # print(f"Adjusting analog at profile {idx} to height: {gl_bin * 7.5}")
        if cov > 0.70:
            rcs_gl_whole[idx, :gl_bin] = polyval(rcs_an_whole[idx, :gl_bin], reg)
        else:
            rcs_gl_whole[idx] = np.zeros_like(rcs_gl_whole[idx])

    return rcs_gl_whole / signal_pc.range.values**2
