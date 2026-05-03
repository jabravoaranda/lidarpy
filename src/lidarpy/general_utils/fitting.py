from typing import Callable

import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats
from scipy.stats import anderson
import numba
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from lidarpy.general_utils.numerics import rolling, rolling_window_test


def linear_regression(x, y):
    """Fit y = a*x + b with scipy linear regression."""
    try:
        x = np.asarray(x)
        y = np.asarray(y)

        idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x_train = x[idx]
        y_train = y[idx]

        lr = stats.linregress(x_train, y_train)
        slope = float(lr.slope)  # type: ignore
        intercept = float(lr.intercept)  # type: ignore
        rvalue = float(lr.rvalue)  # type: ignore
    except Exception as e:
        print("ERROR. In linear_regression. %s" % str(e))
        slope = np.nan
        intercept = np.nan
        rvalue = np.nan

    return slope, intercept, rvalue


def linrest(x, y):
    n = len(y)
    dofreedom = n - 2
    z, _ = np.polyfit(x, y, 1, cov=True)
    p = np.poly1d(z)
    yp = p(x)
    slope = z[0]
    intercept = z[1]
    r2 = np.corrcoef(x, y)[0][1] ** 2
    residual_ss = np.sum((y - yp) ** 2)
    slope_pm = np.sqrt(residual_ss / (dofreedom * np.sum((x - np.mean(x)) ** 2)))
    intercept_pm = slope_pm * np.sqrt(np.sum(x**2) / n)

    return slope, slope_pm, intercept, intercept_pm, r2


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a linear model and return fit/residual diagnostics."""
    x_design = sm.add_constant(x)
    model = sm.OLS(y, x_design).fit()

    statistics = {}
    statistics["residuals"] = model.resid
    statistics["parameters"] = model.params
    statistics["standard_deviation_parameters"] = model.bse
    statistics["msre"] = np.sqrt(np.mean(model.resid**2))
    statistics["anderson"] = anderson(statistics["residuals"], dist="norm")
    statistics["durbin_watson"] = durbin_watson(statistics["residuals"])
    return statistics


def create_linear_interpolator(
    x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    def interpolator(_x):
        return np.interp(_x, x, y)

    return interpolator


def best_slope_fit(mat1: np.ndarray, mat2: np.ndarray, window: int) -> np.ndarray:
    """Find the window index where two profiles have the closest local slope."""
    assert mat1.shape == mat2.shape, "Matrices shape must match"
    assert isinstance(window, int), "Window argument must be an integer"

    x = np.arange(window)
    res = np.array([])

    for idx in range(mat1.shape[0]):
        windowed1 = rolling(mat1[idx], window)
        windowed2 = rolling(mat2[idx], window)

        slopes1 = polyfit(x, windowed1.T, 1)[0]
        slopes2 = polyfit(x, windowed2.T, 1)[0]
        chosen_group = np.argmin(np.abs(slopes1 - slopes2) / slopes1)
        res = np.hstack([res, chosen_group + np.floor(window / 2)])

    return res.astype(int)


@numba.njit(parallel=True)
def windowed_corrcoefs(arr1: np.ndarray, arr2: np.ndarray, w_size: int):
    """Compute windowed correlation coefficients between two 2-D arrays."""
    range_shape = arr1.shape[1] - (w_size - 1)
    corrcoefs = np.empty((arr1.shape[0], range_shape))
    for t_idx in numba.prange(arr1.shape[0]):
        w1 = rolling(arr1[t_idx], w_size)
        w2 = rolling(arr2[t_idx], w_size)
        for idx in numba.prange(range_shape):
            _w1 = w1[idx]
            _w2 = w2[idx]
            coeff = np.corrcoef(_w1, _w2)[1, 0]
            corrcoefs[t_idx][idx] = coeff
    return corrcoefs


def moving_linear_fit(x_array, y_array, window_size, **kwargs):
    """
    Perform a moving linear fit on the given data arrays.
    """
    xdata = rolling_window_test(x_array, window_size).T
    ydata = rolling_window_test(y_array, window_size).T

    d = np.nan * np.ones(len(x_array))
    slope = np.nan * np.ones(len(x_array))
    mrse = np.nan * np.ones(len(x_array))
    anderson_coef = np.nan * np.ones(len(x_array))
    std_slope = np.nan

    for idx in range(ydata.shape[1]):
        stats = linear_fit(xdata[:, idx], ydata[:, idx])
        slope[idx] = stats["parameters"][1]
        std_slope = stats["standard_deviation_parameters"][1]
        d[idx] = stats["durbin_watson"]
        mrse[idx] = stats["msre"]
        anderson_coef[idx] = stats["anderson"][0]
        if kwargs.get("debugger", False):
            ranges = [1100, 1300, 1600, 1900]
            if idx in ranges:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.plot(xdata[:, idx], ydata[:, idx], linewidth=0, marker="o", label=f"{ranges[idx]}")
                ax.plot(
                    xdata[:, idx],
                    np.polyval(np.flip(stats["parameters"]), xdata[:, idx]),
                    label="fit",
                )
                fig.savefig(f"test_dws_{idx}.png", dpi=300)
                plt.close(fig)

    return {
        "slope": slope,
        "std_slope": std_slope,
        "durbin_watson": d,
        "mrse": mrse,
        "anderson": anderson_coef,
    }
