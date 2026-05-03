import numpy as np
import scipy as sp
import numba


def normalize(x):  # , exclude_inf=True):
    """Normalize data in a 1-D array: [0, 1]."""
    n = np.zeros(len(x)) * np.nan
    try:
        idx_fin = np.logical_and(x != -np.inf, x != np.inf)
        x0 = x[idx_fin]
        n[idx_fin] = (x0 - np.nanmin(x0)) / (np.nanmax(x0) - np.nanmin(x0))
    except Exception as e:
        print("ERROR in normalize. %s" % str(e))

    return n


def interp_nan(y, last_value=0):
    """Interpolate NaN values in a 1-D array."""
    if np.isnan(y[-1]):
        y[-1] = last_value
    if np.isnan(y[0]):
        y[0] = last_value

    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]
    f = sp.interpolate.interp1d(x(~nans), y[~nans], kind="cubic")
    y[nans] = f(x(nans))

    return y


def residuals(meas, pred):
    """Residual cost: mean squared normalized residual."""
    n = len(meas)
    try:
        sigma = np.nanstd(meas)
        j_value = np.nansum(((meas - pred) / sigma) ** 2) / n
    except Exception as e:
        print("ERROR. In cost_function %s" % str(e))
        j_value = np.nan
    return j_value


def unique(array):
    """Get unique non-NaN values of a 1-D array."""
    try:
        unq = np.unique(array[~np.isnan(array)])
    except Exception as e:
        unq = np.nan
        print("Error: getting unique values of an array. %s" % str(e))

    return unq


def find_nearest_1d(array, value):  # TODO: Isn't this the same as np.searchsorted?
    """Find nearest value in a 1-D array."""
    array = np.asarray(array)
    if np.logical_and(~np.isnan(value), ~np.isnan(array).all()):
        idx = (np.abs(array - value)).argmin()
        nearest = array[idx]
    else:
        idx = np.nan
        nearest = np.nan

    return idx, nearest


@numba.njit(parallel=True)
def windowed_proportional(arr1: np.ndarray, arr2: np.ndarray, /, *, w_size: int):
    """
    Compute windowed proportional factors and mean absolute proportional
    deviations between two matrices.
    """
    assert arr1.shape == arr2.shape, "Matrices shape must match"

    range_shape = arr1.shape[1] - (w_size - 1)
    proportional = np.full((arr1.shape[0], range_shape), np.nan)
    factor = np.full((arr1.shape[0], range_shape), np.nan)

    for t_idx in numba.prange(arr1.shape[0]):
        w1 = rolling(arr1[t_idx], w_size)
        w2 = rolling(arr2[t_idx], w_size)

        for idx in numba.prange(range_shape):
            _w1 = w1[idx]
            _w2 = w2[idx]

            ratio = np.mean(_w2 / _w1)
            adj = _w1 * ratio
            factor[t_idx][idx] = ratio
            proportional[t_idx][idx] = (np.abs(adj - _w2) / _w2).mean()

    return factor, proportional


@numba.njit()
def rolling(a, window):
    """Apply a rolling window to a 1-D numpy array."""
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@numba.njit
def correlate_vector_to_matrix(vector, matrix):
    """Correlate a vector to each column of a matrix."""
    corr = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        corr[i] = np.corrcoef(vector, matrix[:, i])[0, 1]
    return corr


def rolling_window_test(a, window):
    """Create a rolling window view of the input array."""
    shp = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shp, strides=strides)
