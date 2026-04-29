import numpy as np
from numpy.polynomial.polynomial import polyfit

import numba

from lidarpy.general_utils.utils import linear_fit


def best_slope_fit(mat1: np.ndarray, mat2: np.ndarray, window: int) -> np.ndarray:
    """Best slope fit

    Args:
        mat1 (np.ndarray): Matrix 1
        mat2 (np.ndarray): Matrix 2
        window (int): Window size

    Returns:
        np.ndarray: indexes the best slope fit is found in axis=1. Results in axis=0.
    """

    assert mat1.shape == mat2.shape, "Matrices shape must match"
    assert isinstance(window, int), "Window argument must be an integer"

    x = np.arange(window)
    res = np.array([])

    # Iter in first dimension, equivalent to time in lidar case
    for idx in range(mat1.shape[0]):
        windowed1 = rolling(mat1[idx], window)
        windowed2 = rolling(mat2[idx], window)

        slopes1 = polyfit(x, windowed1.T, 1)[0]
        slopes2 = polyfit(x, windowed2.T, 1)[0]
        chosen_group = np.argmin(np.abs(slopes1 - slopes2) / slopes1)
        res = np.hstack([res, chosen_group + np.floor(window / 2)])  # Append chosen bin

    return res.astype(int)


@numba.njit(parallel=True)
def windowed_corrcoefs(arr1: np.ndarray, arr2: np.ndarray, w_size: int):
    """
    Compute windowed correlation coefficients between two 2D arrays using a sliding window approach.

    Parameters:
    arr1 (np.ndarray): First input array of shape (n_samples, n_features).
    arr2 (np.ndarray): Second input array of shape (n_samples, n_features).
    w_size (int): Size of the sliding window.

    Returns:
    np.ndarray: Array of shape (n_samples, n_windows) containing the correlation coefficients for each window.
    """
    range_shape = arr1.shape[1] - (w_size - 1)
    _corrcoefs = np.empty((arr1.shape[0], range_shape))
    for t_idx in numba.prange(arr1.shape[0]):
        w1 = rolling(arr1[t_idx], w_size)
        w2 = rolling(arr2[t_idx], w_size)
        for idx in numba.prange(range_shape):
            _w1 = w1[idx]
            _w2 = w2[idx]
            coeff = np.corrcoef(_w1, _w2)[1, 0]
            _corrcoefs[t_idx][idx] = coeff
    return _corrcoefs


@numba.njit(parallel=True)

def windowed_proportional(arr1: np.ndarray, arr2: np.ndarray, /, *, w_size: int):
    """
    Compute the windowed proportional factors and their deviations between two matrices.
    This function calculates the proportional factors and their deviations for each window
    of size `w_size` in the input matrices `arr1` and `arr2`. The function uses Numba's 
    Just-In-Time (JIT) compilation to optimize performance and parallel execution.
    Parameters:
    -----------
    arr1 : np.ndarray
        The first input matrix. Must have the same shape as `arr2`.
    arr2 : np.ndarray
        The second input matrix. Must have the same shape as `arr1`.
    w_size : int
        The size of the window to use for the rolling calculations.
    Returns:
    --------
    _factor : np.ndarray
        A matrix containing the proportional factors for each window.
    _proportional : np.ndarray
        A matrix containing the mean absolute proportional deviations for each window.
    Raises:
    -------
    AssertionError
        If the shapes of `arr1` and `arr2` do not match.
    """
    assert arr1.shape == arr2.shape, "Matrices shape must match"
    # assert isinstance(w_size, int), "Window argument must be an integer"

    range_shape = arr1.shape[1] - (w_size - 1)
    _proportional = np.full((arr1.shape[0], range_shape), np.nan)
    _factor = np.full((arr1.shape[0], range_shape), np.nan)

    for t_idx in numba.prange(arr1.shape[0]):
        w1 = rolling(arr1[t_idx], w_size)
        w2 = rolling(arr2[t_idx], w_size)

        for idx in numba.prange(range_shape):
            _w1 = w1[idx]
            _w2 = w2[idx]

            ratio = np.mean(_w2 / _w1)
            adj = _w1 * ratio
            _factor[t_idx][idx] = ratio
            _proportional[t_idx][idx] = (np.abs(adj - _w2) / _w2).mean()

    return _factor, _proportional


@numba.njit()
def rolling(a, window):
    """
    Apply a rolling window to a 1D numpy array.

    Parameters:
    a (numpy.ndarray): Input 1D array.
    window (int): Size of the rolling window.

    Returns:
    numpy.ndarray: A 2D array where each row is a windowed segment of the input array.

    Example:
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> rolling(a, 3)
    array([[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])
    """
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@numba.njit
def correlate_vector_to_matrix(vector, matrix):
    """Correlate a vector to a matrix

    Args:
        vector (np.ndarray): Vector to correlate
        matrix (np.ndarray): Matrix to correlate

    Returns:
        np.ndarray: Correlation result
    """
    corr = np.zeros(matrix.shape[1])
    for i in range(matrix.shape[1]):
        corr[i] = np.corrcoef(vector, matrix[:, i])[0, 1]
    return corr


def rolling_window_test(a, window):
    """
    Create a rolling window view of the input array.

    Parameters:
    a (numpy.ndarray): Input array.
    window (int): Size of the rolling window.

    Returns:
    numpy.ndarray: A view of the input array with the rolling window applied.

    Example:
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> rolling_window_test(a, 3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    """
    shp = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shp, strides=strides)

def moving_linear_fit(x_array, y_array, window_size, **kwargs):
    """
    Perform a moving linear fit on the given data arrays.
    Parameters:
    x_array (np.ndarray): The array of x-values.
    y_array (np.ndarray): The array of y-values.
    window_size (int): The size of the moving window.
    **kwargs: Additional keyword arguments.
    Returns:
    dict: A dictionary containing the following keys:
        - "slope" (np.ndarray): The slope of the linear fit for each window.
        - "std_slope" (float): The standard deviation of the slope.
        - "durbin_watson" (np.ndarray): The Durbin-Watson statistic for each window.
        - "mrse" (np.ndarray): The mean squared relative error for each window.
        - "anderson" (np.ndarray): The Anderson-Darling statistic for each window.
    """ 
    xdata = rolling_window_test(x_array, window_size).T
    ydata = rolling_window_test(y_array, window_size).T
    
    d, slope = np.nan*np.ones(len(x_array)), np.nan*np.ones(len(x_array))
    mrse, anderson_coef = np.nan*np.ones(len(x_array)), np.nan*np.ones(len(x_array))
    for idx in range(ydata.shape[1]):
        # fit_parameters = np.polyfit(xdata[:,idx], ydata[:,idx], deg = degree, full=False)
        stats = linear_fit(xdata[:,idx], ydata[:,idx])
        slope[idx]  = stats["parameters"][1] 
        std_slope = stats["standard_deviation_parameters"][1]
        d[idx] = stats["durbin_watson"]
        mrse[idx] = stats["msre"]
        anderson_coef[idx] = stats["anderson"][0]
        if kwargs.get("debugger", False):
            ranges = [1100, 1300, 1600, 1900]
            if idx in ranges:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(xdata[:,idx], ydata[:,idx], linewidth=0, marker='o', label=f'{ranges[idx]}')
                ax.plot(xdata[:,idx],np.polyval(np.flip(stats["parameters"]), xdata[:,idx]), label='fit')
                fig.savefig(f'test_dws_{idx}.png', dpi=300)
                plt.close(fig)
    results = {"slope": slope, "std_slope": std_slope ,"durbin_watson": d, "mrse": mrse, "anderson": anderson_coef}
    return results
