from datetime import datetime

import matplotlib.pyplot as plt
import numba
import numpy as np
import xarray as xr
from lidarpy.atmo.freudenthaler_molecular_properties import molecular_properties
from lidarpy.atmo.ecmwf import get_ecmwf_day
from lidarpy.general_utils.utils import parse_datetime
from lidarpy.general_utils.smoothing import moving_average
from scipy.signal import savgol_filter


def molecular_properties_2d(
    date: datetime | str,
    heights: np.ndarray,
    times: np.ndarray,
    wavelength: float = 532,
) -> xr.Dataset:
    """Calculate molecular properties for a given date, height ranges, and times.
    This function requests ECMWF (European Centre for Medium-Range Weather Forecasts) temperatures and pressures for a specified date and pipes them to `lidarpy.atmo.atmo.molecular_properties` to calculate molecular properties.

    Args:
        date (datetime | str): The date for which to request ECMWF data. Can be a datetime object or a string.
        heights (np.ndarray): An array of height ranges (in meters) for which to calculate molecular properties.
        times (np.ndarray): An array of times (in hours) for which to calculate molecular properties.
        wavelength (float, optional): The wavelength (in nanometers) for which to calculate molecular properties. Defaults to 532 nm.

    Returns:
        xr.Dataset: A dataset containing the following variables:
            - "molecular_beta": Molecular backscatter coefficient.
            - "molecular_alpha": Molecular extinction coefficient.
            - "attenuated_molecular_beta": Attenuated molecular backscatter coefficient.
            - "molecular_lidar_ratio": Molecular lidar ratio.
    """

    _date = parse_datetime(date)
    atmo_d = get_ecmwf_day(_date, heights=heights, times=times)

    mol_d = molecular_properties(
        wavelength,
        pressure=atmo_d.pressure.values,
        temperature=atmo_d.temperature.values,
        heights=atmo_d.range.values,
        times=atmo_d.time.values,
    )
    return mol_d


def iterative_fitting(
    rcs_profile: np.ndarray,
    attenuated_molecular_backscatter: np.ndarray,
    window_size_bins: int = 5,
    min_bin: int = 600,
    max_bin: int = 1000,
    debugger: bool = False,
) -> np.ndarray:
    """
    Perform iterative fitting on RCS and attenuated molecular backscatter profiles. 
    This function normalizes the RCS and attenuated molecular backscatter profiles
    and performs a linear fit over a sliding window. It calculates the slope of the
    fitted line for each window and returns a boolean matrix indicating the fit quality.
    
    Args:
    rcs_profile (np.ndarray): The range-corrected signal (RCS) profile.
    attenuated_molecular_backscatter (np.ndarray): The attenuated molecular backscatter profile.
    window_size_bins (int, optional): The size of the sliding window in bins. Default is 5.
    min_bin (int, optional): The minimum bin index to start the fitting. Default is 600.
    max_bin (int, optional): The maximum bin index to end the fitting. Default is 1000.
    debugger (bool, optional): If True, the function will plot the fitted lines and the original data. Default is False.

    Returns:
    np.ndarray: A boolean matrix indicating the fit quality.
    
    Raises:
    ValueError: If the shapes of `rcs_profile` and `attenuated_molecular_backscatter` do not match.
    """

    if rcs_profile.shape != attenuated_molecular_backscatter.shape:
        raise ValueError(f"RCS and Betta ranges must match")

    x_axis = np.arange(window_size_bins * 2)
    bool_matrix = np.full_like(rcs_profile, False, dtype=np.bool_)

    slope = []
    slope_mol = []

    for idx in np.arange(min_bin, max_bin):
        _rcs_norm = rcs_profile / rcs_profile[idx]
        _att_norm = (
            attenuated_molecular_backscatter / attenuated_molecular_backscatter[idx]
        )

        prof_slice = _rcs_norm[idx - window_size_bins : idx + window_size_bins]
        att_slice = _att_norm[idx - window_size_bins : idx + window_size_bins]

        coeff_prof = np.polyfit(np.arange(window_size_bins * 2), prof_slice, 1)
        coeff_att = np.polyfit(np.arange(window_size_bins * 2), att_slice, 1)

        slope.append(coeff_prof[0])
        slope_mol.append(coeff_att[0])


        att_data = np.polyval(coeff_att, x_axis)
        r2 = 1 - (
            ((prof_slice - att_data) ** 2).sum()
            / ((prof_slice - att_data.mean()) ** 2).sum()
        )

        if debugger:
            plt.scatter(x_axis, prof_slice, c="g")
            plt.plot(x_axis, np.polyval(coeff_prof, x_axis), c="g")
            plt.scatter(x_axis, att_slice, c="b")
            plt.plot(x_axis, np.polyval(coeff_att, x_axis), c="b")
            plt.show()
            print(f'Mol m: {coeff_att[0]}')
            print(f'Prof m: {coeff_prof[0]}')
            if r2 > 0.25:
                print(f"R^2: {r2}")

            plt.plot(slope)
            plt.plot(slope_mol)
            plt.close()
    return bool_matrix


def split_continous_measurements(
    time_array: np.ndarray, time_greater_than: float = 121 # Two minutest to avoid one profile search multiple DC
) -> list[np.ndarray]:
    
    """Groups times array into clusters with no more than `time_greater_than` seconds between measurements.
        
    Args:
        time_array (np.ndarray): Array of time measurements.
        time_greater_than (float, optional): Maximum allowed time difference between consecutive measurements to be considered in the same cluster. Defaults to 121 seconds.

    Returns:
        list[np.ndarray]: List of numpy arrays, each containing a cluster of time measurements.
    """
    diffs = (time_array[1:] - time_array[0:-1]).astype("f") / 1e9  # type: ignore
    return np.split(time_array, np.where(diffs > time_greater_than)[0] + 1)


def mask_by_slope(
    rcs: xr.DataArray,
    att_beta: xr.DataArray,
    min_height: float = 4000,
    max_height: float = 7000,
    window_size: float = 1000,
    window_time: float = 30,
    max_rel_error: float = 0.15,
    plot_profile: int | None = None,
) -> xr.DataArray:
    rcs_sel = rcs.sel(range=slice(min_height, max_height))
    beta_sel = att_beta.sel(range=slice(min_height, max_height))

    rcs_sel = xr.apply_ufunc(
        moving_average, rcs_sel, kwargs={"window_size": window_time}, dask="allowed"
    )
    rcs_sel = xr.apply_ufunc(
        smooth_profiles, rcs_sel, kwargs={"window_size": 200}, dask="allowed"
    )

    result = np.full_like(beta_sel.values, False, dtype=bool)

    for idx, height in enumerate(rcs_sel.range):
        valid_ranges = rcs_sel.range[
            (height - window_size / 2 <= rcs_sel.range)
            & (height + window_size / 2 >= rcs_sel.range)
        ]
        height_rcs = rcs_sel.loc[dict(range=valid_ranges)]
        height_rcs /= height_rcs.isel(range=0)

        height_beta = beta_sel.loc[dict(range=valid_ranges)]
        height_beta /= height_beta.isel(range=0)

        x = np.arange(height_rcs.shape[1])

        slopes_rcs = np.polyfit(x, height_rcs.values.T, 1)[0]
        slopes_beta = np.polyfit(x, height_beta.values.T, 1)[0]

        result[:, idx] = (
            np.abs((slopes_rcs - slopes_beta) / slopes_beta) <= max_rel_error
        )

    result_data_array = xr.full_like(rcs, False)
    result_data_array.loc[dict(range=slice(min_height, max_height))] = result

    if plot_profile is not None:
        md = (max_height + min_height) / 2
        n_rcs = rcs[plot_profile] / rcs[plot_profile].sel(range=md, method="nearest")
        n_rcs_smth = rcs_sel[plot_profile] / rcs_sel[plot_profile].sel(
            range=md, method="nearest"
        )
        n_beta = att_beta[plot_profile] / att_beta[plot_profile].sel(
            range=md, method="nearest"
        )
        plt.plot(rcs.range, n_rcs)
        plt.plot(att_beta.range, n_beta)
        plt.plot(n_rcs_smth.range, n_rcs_smth)

        results = np.where(result_data_array[plot_profile], n_beta, np.nan)

        plt.plot(result_data_array.range, results, c="r", lw=2)

        plt.yscale("log")
        plt.show()

    return result_data_array


def mask_by_corrcoef(
    rcs: xr.DataArray,
    att_beta: xr.DataArray,
    min_height: float = 1000,
    max_height: float = 15000,
    window_size: float = 500,
    window_time: float = 15,
    min_corr: float = 0.95,
):
    """
    Masks a given range-corrected signal (RCS) DataArray based on the correlation coefficient 
    with an attenuated backscatter (att_beta) DataArray within a specified height range.
    
    Parameters:
    -----------
    rcs : xr.DataArray
        The range-corrected signal DataArray.
    att_beta : xr.DataArray
        The attenuated backscatter DataArray.
    min_height : float, optional
        The minimum height (in meters) for the selection range. Default is 1000.
    max_height : float, optional
        The maximum height (in meters) for the selection range. Default is 15000.
    window_size : float, optional
        The window size (in meters) for the correlation calculation. Default is 500.
    window_time : float, optional
        The window time (in minutes) for the adaptive moving average. Default is 15.
    min_corr : float, optional
        The minimum correlation coefficient threshold for masking. Default is 0.95.
    
    Returns:
    --------
    xr.DataArray
        A DataArray with the same shape as `rcs`, where values within the specified height 
        range are masked based on the correlation coefficient with `att_beta`.
    """
 
    rcs_sel = rcs.sel(range=slice(min_height, max_height))
    beta_sel = att_beta.sel(range=slice(min_height, max_height))

    rcs_sel = xr.apply_ufunc(smooth_profiles, rcs_sel, kwargs={"window_size": 170})
    rcs_sel = xr.apply_ufunc(
        moving_average, rcs_sel, kwargs={"window_size": window_time}
    )

    ranges = rcs_sel.range.values
    
    range_mask_limits = [
        np.where((h - window_size / 2 <= ranges) & (h + window_size / 2 >= ranges))[0]
        for h in ranges
    ]

    lim_indexes = [
        [mask[0], mask[-1]] for mask in range_mask_limits
    ]

    result_array = windowed_correlation(
        rcs_sel.values,
        beta_sel.values,
        range_mask_limits=lim_indexes,
        min_corr=min_corr,
    )

    result_data_array = xr.full_like(rcs, False)
    result_data_array.loc[dict(range=slice(min_height, max_height))] = result_array

    return result_data_array
    # TODO:Finish function


def smooth_profiles(
    arr: np.ndarray, /, *, window_size: int = 11, polyorder: int = 3
) -> np.ndarray:
    """
    Smooths the profiles in a 2D array using the Savitzky-Golay filter.
    Parameters:
    arr (np.ndarray): A 2D array where each row represents a profile to be smoothed.
    window_size (int, optional): The length of the filter window (i.e., the number of coefficients). 
                                 Must be a positive odd integer. Default is 11.
    polyorder (int, optional): The order of the polynomial used to fit the samples. 
                               Must be less than window_size. Default is 3.
    Returns:
    np.ndarray: A 2D array with the same shape as `arr`, where each profile has been smoothed.
    """

    def smooth_profile(_x: np.ndarray) -> np.ndarray:
        return savgol_filter(_x, window_size, 3)

    return np.apply_along_axis(smooth_profile, 1, arr)


@numba.njit(parallel=True)
def windowed_correlation(
    rcs: np.ndarray,
    att_beta: np.ndarray,
    /,
    *,
    range_mask_limits: list[list[int]],
    min_corr: float,
) -> np.ndarray:
    """
    Compute the windowed correlation between two arrays with a given range mask and minimum correlation threshold.
    Parameters
    ----------
    rcs : np.ndarray
        The first input array for correlation computation.
    att_beta : np.ndarray
        The second input array for correlation computation.
    range_mask_limits : list[list[int]]
        A list of lists containing the start and end indices for masking the range of the input arrays.
    min_corr : float
        The minimum correlation threshold to determine if the correlation is significant.
    Returns
    -------
    np.ndarray
        An array of the same shape as `rcs` containing boolean values indicating whether the correlation
        at each point is greater than or equal to `min_corr`.
    """

    result_array = np.full(rcs.shape, np.nan)

    for t_idx in numba.prange(rcs.shape[0]):
        for r_idx in numba.prange(rcs.shape[1]):
            r_masks = range_mask_limits[r_idx]

            rcs_masked = rcs[t_idx][r_masks[0] : r_masks[1]]
            beta_masked = att_beta[t_idx][r_masks[0] : r_masks[1]]

            rcs_masked /= beta_masked[0]

            corr = np.corrcoef(rcs_masked, beta_masked)[0, 1]
            # set_trace()
            result_array[t_idx, r_idx] = corr >= min_corr

    return result_array


def cluster_value(arr: np.ndarray, /, *, value=1) -> list[list[tuple[int, int]]]:
    def cluster_1d(row) -> list[tuple[int, int]]:
        """
        Identifies clusters of consecutive elements in a 2D numpy array that match a specified value.
        Args:
            arr (np.ndarray): A 2D numpy array to be processed.
            value (int, optional): The value to identify clusters of. Defaults to 1.
        Returns:
            list[list[tuple[int, int]]]: A list of lists, where each inner list contains tuples representing
                                        the start position and length of each cluster found in the corresponding
                                        row of the input array.
        """
        _i: int = 0
        count: int = 0
        clusters: list[tuple[int, int]] = []  # list[position, count]
        for (_i, *_), _v in np.ndenumerate(row):
            if _v == value and count == 0:
                count += 1
            elif _v == value and count != 0:
                count += 1
            elif _v != value and count == 0:
                continue
            elif _v != value and count != 0:
                clusters.append((_i - (1 + count), count))
                count = 0

        if count != 0:
            clusters.append((_i - (1 + count), count))

        return clusters

    time_cluster: list[list[tuple[int, int]]] = []

    for time_row in arr:
        time_cluster.append(cluster_1d(time_row))

    return time_cluster


def cluster_at_least(
    clusters: list[list[tuple[int, int]]], n_min: int = 5, /
) -> np.ndarray:
    result = np.full((len(clusters), 2), np.nan, dtype=np.float64)
    """
    Filters and processes clusters based on a minimum threshold.
    This function takes a list of clusters, where each cluster is a list of tuples.
    Each tuple contains two integers. The function filters out tuples where the 
    second integer is less than `n_min`. For each cluster, it selects the tuple 
    with the highest second integer value that meets the threshold, and stores 
    the first integer and the sum of the first and second integers in the result 
    array.
    Parameters:
    -----------
    clusters : list[list[tuple[int, int]]]
        A list of clusters, where each cluster is a list of tuples containing two integers.
    n_min : int, optional
        The minimum threshold for the second integer in the tuples (default is 5).
    Returns:
    --------
    np.ndarray
        A 2D numpy array where each row corresponds to a cluster. Each row contains 
        two values: the first integer from the selected tuple and the sum of the 
        first and second integers from the selected tuple. If no tuple meets the 
        threshold in a cluster, the row contains NaN values.
    """

    for idx, prof_clust in enumerate(clusters):
        filtered = list(filter(lambda t: t[1] >= n_min, prof_clust))

        if len(filtered) == 0:
            continue

        selected = sorted(filtered, key=lambda t: t[1], reverse=True)[0]
        result[idx] = np.array([selected[0], selected[0] + selected[1]])

    return result

