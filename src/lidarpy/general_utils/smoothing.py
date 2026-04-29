import numpy as np
import xarray as xr

import numba as nb

import numpy as np
import xarray as xr

BIN_RES_M = 3.75  # native resolution (m)

def bin_rescale(
    dataset: xr.Dataset,
    factor: int,
    *,
    default_agg: str = "mean",
    var_aggs: dict | None = None,
    bin_coord: str = "center",  # "center" | "left" | "right"
) -> xr.Dataset:
    """
    Bin averaging for LIDAR range dimension.

    - Agrupa 'factor' bins contiguos en 'range' para reducir ruido y mejorar SNR.
    - 'default_agg' se aplica a todas las variables con 'range' salvo que se
      sobrescriba por variable en 'var_aggs' (e.g., {'photon_counts': 'sum'}).
    - 'bin_coord' controla cómo se calcula la nueva coordenada de rango
      (centro del bin por defecto).

    Notas:
      * 'mean' es adecuado para magnitudes intensivas (p.ej., señales normalizadas).
      * 'sum' es adecuado para magnitudes extensivas (p.ej., cuentas de fotones).
    """
    if factor is None or factor <= 1:
        return dataset
    if "range" not in dataset.dims:
        return dataset
    if default_agg not in {"mean", "sum"}:
        raise ValueError("default_agg must be 'mean' or 'sum'")
    var_aggs = var_aggs or {}

    n = dataset.sizes["range"]
    m = (n // factor) * factor
    if m == 0:
        # factor mayor que el tamaño: no tocar
        return dataset

    # Recorta para que sea múltiplo exacto del factor (boundary='trim' hará lo mismo)
    ds_trim = dataset.isel(range=slice(0, m))

    # Construye la nueva coordenada de 'range' a partir de la original
    r = ds_trim["range"].values
    r_blocks = r.reshape(-1, factor)

    if bin_coord == "center":
        new_range = r_blocks.mean(axis=1)
    elif bin_coord == "left":
        new_range = r_blocks[:, 0]
    elif bin_coord == "right":
        new_range = r_blocks[:, -1]
    else:
        raise ValueError("bin_coord must be 'center', 'left', or 'right'")

    new_data_vars = {}
    for name, da in ds_trim.data_vars.items():
        if "range" not in da.dims:
            # variables sin la dimensión 'range' se copian tal cual
            new_data_vars[name] = da
            continue

        agg = var_aggs.get(name, default_agg)
        coarsener = da.coarsen(range=factor, boundary="trim")
        if agg == "mean":
            # mean con skipna=True para ignorar NaNs
            rebinned = coarsener.mean(skipna=True)
        elif agg == "sum":
            rebinned = coarsener.sum()
        else:
            raise ValueError(f"agg for variable '{name}' must be 'mean' or 'sum'")

        # Sustituimos la coordenada 'range' por la nueva
        rebinned = rebinned.assign_coords(range=("range", new_range))
        # Conserva atributos originales de la variable
        rebinned.attrs = {**da.attrs}
        new_data_vars[name] = rebinned

    # Reconstruye el Dataset con coords originales (salvo 'range', que actualizamos)
    new_coords = {k: v for k, v in ds_trim.coords.items() if k != "range"}
    new_ds = xr.Dataset(data_vars=new_data_vars, coords=new_coords).assign_coords(range=("range", new_range))

    # Actualiza atributos globales
    new_ds.attrs.update(dataset.attrs)
    new_ds.attrs["rebin_factor"] = int(factor)
    new_ds.attrs["range_resolution_m"] = float(factor) * float(BIN_RES_M)
    new_ds.attrs["history"] = (
        f"{new_ds.attrs.get('history', '')} | bin_rescale: factor={factor}, "
        f"default_agg={default_agg}, bin_coord={bin_coord}"
    ).strip(" |")

    # Añade metadatos útiles a la coordenada 'range'
    range_attrs = dict(
        long_name="Range (binned)",
        units=dataset["range"].attrs.get("units", "m"),
        bin_width_m=float(factor) * float(BIN_RES_M),
        bin_coord=bin_coord,
        note="Computed via bin averaging (coarsen) from native resolution",
    )
    new_ds["range"].attrs.update(range_attrs)

    return new_ds

def moving_average(data: np.ndarray, window_sizes: np.ndarray | float) -> np.ndarray:
    """
    Computes an adaptive moving average using a variable window size.

    Parameters:
    -----------
    data : np.ndarray
        1D or 2D input data array.
    window_sizes : np.ndarray
        1D array of window sizes, one for each row/height.

    Returns:
    --------
    np.ndarray
        Smoothed data with adaptive averaging.
    """
    if isinstance(window_sizes, float):
        window_sizes = np.full_like(data, window_sizes)  # Same window size for all rows

    # Ensure minimum window size of 1
    window_sizes = np.maximum(window_sizes, 1.0)

    smoothed_data = np.full_like(data, np.nan)  # Initialize output

    for i in range(data.shape[0]):  # Iterate over time steps (or channels)
        w = window_sizes[i]
        # Compute cumulative sum trick for efficient moving average
        cumsum = np.cumsum(np.insert(data[i], 0, 0))
        start_idx = np.maximum(0, np.arange(len(data[i])) - w // 2).astype(int)
        end_idx = np.minimum(len(data[i]), np.arange(len(data[i])) + w // 2 + 1).astype(
            int
        )

        smoothed_data[i] = (cumsum[end_idx] - cumsum[start_idx]) / (end_idx - start_idx)

    return smoothed_data

def sliding_average(
    data_array: xr.DataArray, maximum_range: float, window_range: tuple[int, int]
) -> xr.DataArray:
    """
    Applies a sliding average with an adaptive window size based on the height (range).

    Parameters:
    -----------
    data_array : xr.DataArray
        Input signal array.
    maximum_range : float
        Height at which the smoothing window reaches its maximum size.
    window_range: tuple[int, int]: minimum and maximum window size. First starts at 0 m and second from maximum_range.

    Returns:
    --------
    xr.DataArray
        Smoothed signal.
    """
    minimum_window_size, maximum_window_size = window_range
    range_array = data_array["range"].values  # Heights (range)
    data_values = data_array.values  # Signal values

    # Normalize heights between 0 and 1
    norm_height = np.clip(range_array / maximum_range, 0, 1)

    # Compute window sizes (vectorized)
    window_sizes = np.round(
        minimum_window_size + (maximum_window_size - minimum_window_size) * norm_height
    ).astype(int)
    window_sizes = np.clip(
        window_sizes, 1, None
    )  # Ensure minimum window size is at least 1

    smoothed_data = np.full_like(data_values, np.nan)

    # Efficient adaptive moving average using cumulative sum
    for i in range(
        data_values.shape[0]
    ):  # Iterate over time steps (or channels if time is along axis 0)
        cumsum = np.cumsum(np.insert(data_values[i], 0, 0))  # Compute cumulative sum
        for j in range(len(data_values[i])):  # Iterate over range dimension
            w = window_sizes[j]
            start_idx = max(0, j - w // 2)
            end_idx = min(len(data_values[i]), j + w // 2 + 1)
            smoothed_data[i, j] = (cumsum[end_idx] - cumsum[start_idx]) / (
                end_idx - start_idx
            )

    return xr.DataArray(smoothed_data, dims=data_array.dims, coords=data_array.coords)


def estimate_snr(signal: xr.DataArray) -> xr.DataArray:
    """Estima la SNR como signal / sqrt(signal), evitando divisiones por 0."""
    noise = np.sqrt(np.abs(signal))
    snr = xr.where(noise > 0, signal / noise, 0.0)
    return snr

@nb.njit(parallel=True)
def _adaptive_average(signal, L_array):
    smoothed = np.zeros_like(signal)
    N = signal.shape[-1]

    for n in nb.prange(N):
        L = L_array[n]
        half = L // 2
        start = max(n - half, 0)
        end = min(n + half + 1, N)
        window = signal[..., start:end]
        smoothed[..., n] = np.sum(window) / window.size  # PROMEDIO manual
    return smoothed

def adaptive_sliding_average(
    signal: xr.DataArray,
    snr: xr.DataArray,
    snr_target: float = 5.0,
    L_min: int = 1,
    L_max: int = 50,
    dim: str = "range",
) -> xr.DataArray:
    """
    Faster adaptive sliding average based on SNR per bin using cumulative sums.

    Parameters
    ----------
    signal : xr.DataArray
    snr : xr.DataArray
    snr_target : float
    L_min : int
    L_max : int
    dim : str

    Returns
    -------
    xr.DataArray
    """
    if dim not in signal.dims:
        raise ValueError(f"Dimension '{dim}' not found in signal")

    axis = signal.get_axis_num(dim)

    # Compute window sizes based on SNR
    snr_clipped = snr.clip(min=1e-3)
    L_required = (snr_target / snr_clipped) ** 2
    L_array = np.clip(L_required, L_min, L_max).astype(int)

    data = signal.values
    L_vals = L_array.values
    smoothed = np.full_like(data, np.nan)

    other_dims = [d for d in range(data.ndim) if d != axis]
    shape = [data.shape[d] for d in other_dims]
    it = np.ndindex(*shape)

    for idx in it:
        # Build index to slice the 1D profile
        full_idx = [slice(None) if d == axis else i for d, i in enumerate(idx)]
        profile = data[tuple(full_idx)]
        window_sizes = L_vals[tuple(full_idx)]

        # Cumulative sum
        cumsum = np.insert(np.cumsum(np.nan_to_num(profile)), 0, 0.0)
        counts = ~np.isnan(profile)
        count_cumsum = np.insert(np.cumsum(counts), 0, 0)

        N = len(profile)
        for i in range(N):
            half_w = window_sizes[i] // 2
            start = max(0, i - half_w)
            end = min(N, i + half_w + 1)
            total = cumsum[end] - cumsum[start]
            count = count_cumsum[end] - count_cumsum[start]
            smoothed[tuple(full_idx[:axis] + [i] + full_idx[axis + 1 :])] = (
                total / count if count > 0 else np.nan
            )

    return xr.DataArray(
        smoothed,
        dims=signal.dims,
        coords=signal.coords,
        attrs=signal.attrs,
        name=signal.name + "_smoothed" if signal.name else "smoothed",
    )


from scipy.ndimage import uniform_filter1d

def sliding_average_ATLAS(
    da: xr.DataArray,
    height_dim: str = "range",
    smoothing_window: int = 3,
    err_type: str = "std",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Fast 1D sliding average smoothing using uniform_filter1d along a height dimension.

    Parameters:
        da: xr.DataArray with dims including `height_dim`
        height_dim: dimension along which to apply smoothing (default "range")
        smoothing_window: int, number of bins for moving average (should be odd)
        err_type: 'std' (default) or 'sem' (standard error of the mean)

    Returns:
        smoothed: xr.DataArray with moving average
        error: xr.DataArray with smoothed std or sem
    """

    axis = da.get_axis_num(height_dim)

    # Calculate mean (quitamos vectorize=True)
    mean_vals = xr.apply_ufunc(
        uniform_filter1d,
        da,
        input_core_dims=[[height_dim]],
        output_core_dims=[[height_dim]],
        kwargs={"size": smoothing_window, "axis": axis, "mode": "nearest"},
        dask="parallelized",  # opcional, solo si usas dask
        vectorize=False,  # corregido
        output_dtypes=[da.dtype],
    )

    # Compute error
    squared_diff = (da - mean_vals) ** 2
    var_vals = xr.apply_ufunc(
        uniform_filter1d,
        squared_diff,
        input_core_dims=[[height_dim]],
        output_core_dims=[[height_dim]],
        kwargs={"size": smoothing_window, "axis": axis, "mode": "nearest"},
        dask="parallelized",
        vectorize=False,  # corregido
        output_dtypes=[da.dtype],
    )
    std_vals = np.sqrt(var_vals)
    if err_type == "sem":
        std_vals = std_vals / np.sqrt(smoothing_window)

    smoothed = mean_vals.copy()
    smoothed.name = f"{da.name}_smoothed" if da.name else "smoothed"
    error = std_vals.copy()
    error.name = f"{da.name}_error" if da.name else "error"

    return smoothed, error


def smoothing_ATLAS(dataset: xr.Dataset, smoothing_window=None, err_type="std") -> xr.Dataset:
    """
    Apply sliding average smoothing to all signal_* variables in the dataset.

    Parameters:
        dataset: xarray.Dataset with variables named like 'signal_*'
        smoothing_window: int or tuple; if tuple, its mean is used
        err_type: 'std' or 'sem'

    Returns:
        xr.Dataset with smoothed signals and their associated errors
    """
    if smoothing_window is None:
        smoothing_window = (30.0, 150.0)

    if isinstance(smoothing_window, tuple):
        smoothing_window_scalar = int(np.mean(smoothing_window))
    else:
        smoothing_window_scalar = int(smoothing_window)

    smoothed_ds = dataset.copy(deep=True)

    for var in [v for v in dataset.data_vars if v.startswith("signal_")]:
        da = dataset[var]
        smoothed_da, error_da = sliding_average_ATLAS(
            da,
            height_dim="range",
            smoothing_window=smoothing_window_scalar,
            err_type=err_type,
        )
        smoothed_ds[var] = smoothed_da
        smoothed_ds[f"stderr_{var.split('_', 1)[1]}"] = error_da

    return smoothed_ds
