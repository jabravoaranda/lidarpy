from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from lidarpy.utils.utils import signal_to_rcs

ABLHMethod = Literal["wct", "temporal_variance"]
InputKind = Literal["auto", "lidarpy", "cloudnet"]


def prepare_ablh_input(
    data: xr.Dataset | xr.DataArray,
    *,
    input_kind: InputKind = "auto",
    variable: str | None = None,
    channel: str | None = None,
    convert_lidar_signal_to_rcs: bool = True,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray | None, str]:
    """Normalize supported lidar/ceilometer inputs to ``DataArray(time, range)``.

    Returns the signal used for detection, range in meters above instrument,
    optional height above sea level, and the source variable name.
    """
    if isinstance(data, xr.DataArray):
        signal = _ensure_time_range(data)
        return (
            signal,
            _range_values(signal),
            _height_values(data),
            signal.name or "data",
        )

    if input_kind == "auto":
        input_kind = _infer_input_kind(data, variable=variable, channel=channel)

    if input_kind == "cloudnet":
        source = variable or _first_available(data, ("beta_smooth", "beta", "beta_raw"))
        signal = _ensure_time_range(data[source])
        return signal, _range_values(signal), _height_values(data), source

    if input_kind == "lidarpy":
        source = variable or _lidarpy_signal_name(data, channel=channel)
        signal = _ensure_time_range(data[source])
        if convert_lidar_signal_to_rcs:
            signal = signal_to_rcs(signal, signal["range"])
            signal.attrs = dict(data[source].attrs)
            signal.attrs["range_corrected"] = "true"
        return signal, _range_values(signal), _height_values(data), source

    raise ValueError(f"Unsupported input_kind: {input_kind}")


def detect_ablh(
    data: xr.Dataset | xr.DataArray,
    *,
    method: ABLHMethod = "wct",
    input_kind: InputKind = "auto",
    variable: str | None = None,
    channel: str | None = None,
    min_range: float = 500.0,
    max_range: float = 4000.0,
    wct_width: float = 300.0,
    threshold: float = 0.05,
    threshold_step: float = 0.005,
    time_window_minutes: float = 10.0,
    convert_lidar_signal_to_rcs: bool = True,
) -> xr.Dataset:
    """Detect atmospheric boundary layer height from lidarpy or Cloudnet data."""
    signal, ranges, heights, source = prepare_ablh_input(
        data,
        input_kind=input_kind,
        variable=variable,
        channel=channel,
        convert_lidar_signal_to_rcs=convert_lidar_signal_to_rcs,
    )

    if method == "wct":
        ablh_range = calculate_ablh_wct(
            signal,
            ranges,
            wct_width=wct_width,
            threshold=threshold,
            threshold_step=threshold_step,
            min_range=min_range,
            max_range=max_range,
        )
    elif method == "temporal_variance":
        ablh_range = calculate_ablh_temporal_variance(
            signal,
            ranges,
            signal["time"].values,
            time_window_minutes=time_window_minutes,
            threshold=threshold,
            min_range=min_range,
            max_range=max_range,
        )
    else:
        raise ValueError(f"Unsupported ABLH method: {method}")

    return ablh_to_dataset(
        time=signal["time"].values,
        ablh_range=ablh_range,
        ranges=ranges,
        heights=heights,
        method=method,
        source_variable=source,
        min_range=min_range,
        max_range=max_range,
    )


def calculate_ablh_wct(
    signal: xr.DataArray,
    ranges: np.ndarray,
    *,
    wct_width: float = 300.0,
    threshold: float = 0.05,
    threshold_step: float = 0.005,
    min_range: float = 500.0,
    max_range: float = 4000.0,
) -> np.ndarray:
    """Detect ABLH using a Haar Wavelet Covariance Transform."""
    values = _values_time_range(signal)
    ranges = np.asarray(ranges, dtype=float)
    n_time, n_range = values.shape
    spacing = _median_spacing(ranges)
    window_bins = max(2, int(round(wct_width / spacing)))
    half = max(1, window_bins // 2)
    start_idx, end_idx = _search_indexes(ranges, min_range, max_range)
    end_idx = min(end_idx, n_range - half - 1)

    ablh = np.full(n_time, np.nan, dtype=float)
    search_mask = ranges <= max_range
    for time_idx in range(n_time):
        profile = values[time_idx, :].astype(float)
        finite = np.isfinite(profile) & search_mask
        if not finite.any():
            continue
        max_value = np.nanmax(profile[finite])
        if not np.isfinite(max_value) or max_value == 0:
            continue

        normalized = profile / max_value
        wct = np.full(n_range, np.nan, dtype=float)
        for range_idx in range(half, n_range - half):
            lower = normalized[range_idx - half : range_idx]
            upper = normalized[range_idx : range_idx + half]
            wct[range_idx] = (np.nansum(lower) - np.nansum(upper)) / window_bins

        current = threshold
        while current > 0 and np.isnan(ablh[time_idx]):
            for range_idx in range(start_idx, end_idx):
                if (
                    wct[range_idx] > current
                    and wct[range_idx] > wct[range_idx - 1]
                    and wct[range_idx] > wct[range_idx + 1]
                ):
                    ablh[time_idx] = ranges[range_idx]
                    break
            current -= threshold_step

    return ablh


def calculate_ablh_temporal_variance(
    signal: xr.DataArray,
    ranges: np.ndarray,
    time: np.ndarray,
    *,
    time_window_minutes: float = 10.0,
    threshold: float = 0.05,
    min_range: float = 500.0,
    max_range: float = 4000.0,
) -> np.ndarray:
    """Detect ABLH as the strongest temporal variance layer."""
    values = _values_time_range(signal)
    ranges = np.asarray(ranges, dtype=float)
    n_time, _ = values.shape
    start_idx, end_idx = _search_indexes(ranges, min_range, max_range)

    time_minutes = _time_to_minutes(time)
    if n_time > 1:
        time_step = np.nanmedian(np.diff(time_minutes))
    else:
        time_step = time_window_minutes
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("Time coordinate must be monotonic for temporal variance.")

    window_bins = max(1, int(round(time_window_minutes / time_step)))
    half_window = max(1, window_bins // 2)
    normalizer_mask = ranges <= max_range
    normalizer = np.nanmax(values[:, normalizer_mask], axis=1, keepdims=True)
    normalizer = np.where(
        np.isfinite(normalizer) & (normalizer != 0), normalizer, np.nan
    )
    normalized = values / normalizer

    moving_variance = _centered_moving_nanvariance(
        normalized[:, start_idx:end_idx],
        half_window=half_window,
    )

    ablh = np.full(n_time, np.nan, dtype=float)
    for time_idx in range(n_time):
        search_zone = moving_variance[time_idx, :]
        if search_zone.size == 0 or np.all(np.isnan(search_zone)):
            continue
        relative_idx = int(np.nanargmax(search_zone))
        if search_zone[relative_idx] > threshold:
            ablh[time_idx] = ranges[start_idx + relative_idx]

    return ablh


def ablh_to_dataset(
    *,
    time: np.ndarray,
    ablh_range: np.ndarray,
    ranges: np.ndarray,
    heights: np.ndarray | None,
    method: str,
    source_variable: str,
    min_range: float,
    max_range: float,
) -> xr.Dataset:
    """Build a small NetCDF-ready ABLH dataset."""
    ablh_range = np.asarray(ablh_range, dtype=np.float32)
    if heights is None:
        ablh_height = np.full_like(ablh_range, np.nan, dtype=np.float32)
    else:
        ablh_height = np.interp(ablh_range, ranges, heights, left=np.nan, right=np.nan)
        ablh_height = ablh_height.astype(np.float32)

    return xr.Dataset(
        data_vars={
            "ablh": (
                ("time",),
                ablh_range,
                {
                    "units": "m",
                    "long_name": "Atmospheric boundary layer height above instrument",
                },
            ),
            "ablh_range": (
                ("time",),
                ablh_range,
                {"units": "m", "long_name": "ABLH range above instrument"},
            ),
            "ablh_height": (
                ("time",),
                ablh_height,
                {"units": "m", "long_name": "ABLH height above mean sea level"},
            ),
        },
        coords={"time": pd.to_datetime(time).to_numpy()},
        attrs={
            "title": "Atmospheric boundary layer height detected by lidarpy",
            "method": method,
            "source_variable": source_variable,
            "min_range_m": float(min_range),
            "max_range_m": float(max_range),
        },
    )


def _infer_input_kind(
    dataset: xr.Dataset,
    *,
    variable: str | None,
    channel: str | None,
) -> InputKind:
    if variable and variable in {"beta_smooth", "beta", "beta_raw"}:
        return "cloudnet"
    if channel or any(name.startswith("signal_") for name in dataset.data_vars):
        return "lidarpy"
    if any(name in dataset.data_vars for name in ("beta_smooth", "beta", "beta_raw")):
        return "cloudnet"
    raise ValueError("Could not infer ABLH input kind from dataset variables.")


def _first_available(dataset: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset:
            return name
    raise KeyError(f"None of these variables were found: {', '.join(names)}")


def _lidarpy_signal_name(dataset: xr.Dataset, *, channel: str | None) -> str:
    if channel:
        name = channel if channel.startswith("signal_") else f"signal_{channel}"
        if name not in dataset:
            raise KeyError(f"Variable '{name}' not found in lidarpy dataset.")
        return name
    signal_names = [name for name in dataset.data_vars if name.startswith("signal_")]
    if not signal_names:
        raise KeyError("No lidarpy signal_* variable found.")
    return signal_names[0]


def _ensure_time_range(data_array: xr.DataArray) -> xr.DataArray:
    if "time" not in data_array.dims or "range" not in data_array.dims:
        raise ValueError("ABLH input must have 'time' and 'range' dimensions.")
    return data_array.transpose("time", "range")


def _values_time_range(data_array: xr.DataArray) -> np.ndarray:
    return _ensure_time_range(data_array).values.astype(float)


def _range_values(data_array: xr.DataArray) -> np.ndarray:
    if "range" not in data_array.coords:
        raise ValueError("ABLH input must provide a 'range' coordinate.")
    return data_array["range"].values.astype(float)


def _height_values(data: xr.Dataset | xr.DataArray) -> np.ndarray | None:
    if isinstance(data, xr.Dataset) and "height" in data:
        return data["height"].values.astype(float)
    if isinstance(data, xr.DataArray) and "height" in data.coords:
        return data["height"].values.astype(float)
    return None


def _median_spacing(values: np.ndarray) -> float:
    spacing = float(np.nanmedian(np.diff(values)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Range coordinate must be increasing.")
    return spacing


def _search_indexes(
    ranges: np.ndarray,
    min_range: float,
    max_range: float,
) -> tuple[int, int]:
    if min_range >= max_range:
        raise ValueError("min_range must be lower than max_range.")
    start_idx = int(np.searchsorted(ranges, min_range, side="left"))
    end_idx = int(np.searchsorted(ranges, max_range, side="right"))
    if start_idx >= end_idx:
        raise ValueError("No range bins found within the requested ABLH interval.")
    return start_idx, end_idx


def _time_to_minutes(time: np.ndarray) -> np.ndarray:
    values = np.asarray(time)
    if np.issubdtype(values.dtype, np.datetime64):
        return ((values - values[0]) / np.timedelta64(1, "m")).astype(float)
    return values.astype(float)


def _centered_moving_nanvariance(
    values: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    counts = finite.astype(float)

    padded_sum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(clean, axis=0)])
    padded_sumsq = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(clean * clean, axis=0)]
    )
    padded_count = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(counts, axis=0)]
    )

    centers = np.arange(values.shape[0])
    starts = np.maximum(0, centers - half_window)
    ends = np.minimum(values.shape[0], centers + half_window + 1)

    window_sum = padded_sum[ends] - padded_sum[starts]
    window_sumsq = padded_sumsq[ends] - padded_sumsq[starts]
    window_count = padded_count[ends] - padded_count[starts]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = window_sum / window_count
        variance = window_sumsq / window_count - mean * mean
    variance[window_count < 2] = np.nan
    variance = np.where(variance >= 0, variance, 0.0)
    return variance
