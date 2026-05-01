from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from lidarpy.plot.quicklook import quicklook_lpdr, quicklook_xarray
from lidarpy.retrieval.synthetic.generator import (
    synthetic_raman_signals_2D,
    synthetic_signals_2D,
    synthetic_signals_despo,
)


ARTIFACT_DIR = Path("artifacts/synthetic_quicklook")


def test_quicklook_from_synthetic_signal():
    ranges = np.arange(7.5, 3000.0, 30.0)
    minutes = np.arange(0.0, 20.0, 5.0)
    start = datetime(2023, 8, 30, 3, 0)
    times = np.array([start + timedelta(minutes=float(minute)) for minute in minutes])

    signal, _ = synthetic_signals_2D(
        ranges,
        minutes,
        number_of_initial_nan_values=3,
    )
    signal = signal.assign_coords(time=times).rename("signal_532fta")
    finite_values = signal.values[np.isfinite(signal.values)]
    scale_bounds = (0.0, float(finite_values.max() * 1.4))

    fig, _ = quicklook_xarray(
        signal,
        lidar_name="SYNTHETIC",
        location="Synthetic",
        is_rcs=True,
        scale_bounds=scale_bounds,
        ylims=(0.0, 3000.0),
    )

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ARTIFACT_DIR / "quicklook_synthetic_532fta.png"
        fig.savefig(output_path, format="png", dpi=100, bbox_inches="tight")
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        plt.close(fig)


def test_quicklook_from_synthetic_raman_signal():
    ranges = np.arange(7.5, 3000.0, 30.0)
    minutes = np.arange(0.0, 20.0, 5.0)
    start = datetime(2023, 8, 30, 3, 0)
    times = np.array([start + timedelta(minutes=float(minute)) for minute in minutes])

    signal, _ = synthetic_raman_signals_2D(
        ranges,
        minutes,
        number_of_initial_nan_values=3,
    )
    signal = signal.assign_coords(time=times).rename("signal_607ntp")
    finite_values = signal.values[np.isfinite(signal.values)]
    scale_bounds = (0.0, float(finite_values.max() * 1.4))

    fig, _ = quicklook_xarray(
        signal,
        lidar_name="SYNTHETIC",
        location="Synthetic",
        is_rcs=True,
        scale_bounds=scale_bounds,
        ylims=(0.0, 3000.0),
    )

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ARTIFACT_DIR / "quicklook_synthetic_607ntp.png"
        fig.savefig(output_path, format="png", dpi=100, bbox_inches="tight")
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        plt.close(fig)


def test_quicklook_from_synthetic_lpdr():
    ranges = np.arange(7.5, 3000.0, 30.0)
    minutes = np.arange(0.0, 20.0, 5.0)
    start = datetime(2023, 8, 30, 3, 0)
    times = np.array([start + timedelta(minutes=float(minute)) for minute in minutes])

    _, _, params = synthetic_signals_despo(
        ranges,
        number_of_initial_nan_values=3,
    )
    lpdr_profile = np.asarray(params["despolarization_volumic"])
    lpdr_values = np.tile(lpdr_profile[np.newaxis, :], (times.size, 1))
    lpdr = xr.DataArray(
        lpdr_values,
        dims=("time", "range"),
        coords={"time": times, "range": ranges},
        name="lpdr_532",
    )

    fig, _ = quicklook_lpdr(
        lpdr,
        scale_bounds=(0.0, 0.5),
        ylims=(0.0, 3000.0),
    )

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ARTIFACT_DIR / "quicklook_synthetic_lpdr_532.png"
        fig.savefig(output_path, format="png", dpi=100, bbox_inches="tight")
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        plt.close(fig)
