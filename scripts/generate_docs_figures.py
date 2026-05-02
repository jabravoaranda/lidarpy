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


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"


def _synthetic_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.arange(7.5, 3000.0, 30.0)
    minutes = np.arange(0.0, 20.0, 5.0)
    start = datetime(2023, 8, 30, 3, 0)
    times = np.array([start + timedelta(minutes=float(minute)) for minute in minutes])
    return ranges, minutes, times


def _save_quicklook(signal: xr.DataArray, filename: str) -> None:
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
        fig.savefig(ASSET_DIR / filename, format="png", dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)


def generate_elastic_quicklook() -> None:
    ranges, minutes, times = _synthetic_coordinates()
    signal, _ = synthetic_signals_2D(
        ranges,
        minutes,
        number_of_initial_nan_values=3,
    )
    signal = signal.assign_coords(time=times).rename("signal_532fta")
    _save_quicklook(signal, "synthetic-elastic-quicklook.png")


def generate_raman_quicklook() -> None:
    ranges, minutes, times = _synthetic_coordinates()
    signal, _ = synthetic_raman_signals_2D(
        ranges,
        minutes,
        number_of_initial_nan_values=3,
    )
    signal = signal.assign_coords(time=times).rename("signal_607ntp")
    _save_quicklook(signal, "synthetic-raman-quicklook.png")


def generate_lpdr_quicklook() -> None:
    ranges, _, times = _synthetic_coordinates()
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
        fig.savefig(
            ASSET_DIR / "synthetic-lpdr-quicklook.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_elastic_quicklook()
    generate_raman_quicklook()
    generate_lpdr_quicklook()


if __name__ == "__main__":
    main()
