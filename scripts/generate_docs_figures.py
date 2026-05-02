from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid

from lidarpy.plot.quicklook import quicklook_lpdr, quicklook_xarray
from lidarpy.retrieval.klett import iterative_beta_forward, klett_rcs
from lidarpy.retrieval.raman import retrieve_backscatter, retrieve_extinction
from lidarpy.retrieval.synthetic.generator import (
    synthetic_raman_signals_2D,
    synthetic_signals,
    synthetic_signals_2D,
    synthetic_signals_despo,
)
from lidarpy.utils.utils import signal_to_rcs


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


def _synthetic_retrieval_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ranges = np.arange(30.0, 6000.0, 30.0)
    elastic, raman, params = synthetic_signals(
        ranges,
        wavelengths=(532, 607),
        k_lidar=(1e11, 1e10),
        ae=1.0,
        lr=50.0,
        apply_overlap=False,
        number_of_initial_nan_values=0,
    )
    if raman is None:
        raise RuntimeError("Synthetic Raman signal was not generated.")
    return ranges, elastic.values, raman.values, params


def _reference_beta(params: dict, ranges: np.ndarray, reference: tuple[float, float]) -> float:
    idx_ref = (ranges >= reference[0]) & (ranges <= reference[1])
    return float(np.nanmean(params["particle_beta"][idx_ref]))


def _plot_truth_vs_retrieved(
    ax,
    ranges: np.ndarray,
    truth: np.ndarray,
    retrieved: np.ndarray,
    title: str,
    xlabel: str,
) -> None:
    height_km = ranges / 1000.0
    validated = (height_km >= 0.6) & (height_km <= 2.4)
    retrieved_in_validated_range = np.where(validated, retrieved, np.nan)

    ax.plot(truth, height_km, color="#106c78", linewidth=2.0, label="Synthetic truth")
    ax.plot(
        retrieved_in_validated_range,
        height_km,
        color="#7b4e16",
        linewidth=1.8,
        linestyle="--",
        label="Retrieved",
    )
    ax.axhspan(0.6, 2.4, color="#106c78", alpha=0.08, label="Validated range")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0.0, 3.0)
    ax.grid(True, alpha=0.25)


def generate_retrieval_validation_figure() -> None:
    ranges, elastic, raman, params = _synthetic_retrieval_profiles()
    rcs = signal_to_rcs(elastic, ranges)

    klett_reference = (3000.0, 3500.0)
    klett_beta = klett_rcs(
        rcs,
        ranges,
        params["molecular_beta"],
        reference=klett_reference,
        lr_part=50.0,
        lr_mol=float(8 * np.pi / 3),
        beta_aer_ref=_reference_beta(params, ranges, klett_reference),
    )

    start_height = 600.0
    start_idx = np.abs(ranges - start_height).argmin()
    initial_particle_optical_depth = cumulative_trapezoid(
        params["particle_alpha"],
        ranges,
        initial=0.0,
    )[start_idx]
    forward_beta = iterative_beta_forward(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        start_height=start_height,
        height_top=2400.0,
        initial_particle_optical_depth=float(initial_particle_optical_depth),
    )

    raman_reference = (4500.0, 5500.0)
    raman_alpha = retrieve_extinction(
        raman,
        ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=raman_reference,
        particle_angstrom_exponent=1.0,
        full_overlap_height=500.0,
    )
    raman_beta = retrieve_backscatter(
        signal_raman=raman,
        signal_emission=elastic,
        extinction_profile=raman_alpha,
        range_profile=ranges,
        wavelengths=(532, 607),
        pressure=params["pressure"],
        temperature=params["temperature"],
        reference=raman_reference,
        particle_angstrom_exponent=1.0,
        beta_part_ref=_reference_beta(params, ranges, raman_reference),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    _plot_truth_vs_retrieved(
        axes[0, 0],
        ranges,
        np.asarray(params["particle_beta"]),
        np.asarray(klett_beta),
        "Klett backscatter",
        "Particle beta",
    )
    _plot_truth_vs_retrieved(
        axes[0, 1],
        ranges,
        np.asarray(params["particle_beta"]),
        np.asarray(forward_beta),
        "Iterative elastic backscatter",
        "Particle beta",
    )
    _plot_truth_vs_retrieved(
        axes[1, 0],
        ranges,
        np.asarray(params["particle_alpha"]),
        np.asarray(raman_alpha),
        "Raman extinction",
        "Particle alpha",
    )
    _plot_truth_vs_retrieved(
        axes[1, 1],
        ranges,
        np.asarray(params["particle_beta"]),
        np.asarray(raman_beta),
        "Raman backscatter",
        "Particle beta",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Synthetic truth vs retrieved aerosol properties", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    try:
        fig.savefig(
            ASSET_DIR / "retrieval-synthetic-validation.png",
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
    generate_retrieval_validation_figure()


if __name__ == "__main__":
    main()
