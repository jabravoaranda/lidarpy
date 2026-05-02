from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

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
MOLECULAR_REFERENCE = (5500.0, 5900.0)


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
    xlim: tuple[float, float] | None = None,
    legend: bool = False,
) -> None:
    height_km = ranges / 1000.0

    ax.set_facecolor("white")
    ax.plot(truth, height_km, color="#005f73", linewidth=2.2, label="Synthetic truth")
    ax.plot(
        retrieved,
        height_km,
        color="#d00000",
        linewidth=2.0,
        linestyle="--",
        label="Retrieved",
    )
    ax.axhspan(0.6, 2.4, color="#005f73", alpha=0.045, label="Validated range")
    ax.axhspan(
        MOLECULAR_REFERENCE[0] / 1000.0,
        MOLECULAR_REFERENCE[1] / 1000.0,
        color="#d00000",
        alpha=0.04,
        label="Molecular reference",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Height (km)")
    ax.set_ylim(0.0, 6.0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="lower left", fontsize=8, frameon=True)


def _particle_depolarization_from_volume(
    volume_depolarization: np.ndarray,
    molecular_depolarization: np.ndarray,
    backscattering_ratio: np.ndarray,
) -> np.ndarray:
    numerator = (
        volume_depolarization
        * (1.0 + molecular_depolarization)
        * backscattering_ratio
        - molecular_depolarization * (1.0 + volume_depolarization)
    )
    denominator = (
        (1.0 + molecular_depolarization) * backscattering_ratio
        - (1.0 + volume_depolarization)
    )
    return numerator / denominator


def _synthetic_particle_depolarization() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.arange(30.0, 6000.0, 30.0)
    _, _, params = synthetic_signals_despo(
        ranges,
        apply_overlap=False,
        number_of_initial_nan_values=0,
    )
    volume_depolarization = (
        np.asarray(params["signal_perpendicular_path"])
        / np.asarray(params["signal_parallel_path"])
    )
    particle_depolarization = _particle_depolarization_from_volume(
        volume_depolarization,
        np.asarray(params["despolarization_molecular"]),
        np.asarray(params["backscattering_ratio"]),
    )
    return (
        ranges,
        np.asarray(params["despolarization_particle"]),
        particle_depolarization,
    )


def generate_retrieval_validation_figure() -> None:
    ranges, elastic, raman, params = _synthetic_retrieval_profiles()
    rcs = signal_to_rcs(elastic, ranges)

    klett_beta = klett_rcs(
        rcs,
        ranges,
        params["molecular_beta"],
        reference=MOLECULAR_REFERENCE,
        lr_part=50.0,
        lr_mol=float(8 * np.pi / 3),
        beta_aer_ref=_reference_beta(params, ranges, MOLECULAR_REFERENCE),
    )

    forward_beta = iterative_beta_forward(
        rcs,
        calibration_factor=1e11,
        range_profile=ranges,
        params=params,
        lr_part=50.0,
        start_height=None,
        height_top=5900.0,
    )

    raman_reference = MOLECULAR_REFERENCE
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

    lpdr_ranges, lpdr_truth, lpdr_retrieved = _synthetic_particle_depolarization()

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 6.8), sharey=True)
    _plot_truth_vs_retrieved(
        axes[0],
        ranges,
        np.asarray(params["particle_beta"]) * 1e6,
        np.asarray(klett_beta) * 1e6,
        "Klett backscatter",
        r"Beta (Mm$^{-1}$ sr$^{-1}$)",
        legend=True,
    )
    _plot_truth_vs_retrieved(
        axes[1],
        ranges,
        np.asarray(params["particle_beta"]) * 1e6,
        np.asarray(forward_beta) * 1e6,
        "Iterative elastic backscatter",
        r"Beta (Mm$^{-1}$ sr$^{-1}$)",
    )
    _plot_truth_vs_retrieved(
        axes[2],
        ranges,
        np.asarray(params["particle_alpha"]) * 1e6,
        np.asarray(raman_alpha) * 1e6,
        "Raman extinction",
        r"Alpha (Mm$^{-1}$)",
    )
    _plot_truth_vs_retrieved(
        axes[3],
        ranges,
        np.asarray(params["particle_beta"]) * 1e6,
        np.asarray(raman_beta) * 1e6,
        "Raman backscatter",
        r"Beta (Mm$^{-1}$ sr$^{-1}$)",
    )
    _plot_truth_vs_retrieved(
        axes[4],
        lpdr_ranges,
        lpdr_truth,
        lpdr_retrieved,
        "Particle depolarization",
        "LPDR (#)",
        xlim=(0.0, 0.5),
    )

    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle("Synthetic truth vs retrieved aerosol properties", y=1.04)
    fig.tight_layout()
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
