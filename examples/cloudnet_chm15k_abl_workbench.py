"""Cloudnet CHM15k ABLH detection workbench.

This example mirrors the ABLH retrieval test workflow with a real Cloudnet
ceilometer file: it runs WCT and temporal-variance detection through
``lidarpy.retrieval.ablh.detect_ablh``, writes one ABLH NetCDF per method, and
builds a quicklook from the original Cloudnet data plus the derived ABLH
products.

Run from the repository root with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cloudnet_api_client import APIClient
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lidarpy.retrieval.ablh import detect_ablh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "tests"
    / "data"
    / "RAW"
    / "chm15k_25a1c14a"
    / "20260603_granada_chm15k_25a1c14a.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "cloudnet_chm15k_abl_workbench"
METHODS = ("wct", "temporal_variance")
CLOUDNET_SITE = "granada"
CLOUDNET_DATE = "2026-06-03"
CLOUDNET_FILENAME = "20260603_granada_chm15k_25a1c14a.nc"
CLOUDNET_PRODUCT = "lidar"


def ensure_cloudnet_input(
    input_path: Path,
    *,
    download: bool,
) -> Path:
    """Return an existing Cloudnet file, downloading the default fixture if needed."""
    input_path = input_path.expanduser().resolve()
    if input_path.exists():
        return input_path
    if not download:
        raise FileNotFoundError(
            f"Cloudnet input file not found: {input_path}. "
            "Re-run without --no-download or pass --input to an existing file."
        )
    if input_path.name != CLOUDNET_FILENAME:
        raise FileNotFoundError(
            f"Cloudnet input file not found: {input_path}. "
            "Automatic download is only configured for the default Granada CHM15k file."
        )
    return download_cloudnet_lidar_product(input_path)


def download_cloudnet_lidar_product(target_path: Path) -> Path:
    """Download the default Granada CHM15k lidar product with Cloudnet API client."""
    client = APIClient()
    files = client.files(
        site_id=CLOUDNET_SITE,
        date=CLOUDNET_DATE,
        product_id=CLOUDNET_PRODUCT,
    )
    matching_files = [file for file in files if file.filename == CLOUDNET_FILENAME]
    if not matching_files:
        raise FileNotFoundError(
            f"Cloudnet did not return {CLOUDNET_FILENAME} for "
            f"site={CLOUDNET_SITE}, date={CLOUDNET_DATE}, product={CLOUDNET_PRODUCT}"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_paths = client.download(
        matching_files[0],
        output_directory=target_path.parent,
        progress=True,
        validate_checksum=True,
    )
    downloaded_path = Path(downloaded_paths[0])
    if downloaded_path != target_path:
        downloaded_path.replace(target_path)
    return target_path


def load_cloudnet_for_ablh(
    input_path: Path,
    *,
    variable: str,
    time_frequency: str | None,
) -> xr.Dataset:
    """Load a Cloudnet lidar product and keep only fields needed by ABLH."""
    dataset = xr.open_dataset(input_path)
    if variable not in dataset:
        available = ", ".join(dataset.data_vars)
        raise KeyError(
            f"Variable '{variable}' not found. Available variables: {available}"
        )

    backscatter = dataset[variable].transpose("time", "range")
    if time_frequency:
        backscatter = backscatter.resample(time=time_frequency).median(skipna=True)

    output = xr.Dataset({variable: backscatter})
    if "height" in dataset:
        output["height"] = dataset["height"]
    output.attrs.update(dataset.attrs)
    return output


def run_ceilometer_ablh_workbench(
    input_path: Path,
    output_dir: Path,
    *,
    variable: str = "beta_smooth",
    time_frequency: str | None = "5min",
    min_range: float = 500.0,
    max_range: float = 4000.0,
    wct_threshold: float = 0.02,
    wct_width: float = 300.0,
    temporal_threshold: float = 1e-5,
    temporal_window_minutes: float = 10.0,
) -> tuple[dict[str, Path], Path]:
    """Run WCT and temporal-variance ABLH detection on Cloudnet ceilometer data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_cloudnet_for_ablh(
        input_path,
        variable=variable,
        time_frequency=time_frequency,
    )

    frequency_label = time_frequency or "native"
    range_label = f"r{min_range:g}-{max_range:g}m"
    output_paths: dict[str, Path] = {}
    results: dict[str, xr.Dataset] = {}

    method_options = {
        "wct": {
            "threshold": wct_threshold,
            "wct_width": wct_width,
            "time_window_minutes": temporal_window_minutes,
        },
        "temporal_variance": {
            "threshold": temporal_threshold,
            "wct_width": wct_width,
            "time_window_minutes": temporal_window_minutes,
        },
    }

    for method in METHODS:
        result = detect_ablh(
            dataset,
            input_kind="cloudnet",
            variable=variable,
            method=method,
            min_range=min_range,
            max_range=max_range,
            **method_options[method],
        )
        result.attrs.update(
            {
                "source_cloudnet_file": str(input_path),
                "cloudnet_variable": variable,
                "time_frequency": frequency_label,
                "wct_threshold": float(wct_threshold),
                "wct_width_m": float(wct_width),
                "temporal_threshold": float(temporal_threshold),
                "temporal_window_minutes": float(temporal_window_minutes),
            }
        )
        output_path = (
            output_dir
            / f"{input_path.stem}_{variable}_{frequency_label}_{range_label}_{method}_ablh.nc"
        )
        result.to_netcdf(output_path)
        output_paths[method] = output_path
        results[method] = result

    quicklook_path = (
        output_dir
        / f"{input_path.stem}_{variable}_{frequency_label}_{range_label}_ablh.png"
    )
    plot_cloudnet_ablh_quicklook(
        cloudnet=dataset,
        ablh_results=results,
        variable=variable,
        min_range=min_range,
        max_range=max_range,
        output_path=quicklook_path,
    )
    return output_paths, quicklook_path


def plot_cloudnet_ablh_quicklook(
    *,
    cloudnet: xr.Dataset,
    ablh_results: dict[str, xr.Dataset],
    variable: str,
    min_range: float,
    max_range: float,
    output_path: Path,
) -> None:
    """Plot Cloudnet signal and overlaid ABLH from all requested methods."""
    backscatter = cloudnet[variable].transpose("time", "range")
    ranges = backscatter["range"].values.astype(float)
    range_mask = (ranges >= min_range) & (ranges <= max_range)
    times = backscatter["time"].values
    plot_signal = _log10_positive(backscatter.values[:, range_mask])

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        times,
        ranges[range_mask] / 1000.0,
        plot_signal.T,
        shading="auto",
        cmap="viridis",
    )
    colors = {"wct": "white", "temporal_variance": "tab:red"}
    labels = {
        "wct": "ABLH WCT",
        "temporal_variance": "ABLH temporal variance",
    }
    for method, result in ablh_results.items():
        ax.plot(
            result["time"].values,
            result["ablh"].values / 1000.0,
            color=colors.get(method, "tab:orange"),
            lw=1.6 if method == "temporal_variance" else 2.0,
            label=labels.get(method, method),
        )

    title = cloudnet.attrs.get("title", "Cloudnet ceilometer")
    ax.set_title(f"{title}: ABLH from {variable}")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Range above instrument [km]")
    ax.set_ylim(min_range / 1000.0, max_range / 1000.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"log10({variable})")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _log10_positive(values: np.ndarray) -> np.ndarray:
    positive = np.where(values > 0, values, np.nan)
    finite = np.isfinite(positive)
    if not finite.any():
        raise ValueError("Selected Cloudnet variable has no finite positive data.")
    floor = np.nanpercentile(positive[finite], 1)
    if not np.isfinite(floor) or floor <= 0:
        floor = np.nanmin(positive[finite])
    positive = np.where(np.isfinite(positive), positive, floor)
    return np.log10(np.maximum(positive, floor))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail if the default Cloudnet file is missing instead of downloading it.",
    )
    parser.add_argument(
        "--variable",
        default="beta_smooth",
        choices=["beta_smooth", "beta", "beta_raw"],
        help="Cloudnet backscatter variable used for detection.",
    )
    parser.add_argument(
        "--time-frequency",
        default="5min",
        help="Pandas/xarray resampling frequency. Use 'none' for native time.",
    )
    parser.add_argument("--min-range", type=float, default=500.0)
    parser.add_argument("--max-range", type=float, default=4000.0)
    parser.add_argument("--wct-threshold", type=float, default=0.02)
    parser.add_argument("--wct-width", type=float, default=300.0)
    parser.add_argument("--temporal-threshold", type=float, default=1e-5)
    parser.add_argument("--temporal-window-minutes", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ensure_cloudnet_input(args.input, download=not args.no_download)
    time_frequency = (
        None if args.time_frequency.lower() == "none" else args.time_frequency
    )
    ablh_paths, quicklook_path = run_ceilometer_ablh_workbench(
        input_path=input_path,
        output_dir=args.output_dir,
        variable=args.variable,
        time_frequency=time_frequency,
        min_range=args.min_range,
        max_range=args.max_range,
        wct_threshold=args.wct_threshold,
        wct_width=args.wct_width,
        temporal_threshold=args.temporal_threshold,
        temporal_window_minutes=args.temporal_window_minutes,
    )
    for method, path in ablh_paths.items():
        print(f"Saved {method} ABLH NetCDF: {path}")
    print(f"Saved ABLH quicklook: {quicklook_path}")


if __name__ == "__main__":
    main()
