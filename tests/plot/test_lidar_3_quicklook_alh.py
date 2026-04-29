from pathlib import Path

import xarray as xr

from lidarpy.plot.quicklook import quicklook_from_file


def _default_quicklook_channel(dataset_path: Path) -> str:
    with xr.open_dataset(dataset_path) as dataset:
        signal_vars = [name for name in dataset.data_vars if name.startswith("signal_")]
    assert signal_vars
    if "signal_1064fta" in signal_vars:
        return "1064fta"
    return signal_vars[0].split("signal_", 1)[1]


def test_quicklook_from_converted_alhambra_rs(alhambra_rs_nc: Path, tmp_path: Path):
    channel = _default_quicklook_channel(alhambra_rs_nc)
    output_dir = tmp_path / "quicklooks"
    output_dir.mkdir()

    quicklook_from_file(
        alhambra_rs_nc,
        channels=[channel],
        output_dir=output_dir,
        scale_bounds="auto",
        smoothing_method="none",
        apply_dc=False,
    )

    output_files = list(output_dir.glob(f"quicklook_alh_{channel}_*.png"))
    assert len(output_files) == 1
    assert output_files[0].stat().st_size > 0
