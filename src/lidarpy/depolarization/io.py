from pathlib import Path

import xarray as xr

from pathlib import Path

from pathlib import Path
from typing import Dict, Union


def ghk_output_reader(filepath: Path) -> Dict[str, Union[float, None]]:
    """It loads in a dictionary the ghk parameters from `filepath`.

    Args:
        filepath (Path): ghk filepath generated for the Volker's software `GHK_0.9.8h_Py3.7.py`.

    Raises:
        FileNotFoundError: filepath not found.
        ValueError: GHK parameters not found in the filepath.

    Returns:
        Dict[str, Union[float, None]]: Dictionary with keys: GR, GT, HR, HT, K1, K2, ..., K7.
    """    
    GHK: Dict[str, Union[float, None]] = {
        "GR": None,
        "GT": None,
        "HR": None,
        "HT": None,
        "K1": None,
        "K2": None,
        "K3": None,
        "K4": None,
        "K5": None,
        "K6": None,
        "K7": None,
    }
    if not filepath.is_file():
        raise FileNotFoundError(f"{filepath} not found.")

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line[0:2] == "GR":
                break
        line = f.readline().replace(" ", "").split(",")
        try:
            (
                GHK["GR"],
                GHK["GT"],
                GHK["HR"],
                GHK["HT"],
                GHK["K1"],
                GHK["K2"],
                GHK["K3"],
                GHK["K4"],
                GHK["K5"],
                GHK["K6"],
                GHK["K7"],
            ) = [float(val) if val else None for val in line]
        except (ValueError, TypeError, IndexError):
            raise ValueError("Unexpected line format encountered.")

    return GHK


