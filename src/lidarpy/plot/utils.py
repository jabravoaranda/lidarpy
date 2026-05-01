from pathlib import Path
import numpy as np
from matplotlib import colors
from loguru import logger
from typing import Literal

from lidarpy.general_utils.io import read_yaml

BoundsType = tuple[float, float] | Literal["auto", "limits", "from_info"]


def get_norm(
    rcs: np.ndarray,
    scale_bounds: BoundsType,
    color_resolution: int = 128,
    lidar_name: str | None = None,
    channel: str | None = None,
) -> tuple[colors.BoundaryNorm, np.ndarray]:
    """Get the color normalization and bounds for the colorbar.

    Args:

        - rcs (np.ndarray): Range corrected signal.
        - scale_bounds (BoundsType): Bounds for the colorbar.
        - color_resolution (int, optional): Colorbar resolution. Defaults to 128.

    Returns:

        - tuple[matplotlib.colors.BoundaryNorm, np.ndarray]: Color normalization and bounds.
    """
    finite_rcs = rcs[np.isfinite(rcs)]
    if finite_rcs.size == 0:
        raise ValueError("rcs must contain at least one finite value")

    match scale_bounds:
        case "auto":
            bounds = np.linspace(0, finite_rcs.max() * 0.6, color_resolution)
        case "limits":
            bounds = np.linspace(finite_rcs.min(), finite_rcs.max(), color_resolution)
        case "from_info":
            if lidar_name is None or channel is None:
                raise ValueError("lidar_name and channel must be provided using option `from_info`.")
            PLOT_INFO = read_yaml(Path(__file__).parent / "info.yml")
            vmin = PLOT_INFO["limits"]["Vmin"][lidar_name][channel]
            vmax = PLOT_INFO["limits"]["Vmax"][lidar_name][channel]
            bounds = np.linspace(vmin, vmax, color_resolution)
        case _:
            bounds = np.linspace(*scale_bounds, color_resolution)

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=2**8, clip=False)
    logger.debug(f"Color bounds min - max: {bounds.min()} - {bounds.max()}")
    return norm, bounds
