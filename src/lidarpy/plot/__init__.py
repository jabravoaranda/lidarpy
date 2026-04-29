from pathlib import Path
from lidarpy.general_utils.io import read_yaml

PLOT_INFO = read_yaml(Path(__file__).parent.absolute() / "info.yml")

__all__ = ["PLOT_INFO"]
