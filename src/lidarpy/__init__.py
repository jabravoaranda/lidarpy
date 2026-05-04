from . import (
    atmo,
    depolarization,
    general_utils,
    nc_convert,
    plot,
    preprocessing,
    retrieval,
    scc,
    utils,
)

__version__ = "0.1.0"

__all__ = [
    "preprocessing",
    "retrieval",
    "scc",
    "depolarization",
    "plot",
    "utils",
    "nc_convert",
    "atmo",
    "general_utils",
]


__doc__ = """
    The top of the lidar module that is compatible (at least) with GFAT lidars: "Veleta", "Mulhacén" and "Alhambra"
"""
