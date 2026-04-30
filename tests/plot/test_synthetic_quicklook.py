from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

from lidarpy.plot.quicklook import quicklook_xarray
from lidarpy.retrieval.synthetic.generator import synthetic_signals_2D


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

    fig, _ = quicklook_xarray(
        signal,
        lidar_name="SYNTHETIC",
        location="Synthetic",
        is_rcs=True,
        scale_bounds="auto",
        ylims=(0.0, 3000.0),
    )

    try:
        output = BytesIO()
        fig.savefig(output, format="png", dpi=100, bbox_inches="tight")
        assert output.tell() > 0
    finally:
        plt.close(fig)
