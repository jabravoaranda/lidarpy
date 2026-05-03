import warnings
from typing import Any
from datetime import datetime, timedelta
from functools import cached_property

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from .klett import klett_rcs
from lidarpy.atmo.freudenthaler_molecular_properties import molecular_properties
from lidarpy.general_utils.dates import parse_datetime
from lidarpy.plot.quicklook import quicklook_xarray
from lidarpy.atmo.ecmwf import get_ecmwf_temperature_pressure

try:
    from lidarpy.plot.profile import plot_profile
except ModuleNotFoundError:
    def plot_profile(range_profile, signal_profile):
        plt.plot(signal_profile, range_profile)


warnings.filterwarnings("once")


class RetrievalHelper:
    def __init__(self, preprocessed_data: xr.Dataset, channel: str = "532xpa"):
        if channel in preprocessed_data.channel:
            self._data = preprocessed_data
            self._channel = channel
        else:
            raise ValueError(
                f"Channel {channel} not available in current dataset select another one from: {preprocessed_data.channel}"
            )

        self._k_lidar: float | None = None

    @cached_property
    def wavelength(self) -> int:
        return int("".join(filter(str.isdigit, self.channel)))

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def data(self) -> xr.Dataset:
        return self._data

    @property
    def k_lidar(self) -> float:
        if self._k_lidar is None:
            raise ValueError("k_lidar has not been calibrated")
        return self._k_lidar

    @cached_property
    def rcs(self) -> np.ndarray:
        return self.data[f"signal_{self.channel}"].values * np.tile(
            self.data.range**2, (self.data.time.shape[0], 1)
        )

    def closest_profile(self, date: datetime | str, print_date=False) -> int:
        _date = parse_datetime(date)
        idx = np.searchsorted(self.data.time, np.datetime64(_date))

        date_found = (
            self.data.time[idx].values.astype("M8[ms]").astype("O")
        )  # This is to convert to pydatetime

        distance = date_found - _date

        td = timedelta(minutes=60)

        if (distance > td) or (distance < -td):
            warnings.warn(
                f"Using datetime with more than 1 hour of difference from the input ({date_found})"
            )

        if print_date:
            print(f"Date found: {date_found} at {distance} distance")

        return idx  # type: ignore

    def plot_profile(self, profile_idx: int, rcs: bool = True):
        if rcs:
            plot_profile(self.data.range, self.rcs[profile_idx])
        else:
            raise NotImplementedError()

    def klett_on_profile(self, profile_idx: int) -> Any:
        date: datetime = self.data.time.values.astype("M8[ms]").astype("O")[profile_idx]
        tp_df = get_ecmwf_temperature_pressure(date, heights=self.data.range.values)
        atmo_params = molecular_properties(
            self.wavelength,
            tp_df.pressure,
            tp_df.temperature,
            heights=self.data.range.values,
        )
        # set_trace()
        part_beta = klett_rcs(self.rcs[profile_idx], range_profile=self.data.range.values, beta_mol_profile=atmo_params["molecular_beta"].values)  # type: ignore

        return part_beta

    def quicklook(self) -> None:
        warnings.warn("Work in progress")
        quicklook_xarray(self.data, signal_var=f"signal_{self.channel}")

    def show(self) -> None:
        plt.show()

    def __repr__(self) -> str:
        return f"{self.data.attrs['system']} lidar data with {self.data.time.shape[0]} time measurements between {self.data.time[0]} and {self.data.time[-1]}"  # type: ignore
