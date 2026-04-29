from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset  # type: ignore
from scipy.interpolate import interp1d

from lidarpy.general_utils.utils import parse_datetime

API_URL = "https://cloudnet.fmi.fi/api"



def get_ecmwf_day(
    target_datetime: datetime | str, times: np.ndarray, heights: np.ndarray, base_query_params = {"site": "granada"}
) -> xr.Dataset:
    """Fetches the ECMWF API to retrieve temperature and pressure data

    Args:

        - target_datetime (dt.datetime | str): A data, hour doesn't affect


    Returns:

        - xr.Dataset: x array dataset with variables: ["pressure", "temperature", "height", "time"]

    Raises:

        - FileNotFoundError: There is no available data in the ECMWF data
    """
    _date = parse_datetime(target_datetime)

    response_filename = requests.get(
        f"{API_URL}/model-files",
        params={**base_query_params, "date": _date.date().isoformat()},
    )

    if not response_filename.ok:
        raise RuntimeError(
            f"Request returned code {response_filename.status_code}: {response_filename.text}"
        )

    response_filename = response_filename.json()

    if len(response_filename) == 0:
        raise FileNotFoundError("No data found for the given day")

    file = requests.get(response_filename[0]["downloadUrl"])
    nc = Dataset("ecmwf.nc", memory=file.content)
    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(nc), decode_timedelta=True)  # type: ignore

    if heights is None:
        heights = np.arange(2666) * 7.5

    d_height = np.tile(heights, (25, 1))
    pressure = np.empty_like(d_height)
    temperature = np.empty_like(d_height)

    hours = np.arange(25)

    for hour in hours:
        pressure[hour, :] = np.interp(
            heights, nc["height"][hour, :], nc["pressure"][hour, :]
        )
        temperature[hour, :] = np.interp(
            heights, nc["height"][hour, :], nc["temperature"][hour, :]
        )

    pressure = np.apply_along_axis(
        lambda row: np.interp(
            times.astype("float64"), dataset.time.values.astype("float64"), row
        ).T,
        0,
        pressure,
    )

    temperature = np.apply_along_axis(
        lambda row: np.interp(
            times.astype("float64"), dataset.time.values.astype("float64"), row
        ).T,
        0,
        temperature,
    )

    return xr.Dataset(
        {
            "temperature": (["time", "range"], temperature),
            "pressure": (["time", "range"], pressure),
        },
        coords={"range": heights, "time": times},
    )  # type: ignore


def get_ecmwf_temperature_pressure(
    date: datetime | str,
    hour: int | None = None,
    heights: np.ndarray | list | None = None,
    base_query_params = {"site": "granada"}
) -> pd.DataFrame:
    """Fetch the ecmwf API and downloads temperature and pressure for a given hour date and hour (optional and default to 12:00). Returns the information in a DataFrame

    Args:

        - date (dt.datetime | str): datetime or str datetime
        - hour (int | None, optional): If need an specific time for the data retieval. Defaults to None.
        - heights (np.ndarray | list | None, optional): A heights profile . Defaults to None.


    Returns:

        - pd.DataFrame: DataFrame with three columns: height, temperature, pressure

    Raises:

        - FileNotFoundError: If ecmwf model doesn't have data available for that date
        - ValueError: Raises in case ranges input array shape is not as expected
    """

    _date = parse_datetime(date)

    if hour is None:
        _date = _date.replace(hour=12)
    else:
        _date = _date.replace(hour=hour)

    hour_int: int = _date.hour

    response_filename = requests.get(
        f"{API_URL}/model-files",
        params={**base_query_params, "date": _date.date().isoformat()},
    ).json()

    if len(response_filename) == 0:
        raise FileNotFoundError("No data found for the given day")

    file = requests.get(response_filename[0]["downloadUrl"])
    nc = Dataset("ecmwf.nc", memory=file.content)

    height = nc.variables["height"][hour_int]
    temperature = nc.variables["temperature"][hour_int]
    pressure = nc.variables["pressure"][hour_int]

    if heights is None:
        return pd.DataFrame(
            {"height": height, "temperature": temperature, "pressure": pressure}
        )

    if np.ndim(heights) != 1:
        raise ValueError("heights must be 1-dimensinal")

    # interpolations, case when range doesn't match API heights
    temperature_fn = interp1d(
        height, temperature, bounds_error=False, fill_value="extrapolate"  # type: ignore
    )
    pressure_fn = interp1d(
        height, pressure, bounds_error=False, fill_value="extrapolate"  # type: ignore
    )

    return pd.DataFrame(
        {
            "height": heights,
            "temperature": temperature_fn(heights),
            "pressure": pressure_fn(heights),
        }
    )

def get_data_as_ascii(
    target_datetime: datetime,    
    output_dir: Path | str | None = None,
    base_query_params = {"site": "granada"}
) -> Path:
    """ Fetches temperature and pressure data for a given target datetime,
    saves the data in ASCII format to a file, and returns the path of the saved file.

    Args:

        - target_datetime (datetime.datetime): The target date and time.
        - output_dir (Path | str | None, optional): The directory where the output file    will be saved. If not provided, the current working directory is used.


    Returns:

        - Path: The path of the saved ASCII file.

    Raises:

        - ValueError: If the output_dir is not a valid directory.
        - FileNotFoundError: If no data is found for the given day.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not output_dir.exists() or not output_dir.is_dir():
        raise ValueError(f"output_dir must be a valid directory: {output_dir}")

    hour_int: int = target_datetime.hour

    response_filename = requests.get(
        f"{API_URL}/model-files",
        params={**base_query_params, "date": target_datetime.date().isoformat()},
    ).json()

    if len(response_filename) == 0:
        raise FileNotFoundError("No data found for the given day")

    # Fetch temperature and pressure data
    file = requests.get(response_filename[0]["downloadUrl"])
    nc = Dataset(output_dir / "ecmwf.nc", memory=file.content)

    height = nc.variables["height"][hour_int]
    temperature = nc.variables["temperature"][hour_int]
    pressure = nc.variables["pressure"][hour_int]

    # Generate ASCII content
    ascii_content = ""
    for idx, h in enumerate(height):
        t = temperature[idx]
        p = pressure[idx]/100.0
        line = f"{h:.2f}\t{t:.2f}\t{p:.2f}\n"
        ascii_content += line

    # Create filename using target_datetime
    current_datetime = target_datetime.strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"{current_datetime}_ecmwf.txt"

    # Save ASCII content to file
    with open(output_file, "w") as file_:
        file_.write("height[m] temperature[K] pressure[hPa]\n")
        file_.write(ascii_content)

    return output_file