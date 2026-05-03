import datetime as dt
import re

import numpy as np
import pandas as pd
from loguru import logger


def numpy_to_datetime(numpy_date):
    """Convert a numpy datetime64 object to a python datetime object."""
    if isinstance(numpy_date, np.ndarray):
        numpy_date = numpy_date[0]

    try:
        timestamp = dt.datetime.utcfromtimestamp(numpy_date.tolist() / 1e9)
    except Exception as e:
        timestamp = None
        print(str(e))

    return timestamp


def datetime_np2dt(numpy_date):
    """Convert a numpy datetime64 object to a python datetime object."""
    try:
        timestamp = pd.Timestamp(numpy_date).to_pydatetime()
    except Exception as e:
        timestamp = None
        logger.error(str(e))
        raise e
    return timestamp


def str_to_datetime(date_str: str) -> dt.datetime:
    """Parse supported compact date strings into ``datetime``."""
    assert isinstance(date_str, str), "date_str must be String Type"

    formats = [
        (r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}", "%Y%m%dT%H%M%S"),
        (r"\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}", "%Y%m%d_%H%M%S"),
        (r"\d{4}\d{2}\d{2}T\d{2}\d{2}", "%Y%m%dT%H%M"),
        (r"\d{4}\d{2}\d{2}_\d{2}\d{2}", "%Y%m%d_%H%M"),
        (r"\d{4}\d{2}\d{2}T\d{2}", "%Y%m%dT%H"),
        (r"\d{4}\d{2}\d{2}_\d{2}", "%Y%m%d_%H"),
        (r"\d{4}\d{2}\d{2}", "%Y%m%d"),
        (r"\d{4}\d{2}", "%Y%m"),
        (r"\d{4}", "%Y"),
    ]

    idx = 0
    match = False
    date_format = ""
    while not match:
        if idx < len(formats):
            candidate = re.search(formats[idx][0], date_str)
            if candidate is not None:
                date_format = formats[idx][1]
                match = True
            else:
                idx += 1
        else:
            match = True

    if date_format is not None:
        try:
            date_dt = dt.datetime.strptime(date_str, date_format)
        except Exception as e:
            print(f"{date_str} has more complex format than found ({date_format})")
            raise NotImplementedError(e.with_traceback)
    else:
        raise RuntimeError(f"Cannot understand the format of {date_str}")

    return date_dt


def parse_datetime(
    date: dt.datetime | dt.date | str | np.datetime64 | pd.DatetimeIndex | pd.Timestamp,
) -> dt.datetime:
    """Cast supported string/date objects to python ``datetime``."""
    if isinstance(date, dt.datetime):
        return date
    if isinstance(date, dt.date):
        return dt.datetime(date.year, date.month, date.day)
    if isinstance(date, str):
        try:
            return dt.datetime.fromisoformat(date)
        except ValueError:
            return str_to_datetime(date)
    if isinstance(date, np.datetime64):
        return datetime_np2dt(date)
    if isinstance(date, pd.DatetimeIndex):
        return date.to_pydatetime()[0]
    if isinstance(date, pd.Timestamp):
        return date.to_pydatetime()
    raise ValueError(f"{date} is not a valid date")
