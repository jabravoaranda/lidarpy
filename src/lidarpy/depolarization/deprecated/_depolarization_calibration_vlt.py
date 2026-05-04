import pandas as pd
import datetime as dt


def create_calibration_dataframe_vlt():
    # Dictionary organized by wavelengths
    depoCalib = {"xf": {}}

    depoCalib["xf"]["355"] = pd.DataFrame(
        columns=["date", "eta_an", "eta_pc", "GR", "GT", "HR", "HT", "K"]
    )

    dict0 = {
        "date": dt.datetime(2021, 12, 1),
        "eta_an": 1.0,
        "eta_pc": 1.0,
        "GR": 1.0,
        "GT": 1,
        "HR": -1.0,
        "HT": 1.0,
        "K": 1.0,
    }  # Considering y = + 1 i.e., parallel in T.

    dict1 = {
        "date": dt.datetime(2021, 12, 1),
        "eta_an": 1.0,
        "eta_pc": 1.0,
        "GR": 1.0,
        "GT": 1,
        "HR": 1.0,
        "HT": -1.0,
        "K": 1.0,
    }  # Considering y = - 1 i.e., parallel in R.
    depoCalib["xf"]["355"].loc[0] = pd.Series(dict0)
    depoCalib["xf"]["355"] = depoCalib["xf"]["355"].set_index("date")
    return depoCalib
