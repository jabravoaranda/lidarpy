import pandas as pd
import datetime as dt


def create_calibration_dataframe_alh():
    # Dictionary organized by wavelengths
    depoCalib = {"ff": {}, "nf": {}}

    depoCalib["ff"]["355"] = pd.DataFrame(
        columns=["date", "eta_an", "eta_pc", "GR", "GT", "HR", "HT", "K"]
    )
    depoCalib["nf"]["355"] = pd.DataFrame(
        columns=["date", "eta_an", "eta_pc", "GR", "GT", "HR", "HT", "K"]
    )
    depoCalib["nf"]["532"] = pd.DataFrame(
        columns=["date", "eta_an", "eta_pc", "GR", "GT", "HR", "HT", "K"]
    )

    dict0 = {
        "date": dt.datetime(2023, 3, 2),
        "eta_an": 1.4664,
        "eta_pc": 1.2291,
        "GR": 2.28007,
        "GT": 2.36168,
        "HR": 2.27859,
        "HT": -2.36014,
        "K": 1.002,
    }  # Considering y = + 1 i.e., parallel in T.
    dict1 = {
        "date": dt.datetime(2023, 3, 2),
        "eta_an": 1.4383,
        "eta_pc": 1.336,
        "GR": 2.15778,
        "GT": 1.83852,
        "HR": 2.15241,
        "HT": -1.83393,
        "K": 1.002,
    }  # Considering y = - 1 i.e., parallel in R.
    depoCalib["ff"]["355"].loc[0] = pd.Series(dict1)
    depoCalib["nf"]["355"].loc[0] = pd.Series(dict1)
    depoCalib["nf"]["532"].loc[0] = pd.Series(dict0)
    depoCalib["ff"]["355"] = depoCalib["ff"]["355"].set_index("date")
    depoCalib["nf"]["355"] = depoCalib["nf"]["355"].set_index("date")
    depoCalib["nf"]["532"] = depoCalib["nf"]["532"].set_index("date")
    return depoCalib
