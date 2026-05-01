from enum import Enum
from typing import Tuple, TypedDict
import xarray as xr
from typing_extensions import NotRequired

import numpy as np


class DataType(str, Enum):
    raw = "RAW"
    produt = "PRODUCTS"

    def __str__(self):
        return self.value

    def to_level(self):
        if self == DataType.raw:
            return "0a"
        elif self == DataType.produt:
            return "1a"
        else:
            raise ValueError("Invalid DataType")


class AgoraStation(str, Enum):
    ugr = "UGR"
    sns = "SNS"

    def __str__(self):
        return self.value


class LidarName(str, Enum):
    mhc = "mulhacen"
    alh = "alhambra"
    vlt = "veleta"
    prc = "pericles"
    ipr = "ipral"
    gnl = "generalife"

    def __str__(self):
        return self.value


class MeasurementType(str, Enum):    
    RS = "RS"
    HF = "HF"
    DC = "DC"
    TC = "TC"
    DP = "DP"
    OT = "OT"

    def __str__(self) -> str:
        return self.value


class Telescope(str, Enum):
    xf = "xf"
    ff = "ff"
    nf = "nf"

    def __str__(self) -> str:
        return self.value


class _LidarInfo(TypedDict):
    mulhacen: dict
    alhambra: dict
    veleta: dict


class _LidarMetadata(TypedDict):
    nick2name: dict
    name2nick: dict
    measurement_type: list
    code_telescope_str2number: dict
    code_mode_str2number: dict
    code_polarization_str2number: dict
    code_mode_number2str: dict
    code_polarization_number2str: dict
    emails: list[str]


class LidarInfoType(TypedDict):
    lidars: _LidarInfo
    metadata: _LidarMetadata


class ParamsDict(TypedDict):
    k_lidar: float | Tuple[float, float]
    particle_alpha: np.ndarray
    particle_alpha_raman: NotRequired[np.ndarray]
    particle_beta: np.ndarray
    particle_beta_raman: NotRequired[np.ndarray]
    molecular_alpha: np.ndarray
    molecular_alpha_raman: NotRequired[np.ndarray]
    molecular_beta: np.ndarray
    molecular_beta_raman: NotRequired[np.ndarray]
    particle_accum_ext: np.ndarray
    particle_accum_ext_raman: NotRequired[np.ndarray]
    molecular_accum_ext: np.ndarray
    molecular_accum_ext_raman: NotRequired[np.ndarray]
    attenuated_molecular_backscatter: np.ndarray
    attenuated_molecular_backscatter_raman: xr.DataArray
    transmittance_elastic: np.ndarray
    transmittance_raman: np.ndarray
    overlap: np.ndarray
    angstrom_exponent_fine: float | Tuple[float, float]
    angstrom_exponent_coarse: float | Tuple[float, float]
    particle_angstrom_exponent:float | Tuple[float, float] 
