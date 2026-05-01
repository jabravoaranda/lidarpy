from typing import Any
import numpy as np
import math
import pandas as pd
import xarray as xr
from pandas.core.arrays import ExtensionArray
from scipy.constants import physical_constants
from lidarpy.atmo.atmo import transmittance
from lidarpy.atmo.freudenthaler_molecular_properties import molecular_depolarization, f_kbwt, f_kbwc


def retrieve_molecular_extinction_and_backscatter(
    r: np.ndarray[Any, np.dtype[np.float64]],
    temperature: np.ndarray[Any, np.dtype[np.float64]],
    pressure: np.ndarray[Any, np.dtype[np.float64]],
    wavelength: float,
    component: str = "ideal",  
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    r"""Function that gets as input the altitude in [m], the temperature in [K], the pressure in [Pa], and the wavelength in [nm].
    Args:
       - r (np.ndarray[np.float64]): altitude array [m] `lidarpy.atmo.atmo`.
       - temperature (float | np.ndarray[float]): surface temperature or temperature profile [$K$]. If temperature is float, scaled Standard Atmosphere is used.
       - pressure (float | np.ndarray[float]): surface pressure or pressure profile [$Pa$]. If pressure is float, scaled Standard Atmosphere is used.
       - wavelength (float): wavelength [$nm$]
       - component (str, optional): consider the king factor in the molecular lidar ratio (option: `ideal` | `total` | `cabannes`). If idealm $8\pi/3$ otherwise, according to Freudenthaler et al., 2018). Defaults to "ideal".

    Returns:
        tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.float64]: alfasca: Rayleigh extinction [$m^{-1}$], betasca: Rayleigh backscatter [$m^{-1}sr^{-1}$], molecular_lidar_ratio: molecular lidar ratio [$sr$]

    References:
       - Rayleigh Extinction using the King factor [$km^{-1}$]
       - See: Bodhaine et al. 1999. "On Rayleigh Optical Depth Calculations"
       - Freudenthaler et al. 2018. "EARLINET Quality Assurance Tools".
    """                

    # Input units: r [m], T0 [Kelvin], P0 [Pa], lambda [nm]
    # Conversion to the units of this function:
    r = (r / 1e3).astype(np.float64)  # Convert altitude to [km]
    # Altitude is in m, eventhough they say it should be in km

    # pressure = (pressure / 100)
    pressure = (pressure / 100).astype(np.float64)  # Convert pressure to [mb]
    temperature = temperature.astype(np.float64)  # Convert temperature to [K]
    NA = physical_constants["Avogadro constant"][0]  # [mol-1]
    Ns = np.array(NA / 22.4141 * 273.15 / 1013.25 * 1e3 *(pressure / temperature), dtype=np.float64) # [molecules/m3] P[mb], T[K] #Convert to float64 to avoid overflow error in Ns**2
        
    # COMPUTE RAYLEIGH EXTINCTION [km-1]. Eq.(22)-(23)
    lab = wavelength * 1e-3

    # Refractive index. Eq.(21) given at 288.15 K, 1013.25 mb, 360 ppm CO2
    n = 1 + pressure / 1013.25 * 288.15 / temperature * 1e-8 * (
        8060.77 + 2481070 / (132.274 - lab ** (-2)) + 17456.3 / (39.32957 - lab ** (-2))
    )

    # F-factor or King Factor. Eq.(23)(5)(6)
    kingf = (
        78.084 * (1.034 + 3.17e-4 / lab**2)
        + 20.946 * (1.096 + 1.385e-3 / lab**2 + 1.448e-4 / lab**4)
        + 0.934 * 1.00
        + 0.036 * 1.15
    ) / (78.084 + 20.946 + 0.934 + 0.036)

    # Scattering cross section [m2/molecule]
    sigma = (
        8
        * math.pi**3
        * (n**2 - 1) ** 2
        / (3 * (wavelength * 1e-9) ** 4 * Ns**2)
        * kingf
    )
    sigma = 8 * math.pi ** 3 * (n ** 2 - 1) ** 2 / (3 * (wavelength * 1e-9) ** 4 * Ns**2) * kingf

    if component == "total":
        lr_function = f_kbwt
    elif component == "cabannes":
        lr_function = f_kbwc
    elif component == "ideal":
        lr_function = lambda x: 1.0
    else:
        raise ValueError(f"{component} not found.")
    molecular_lidar_ratio = (8.*np.pi/3.) * lr_function(wavelength)
    
    # Extinction [m-1]
    alfasca = sigma * Ns
    # betasca = alfasca / 8.37758041 #From Adolfo's code
    betasca = alfasca / molecular_lidar_ratio

    return alfasca, betasca, np.full_like(alfasca, molecular_lidar_ratio)

def molecular_properties(
    wavelength: float,
    pressure: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray | pd.Series,
    temperature: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray | pd.Series,
    heights: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray | pd.Series,
    times: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    component: str = "ideal",
) -> xr.Dataset:
    """Optical molecular properties of the atmosphere.
    Args:

        - wavelength (float): wavelength of our desired beta molecular attenuated
        - pressure (np.ndarray[Any, np.dtype[np.float64]]): pressure profile
        - temperature (np.ndarray[Any, np.dtype[np.float64]]): temperature profile
        - heights (np.ndarray[Any, np.dtype[np.float64]]): height profile
        - times (np.ndarray[Any, np.dtype[np.float64]] | None, optional): time array. Defaults to None. Note: this is not used in the calculation yet. In development.
        - component (str, optional): _description_. Defaults to "ideal".

    Returns:

        - xr.Dataset: molecular backscatter profile [molecular_beta], molecular extinction profile [molecular_alpha], molecular lidar ratio [molecular_lidar_ratio], molecular depolarization ratio [molecular_depolarization_ratio]
    """
    if isinstance(pressure, ExtensionArray):
        pressure = pressure.to_numpy()
    if isinstance(temperature, ExtensionArray):
        temperature = temperature.to_numpy()
    if isinstance(heights, ExtensionArray):
        heights = heights.to_numpy()

    if isinstance(pressure, pd.Series):
        pressure = np.array(pressure.values)
    if isinstance(temperature, pd.Series):
        temperature = np.array(temperature.values)
    if isinstance(heights, pd.Series):        
        heights = np.array(heights.values)        

    # molecular backscatter and extinction #
    molecular_extinction, molecular_backscatter, molecular_lidar_ratio = retrieve_molecular_extinction_and_backscatter(heights, temperature, pressure, wavelength, component=component)
    
    #TODO: molecular depolarization ratio defined as ideal? 
    molecular_depolarization_ratio = molecular_depolarization(wavelength, component='cabannes')

    attenuated_molecular_backscatter = (
            molecular_backscatter
            * transmittance(molecular_extinction, heights=heights) ** 2
        )


    if times is None:
        mol_properties = xr.Dataset(
            {
                "molecular_beta": (["range"], molecular_backscatter),
                "molecular_alpha": (["range"], molecular_extinction),
                "molecular_lidar_ratio": (["range"], molecular_lidar_ratio),
                "molecular_depolarization": ([], molecular_depolarization_ratio),
                "atten_molecular_beta": (["range"], attenuated_molecular_backscatter),
            },
            coords={"range": heights},
        )
    else:
        mol_properties = xr.Dataset(
            {
                "molecular_beta": (["time", "range"], molecular_backscatter),
                "molecular_alpha": (["time", "range"], molecular_extinction),          
                "atten_molecular_beta": (["time", "range"], attenuated_molecular_backscatter),      
                "molecular_lidar_ratio": (["time"], molecular_lidar_ratio),
                "molecular_depolarization": (["time"], molecular_depolarization_ratio)

            },
            coords={
                "time": times,
                "range": heights,
            },  # FIXME: time is not used in the calculation yet
        )
    return mol_properties
