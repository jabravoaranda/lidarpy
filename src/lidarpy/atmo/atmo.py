"""
Tomado prestado de lidar_processing.lidar_processing.helper_functions
"""
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.constants import physical_constants


def _standard_atmosphere(
    altitude: float,
    temperature_surface: float = 288.15,
    pressure_surface: float = 101325.0,
) -> tuple[float, float, float]:
    """Calculation of Temperature and Pressure in (scaled) Standard Atmosphere at a given height.

    Args:
        - altitude (float): The altitude above sea level in meters.
        - temperature_surface (float, optional): Temperature at surface level. Defaults to 288.15.
        - pressure_surface (float, optional): Pressure at surface level. Defaults to 101325.0.

    Returns:
        - tuple[float, float, float]: pressure, temperature, and density values.

    References:
        - http://home.anadolu.edu.tr/~mcavcar/common/ISAweb.pdf
    """    

    # Dry air specific gas constant. (J * kg^-1 * K^-1)
    R = 287.058

    g = 9.8  # m/s^2

    # Temperature calculation.
    if altitude < 11000:
        temperature = temperature_surface - 6.5 * altitude / 1000.0
    else:
        temperature = temperature_surface - 6.5 * 11000 / 1000.0
    # Pressure calculation.
    if altitude < 11000:
        pressure: float = (
            pressure_surface * (1 - (0.0065 * altitude / temperature_surface)) ** 5.2561
        )
    else:
        # pressure = pressure_surface*((temperature/scaled_T[idx])**-5.2199))\
        #                       *np.exp((-0.034164*(_height - z_tmp))/scaled_T[idx])
        tropopause_pressure = (
            pressure_surface * (1 - (0.0065 * 11000 / temperature_surface)) ** 5.2561
        )
        tropopause_temperature = temperature
        pressure = tropopause_pressure * np.exp(
            -(altitude - 11000) * (g / (R * tropopause_temperature))
        )

    #number concentration.
    density = number_concentration_from_meteo(pressure=pressure, temperature=temperature, atmospheric_component="total" )

    return pressure, temperature, density # type: ignore


def standard_atmosphere(
    altitude: np.ndarray[Any, np.dtype[np.float64]],
    temperature_surface: float = 288.15,
    pressure_surface: float = 101325.0,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Calculation of Temperature and Pressure Profiles in Standard Atmosphere.

    Args:
        - altitude (np.ndarray[Any, np.dtype[np.float64]]): _description_
        - temperature_surface (float, optional): _description_. Defaults to 288.15.
        - pressure_surface (float, optional): _description_. Defaults to 101325.0.

    Returns:
        - tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]: pressure, temperature, density
    """

    pressure, temperature, density = (
        np.ones(len(altitude),dtype=np.float64),
        np.ones(len(altitude),dtype=np.float64),
        np.ones(len(altitude),dtype=np.float64),
    )
    for idx, altitude_ in enumerate(altitude):
        pressure[idx], temperature[idx], density[idx] = _standard_atmosphere(
            altitude=altitude_,
            temperature_surface=temperature_surface,
            pressure_surface=pressure_surface,
        )
    return pressure, temperature, density


def generate_meteo_profiles(
    heights: np.ndarray[Any, np.dtype[np.float64]] | pd.Series,
    pressure: float | np.ndarray[Any, np.dtype[np.float64]] | pd.Series | None = None,
    temperature: float | np.ndarray[Any, np.dtype[np.float64]] | pd.Series | None = None,
) -> pd.DataFrame:
    """If Pressure and Temperature are floats, it provides scaled standard atmosphere using P,T as surface values. If vectors, scaled standard atmosphere if used on the top to fullfil them up to the heights vector provided.

    Args:
        - pressure (float | np.ndarray[Any, np.dtype[np.float64]]): pressure
        - temperature (float | np.ndarray[Any, np.dtype[np.float64]]): temperature
        - heights (np.ndarray[Any, np.dtype[np.float64]]): range vector

    Returns:
        - pd.DataFrame: `height`, `pressure`, and `temperature` data
    """

    P = pressure
    T = temperature

    if isinstance(heights, pd.Series):
        heights = np.array(heights.values)
    if isinstance(P, pd.Series):
        P = np.array(P.values)
    if isinstance(T, pd.Series):
        T = np.array(T.values)

    if P is None and T is None:
        # standard atmosphere profile:
        extended_P, extended_T, _ = standard_atmosphere(heights)
    elif isinstance(P, float) and isinstance(T, float):
        # standard atmosphere profile:
        extended_P, extended_T, _ = standard_atmosphere(
            heights, pressure_surface=P, temperature_surface=T
        )
    elif isinstance(P, np.ndarray) and isinstance(T, np.ndarray):
        # standard atmosphere profile:
        Psa, Tsa, _ = standard_atmosphere(heights)

        if P.size == Psa.size:  # if they are the same size, we leave them be
            extended_P = P
        elif (
            P.size > Psa.size
        ):  # If our pressure vector is bigger than 'heights', we make it the same size
            maxsa = Psa.size
            extended_P = np.ones(heights.size) * np.nan
            for i in range(maxsa):
                extended_P[i] = P[i]
        else:  # if we don't have enough data to make a full pressure profile
            extended_P = P
            maxh = P.size  # number of the last data
            for i in range(Psa.size - maxh):
                extended_P = np.append(
                    extended_P, Psa[maxh + i]
                )  # we use standard atmosphere as our pressure profile

        if T.size == Tsa.size:  # if they are the same size, we leave them be
            extended_T = T
        elif (
            T.size > Tsa.size
        ):  # If our temperature vector is bigger than 'heights', we make it the same size
            maxsa = Tsa.size
            extended_T = np.ones(heights.size) * np.nan
            for i in range(maxsa):
                extended_T[i] = T[i]
        else:  # if we don't have enough data to make a full temperature profile
            extended_T = T
            maxh = T.size  # number of the last data
            for i in range(Tsa.size - maxh):
                extended_T = np.append(
                    extended_T, Tsa[maxh + i]
                )  # we use standard atmosphere as our temperature profile
    else:
        raise ValueError("Pressure and Temperature must be both floats or vectors.")

    atmospheric_profiles = pd.DataFrame(
        {
            "height": heights,
            "temperature": extended_T,
            "pressure": extended_P,
        }
    )

    return atmospheric_profiles


def transmittance(
    alpha: np.ndarray[Any, np.dtype[np.float64]],
    heights: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Transmittance
    .. math:: 
        T= exp[-integral{\\alpha*dz}_[0,z]]

    Args:
        - alpha (np.ndarray[Any, np.dtype[np.float64]]): extinction coefficient profile.
        - heights (np.ndarray[Any, np.dtype[np.float64]]): heights profile.

    Returns:
        - np.ndarray[Any, np.dtype[np.float64]]: transmittance.
    """    

    delta_height = float(np.median(np.diff(heights)))
    integrated_extinction = cumulative_trapezoid(alpha, initial=0, dx=delta_height)
    return np.exp(-integrated_extinction)


def attenuated_backscatter(
    backscatter: np.ndarray[Any, np.dtype[np.float64]],
    transmittance: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate Attenuated Backscatter

    Args:
        - backscatter ([type]): Backscattering coefficient profile.
        - transmittance ([type]): Transmittance profile.
    """

    return backscatter * transmittance**2

def number_concentration_from_meteo(
    pressure: float | np.ndarray[Any, np.dtype[np.float64]],
    temperature: float | np.ndarray[Any, np.dtype[np.float64]],
    atmospheric_component: str = "total",
) -> Union[float, np.ndarray[Any, np.dtype[np.float64]]]:
    r"""Calculate the number density for a given temperature and pressure. This method does not take into account the compressibility of air.

    Args:
        - pressure (float | np.ndarray): Pressure in Pa.
        - temperature (float | np.ndarray): Temperature in K.
        - atmospheric_component (str, optional): Atmospheric component (molecules). Defaults to "total" containing all the components.

    Returns:
        - np.ndarray[Any, np.dtype[np.float64]]: Number density of the atmosphere [$molecules/m^{-3}$]
    
    References:
        - Tab.1, p.1857. Constituents and mean molecular weight of dry air.
    """    

    # dictionary of molecular weights
    molecular_weights = {
        "nitrogen": 78.084e-2,
        "oxigen": 20.946e-2,
        "argon": 0.934e-2,
        "neon": 1.818e-5,
        "helium": 5.24e-6,
        "kripton": 1.14e-6,
        "hydrogen": 5.80e-7,
        "xenon": 9.00e-8,
        "CO2": 0.0360e-2,
        "CH4": 1.6e-6,
        "NO2": 5e-7,
        "total": 100.0,
    }        
    NA = physical_constants['Avogadro constant'][0] #['mol^-1']
    R  = physical_constants['molar gas constant'][0] #[J/(mol·K)]
    return NA * (molecular_weights[atmospheric_component] / R) * (pressure / temperature)

def extrapolate_aod(
    wv1: float,
    wv2: float,
    aod2: np.ndarray[Any, np.dtype[np.float64]],
    ae: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    r"""Extrapolates AOD at given wavelength using AOD at near wavelength and an appropriate Angstrom Exponent
    The Equation
    .. math::
        aod_{\\lambda_1} = aod_{\\lambda_2} \\cdot \\dfrac{\\lambda_1}{\\lambda_2} \\cdot AE(\\lambda_x - \\lambda_y)

    Args:

        - wv1 (float): wavelength 1
        - wv2 (float): wavelength 2
        - aod2 (np.ndarray[Any, np.dtype[np.float64]]): aerosol optical depth
        - ae (float): Angstrom exponent coefficient

    Returns:
    
        - np.ndarray[Any, np.dtype[np.float64]]: extrapolated AOD
    """

    return aod2 * (wv1 / wv2) ** (-ae)


def interpolate_aod(
    wv_arr: float, aod_arr: np.ndarray[Any, np.dtype[np.float64]], wv0: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Fit log_wv, log_aod to 2nd-order Polynomial [O'Neill et al., 2001]

    Args:
        - wv_arr (float): wavelength
        - aod_arr (np.ndarray[Any, np.dtype[np.float64]]): aerosol optical depth
        - wv0 (float): target wavelength

    Returns:
        - np.ndarray[Any, np.dtype[np.float64]]: interpolated AOD
    """

    y = np.log(aod_arr)
    x = np.log(wv_arr)

    coeff = np.polyfit(x, y, 2, full=True)
    # rr = np.sum( (y - np.polyval(coeff[0], x)) **2) # residuals
    if coeff[1][0].squeeze() < 0.1:
        aod0 = np.exp(np.polyval(coeff[0], np.log(wv0)))
    else:
        raise ValueError("Fit not appropiate.")

    return aod0


def calculate_angstrom_exponent(
    wv1: float, wv2: float, aod1: float | np.ndarray, aod2: float | np.ndarray
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    """Retrieve Ansgstrom exponent using:
    .. math::
        aod_{\\lambda_1} = aod_{\\lambda_2} \\cdot \\dfrac{\\lambda_1}{\\lambda_2} 

    Args:

        - wv1 (float): first wavelength
        - wv2 (float): second wavelength
        - aod1 (float | np.ndarray): AOD at first wavelength
        - aod2 (float | np.ndarray): AOD at second wavelength

    Returns:

        - float | np.ndarray: angstrom exponent coefficient
    """
    return -(np.log(aod1 / aod2)) / (np.log(wv1 / wv2))
