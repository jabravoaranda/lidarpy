
# Scattering parameters according to Freudenthaler V., 2015.
from typing import Any
import numpy as np
from scipy import interpolate
import xarray as xr
from pandas.core.arrays import ExtensionArray

from lidarpy.atmo.atmo import attenuated_backscatter, transmittance


Fk = {
    308: 1.05574,
    351: 1.05307,
    354.717: 1.05209,
    355: 1.05288,
    386.890: 1.05166,
    400: 1.05125,
    407.558: 1.05105,
    510.6: 1.04922,
    532: 1.04899,
    532.075: 1.04899,
    607.435: 1.04839,
    710: 1.04790,
    800: 1.04763,
    1064: 1.04721,
    1064.150: 1.04721,
}

epsilon = {
    308: 0.25083,
    351: 0.238815,
    354.717: 0.234405,
    355: 0.23796,
    386.890: 0.23247,
    400: 0.230625,
    407.558: 0.229725,
    510.6: 0.22149,
    532: 0.220455,
    532.075: 0.220455,
    607.435: 0.217755,
    710: 0.21555,
    800: 0.214335,
    1064: 0.212445,
    1064.150: 0.212445,
}

Cs = {
    308: 3.6506e-5,  # K/hPa/m
    351: 2.0934e-5,
    354.717: 2.0024e-5,
    355: 1.9957e-5,
    386.890: 1.3942e-5,
    400: 1.2109e-5,
    407.558: 1.1202e-5,
    510.6: 4.4221e-6,
    532: 3.7382e-6,
    532.075: 3.7361e-6,
    607.435: 2.1772e-6,
    710: 1.1561e-6,
    800: 7.1364e-7,
    1064: 2.2622e-7,
    1064.150: 2.2609e-7,
}

BsT = {
    308: 4.2886e-6,
    351: 2.4610e-6,
    354.717: 2.3542e-6,
    355: 2.3463e-6,
    400: 1.4242e-6,
    510.6: 5.2042e-7,
    532: 4.3997e-7,
    532.075: 4.3971e-7,
    710: 1.3611e-7,
    800: 8.4022e-8,
    1064: 2.6638e-8,
    1064.150: 2.6623e-8,
}

BsC = {
    308: 4.1678e-6,
    351: 2.3949e-6,
    354.717: 2.2912e-6,
    355: 2.2835e-6,
    400: 1.3872e-6,
    510.6: 5.0742e-7,
    532: 4.2903e-7,
    532.075: 4.2878e-7,
    710: 1.3280e-7,
    800: 8.1989e-8,
    1064: 2.5999e-8,
    1064.150: 2.5984e-8,
}

BsC_parallel = {
    308: 4.15052184e-6,
    351: 2.38547616e-06,
    354.717: 2.28368241e-06,
    355: 2.27451222e-06,
    400: 1.38198631e-06,
    510.6: 5.05563542e-07,
    532: 4.27459520e-07,
    532.075: 4.27219387e-07,
    710: 1.32322062e-07,
    800: 8.16989147e-08,
    1064: 2.59074156e-08,
    1064.150: 2.58925276e-08,
}

BsC_perpendicular = {
    308: 1.72550768e-08,
    351: 9.44466863e-09,
    354.717: 8.87554356e-09,
    355: 8.97326485e-09,
    400: 5.28492464e-09,
    510.6: 1.85714694e-09,
    532: 1.56293632e-09,
    532.075: 1.56205831e-09,
    710: 4.73100855e-10,
    800: 2.90465461e-10,
    1064: 9.13006514e-11,
    1064.150: 9.12481844e-11,
}

KbwT = {
    308: 1.01610,
    351: 1.01535,
    354.717: 1.01530,
    355: 1.01530,
    400: 1.01484,
    510.6: 1.01427,
    532: 1.01421,
    532.075: 1.01421,
    710: 1.01390,
    800: 1.01383,
    1064: 1.01371,
    1064.150: 1.01371,
}

KbwC = {
    308: 1.04554,
    351: 1.04338,
    354.717: 1.04324,
    355: 1.04323,
    400: 1.04191,
    510.6: 1.04026,
    532: 1.04007,
    532.075: 1.04007,
    710: 1.03919,
    800: 1.03897,
    1064: 1.03863,
    1064.150: 1.03863,
}

molecular_depolarizationC = {
    351: 0.004158,
    354.717: 0.003959,
    355: 0.003956,
    400: 0.003825,
    510.6: 0.003673,
    532: 0.003656,
    532.075: 0.003656,
    710: 0.003575,
    800: 0.003555,
    1064: 0.003524,
    1064.150: 0.003524,
}

molecular_depolarizationT = {
    351: 0.01559,
    354.717: 0.01554,
    355: 0.01554,
    400: 0.01507,
    510.6: 0.01448,
    532: 0.01441,
    532.075: 0.01441,
    710: 0.01410,
    800: 0.01402,
    1064: 0.01390,
    1064.150: 0.01390,
}

# Create interpolation function once, to avoid re-calculation (does it matter?)
f_ext = interpolate.interp1d(list(Cs.keys()), list(Cs.values()), kind="cubic")
f_bst = interpolate.interp1d(list(BsT.keys()), list(BsT.values()), kind="cubic")
f_bsc = interpolate.interp1d(list(BsC.keys()), list(BsC.values()), kind="cubic")
f_bsc_parallel = interpolate.interp1d(
    list(BsC_parallel.keys()), list(BsC_parallel.values()), kind="cubic"
)
f_bsc_perpendicular = interpolate.interp1d(
    list(BsC_perpendicular.keys()), list(BsC_perpendicular.values()), kind="cubic"
)

# Splines introduce arifacts due to limited input resolution
f_kbwt = interpolate.interp1d(list(KbwT.keys()), list(KbwT.values()), kind="linear")
f_kbwc = interpolate.interp1d(list(KbwC.keys()), list(KbwC.values()), kind="linear")
f_molt = interpolate.interp1d(
    list(molecular_depolarizationT.keys()),
    list(molecular_depolarizationT.values()),
    kind="linear",
)
f_molc = interpolate.interp1d(
    list(molecular_depolarizationC.keys()),
    list(molecular_depolarizationC.values()),
    kind="linear",
)


def molecular_backscatter(
    wavelength: float,
    pressure: np.ndarray[Any, np.dtype[np.float64]],
    temperature: np.ndarray[Any, np.dtype[np.float64]],
    component: str = "total",
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Molecular backscatter calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    beta_molecular: float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in [
        "total",
        "cabannes",
        "cabannes_parallel",
        "cabannes_perpendicular",
    ]:
        raise ValueError(
            "Molecular backscatter available only for 'total' or 'cabannes' component."
        )

    if component == "total":
        bs_function = f_bst
    elif component == "cabannes":
        bs_function = f_bsc
    elif component == "cabannes_parallel":
        bs_function = f_bsc_parallel
    elif component == "cabannes_perpendicular":
        bs_function = f_bsc_perpendicular
    else:
        raise ValueError(f"{component} not found.")

    Bs = bs_function(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.0

    # Calculate the molecular backscatter coefficient.
    beta_molecular = Bs * pressure / temperature

    return beta_molecular


def molecular_lidar_ratio(wavelength: float, component: str = "total") -> float:
    """
    Molecular lidar ratio.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    lidar_ratio_molecular : float
       The molecular backscatter coefficient. (m^-1 * sr^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    if component not in ["total", "cabannes"]:
        raise ValueError(
            "Molecular lidar ratio available only for 'total' or 'cabannes' component."
        )

    if component == "total":
        k_function = f_kbwt
    else:
        k_function = f_kbwc

    Kbw = k_function(wavelength)

    lidar_ratio_molecular = 8 * np.pi / 3.0 * Kbw

    return lidar_ratio_molecular


def molecular_depolarization(wavelength: float, component: str = "total") -> float:
    """
    Molecular lidar ratio.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    component : str
       One of 'total' or 'cabannes'.

    Returns
    -------
    molecular volume depolarization: float

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """

    if component not in ["total", "cabannes"]:
        raise ValueError(
            "Molecular lidar ratio available only for 'total' or 'cabannes' component."
        )

    if component == "total":
        moldepo_function = f_molt
    else:
        moldepo_function = f_molc

    molecular_depolarization = moldepo_function(wavelength)

    return molecular_depolarization


def molecular_extinction(
    wavelength: float,
    pressure: np.ndarray[Any, np.dtype[np.float64]],
    temperature: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Molecular extinction calculation.

    Parameters
    ----------
    wavelength : float
       The wavelength of the radiation in air. From 308 to 1064.15
    pressure : float
       The atmospheric pressure. (Pa)
    temperature : float
       The atmospheric temperature. (K)

    Returns
    -------
    alpha_molecular: float
       The molecular extinction coefficient. (m^-1)

    References
    ----------
    Freudenthaler, V. Rayleigh scattering coefficients and linear depolarization
    ratios at several EARLINET lidar wavelengths. p.6-7 (2015)
    """
    cs = f_ext(wavelength)

    # Convert pressure to correct units for calculation. (Pa to hPa)
    pressure = pressure / 100.0

    # Calculate the molecular backscatter coefficient.
    alpha_molecular = cs * pressure / temperature

    return alpha_molecular


def molecular_properties(
    wavelength: float,
    pressure: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray,
    temperature: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray,
    heights: np.ndarray[Any, np.dtype[np.float64]] | ExtensionArray,
    times: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    component: str = "total",
) -> xr.Dataset:
    """Molecular Attenuated  Backscatter: beta_mol_att = beta_mol * Transmittance**2

    Args:
        wavelength (float): wavelength of our desired beta molecular attenuated
        pressure (np.ndarray[Any, np.dtype[np.float64]]): pressure profile
        temperature (np.ndarray[Any, np.dtype[np.float64]]): temperature profile
        heights (np.ndarray[Any, np.dtype[np.float64]]): height profile
        times (np.ndarray[Any, np.dtype[np.float64]] | None, optional): time array. Defaults to None. Note: this is not used in the calculation yet. In development.
        component (str, optional): _description_. Defaults to "total".

    Returns:
        xr.Dataset: molecular attenuated backscatter profile, molecular backscatter profile, molecular extinction profile, molecular lidar ratio, molecular depolarization ratio
    """
    if isinstance(pressure, ExtensionArray):
        pressure = pressure.to_numpy()
    if isinstance(temperature, ExtensionArray):
        temperature = temperature.to_numpy()
    if isinstance(heights, ExtensionArray):
        heights = heights.to_numpy()
    if times is not None and isinstance(times, ExtensionArray):
        times = times.to_numpy()

    # molecular backscatter and extinction #
    beta_mol = molecular_backscatter(
        wavelength, pressure, temperature, component=component
    )
    alfa_mol = molecular_extinction(wavelength, pressure, temperature)
    lr_mol = molecular_lidar_ratio(wavelength, component=component)

    depo_mol = molecular_depolarization(wavelength, component=component)

    """ transmittance """
    transmittance_array = transmittance(alfa_mol, heights)

    """ attenuated molecular backscatter """
    att_beta_mol = attenuated_backscatter(beta_mol, transmittance_array)

    if times is None:
        mol_properties = xr.Dataset(
            {
                "molecular_beta": (["range"], beta_mol),
                "molecular_alpha": (["range"], alfa_mol),
                "attenuated_molecular_beta": (["range"], att_beta_mol),
                "molecular_lidar_ratio": lr_mol,
                "molecular_depolarization": depo_mol,
            },
            coords={"range": heights},
        )
    else:
        mol_properties = xr.Dataset(
            {
                "molecular_beta": (["time", "range"], beta_mol),
                "molecular_alpha": (["time", "range"], alfa_mol),
                "attenuated_molecular_beta": (["time", "range"], att_beta_mol),
                "molecular_lidar_ratio": ([], lr_mol),
                "molecular_depolarization": depo_mol,
            },
            coords={
                "time": times,
                "range": heights,
            },  # FIXME: time is not used in the calculation yet
        )
    return mol_properties

