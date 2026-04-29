# -*- coding: utf-8 -*-
""" Retrievals of backscatter and extinction based on Raman measurements

.. warning::
   These functions have not been tested!
"""
from typing import Tuple
from loguru import logger
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from lidarpy.atmo.atmo import number_concentration_from_meteo, transmittance
from lidarpy.atmo.rayleigh import molecular_properties
from lidarpy.utils.utils import signal_to_rcs


def retrieve_extinction_deprecated(
    signal: np.ndarray,
    ranges: np.ndarray,
    wavelengths: Tuple[float, float],
    pressure: np.ndarray,
    temperature: np.ndarray,
    window_size_m: float = 100.0,
    savgol_order: int = 1,
    particle_angstrom_exponent: float = 1,
) -> np.ndarray:
    """Retrieve particle extinction from lidar signal.

    This function calculates the particle extinction from the lidar signal using
    range corrected signal (rcs) and molecular properties.

    Args:
       signal (np.ndarray): Lidar signal.
       ranges (np.ndarray): Array of range values corresponding to the signal.
       wavelengths (Tuple[float, float]): Tuple of two wavelengths (in nm).
       pressure (np.ndarray): Atmospheric pressure profile.
       temperature (np.ndarray): Atmospheric temperature profile.
       window_size_m (float, optional): Window size for derivative calculation in meters.
       savgol_order (int, optional): Order of Savitzky-Golay filter.
       particle_angstrom_exponent (float, optional): Particle ??ngstr??m exponent.

    Returns:
       Union[np.ndarray, None]: Array containing the particle extinction values.
       If the number density or signal is invalid, the corresponding element in
       the result will be set to np.nan. If the input signal and number density
       are both invalid, the function returns None.
    """

    # Retrieve range corrected signal
    rcs = signal_to_rcs(signal, ranges)

    elastic_rayleigh_beta = molecular_properties(wavelengths[0], pressure, temperature, heights=ranges)
    raman_rayleigh_beta = molecular_properties(wavelengths[1], pressure, temperature, heights=ranges)
    elastic_alpha_molecular = elastic_rayleigh_beta['molecular_alpha']
    raman_alpha_molecular = raman_rayleigh_beta['molecular_alpha']

    # Retrieve number density
    number_density = number_concentration_from_meteo(pressure, temperature, atmospheric_component='nitrogen')

    # Calculate ratio
    ratio = np.nan * np.ones(signal.shape)
    valid_idx = np.logical_and(number_density > 0, signal > 0)
    ratio[valid_idx] = np.ma.log(number_density[valid_idx] / rcs[valid_idx])

    # Apply derivative
    dz = np.median(np.diff(ranges))  # type: ignore
    window_size_bin = np.floor(window_size_m / dz).astype(int)
    
    derivative = savgol_filter(
        ratio,
        window_size_bin,
        savgol_order,
        deriv=1,
        delta=dz,
        mode="nearest",
        cval=np.nan,
    )  # Calculate 1st derivative

    # Calculate particle extinction
    cte = 1 + (wavelengths[1] / wavelengths[0]) ** (-particle_angstrom_exponent)
    particle_alpha = (
        derivative - raman_alpha_molecular - elastic_alpha_molecular
    ) / cte

    # Fill like overlap
    idx_overlap = np.ceil(window_size_bin / 2).astype(int)
    particle_alpha[:idx_overlap] = particle_alpha[idx_overlap]

    return particle_alpha

def retrieve_backscatter(
    signal_raman: np.ndarray,
    signal_emission: np.ndarray,
    extinction_profile: np.ndarray,
    range_profile: np.ndarray,
    wavelengths: Tuple[float, float],
    pressure: np.ndarray,
    temperature: np.ndarray,
    reference: float | Tuple[float, float],
    particle_angstrom_exponent: float = 1,
    beta_part_ref: float = 0,
) -> np.ndarray:
    """Calculates the aerosol backscatter coefficient.

    This function calculates the aerosol backscatter coefficient based on preprocessed
    elastic and Raman signals and the retrieved aerosol extinction coefficient.

    Parameters:
       signal_raman (np.ndarray): The range-corrected Raman signal (1D array of size M).
       signal_emission (np.ndarray): The range-corrected elastic signal (at the emission wavelength, 1D array of size M).
       extinction_profile (np.ndarray): The aerosol extinction coefficient at each point of the signal profile (1D array of size M).
       range_profile (np.ndarray): Array of range values corresponding to the signal (1D array of size M).
       wavelengths (Tuple[float, float]): Tuple of two wavelengths (in nm) for the emission and Raman signals.
       pressure (np.ndarray): Atmospheric pressure profile (1D array of size M, [Pa]).
       temperature (np.ndarray): Atmospheric temperature profile (1D array of size M, [K]).
       reference float | (Tuple[float, float]): Reference altitude range.
       particle_angstrom_exponent (float, optional): Particle ??ngstr??m exponent (default: 1).
       beta_aer_ref (float, optional): The molecular backscatter coefficient at the reference altitude (default: 0).

    Returns:
       np.ndarray: The aerosol backscatter coefficient [m^-1].

    References:
       Ansmann, A. et al. Independent measurement of extinction and backscatter profiles in cirrus clouds by using a combined Raman elastic-backscatter lidar.
       Applied Optics Vol. 31, Issue 33, pp. 7113-7131 (1992)
    """
    # --- Convertir reference float a tupla ---
    if isinstance(reference, float):
        reference = (reference, reference)
    ymin, ymax = reference

    # --- ??ndices de referencia ---
    if ymin == ymax:        
        idx_ref = np.zeros(len(range_profile), dtype=bool)
        idx_ref[np.abs(range_profile - ymin).argmin()] = True
    else:
        idx_ref = np.logical_and(range_profile >= ymin, range_profile <= ymax)
        if not idx_ref.any():
            raise ValueError("Range `reference` out of rcs size.")

    # Calculate profiles of molecular extinction
    atmo_data = molecular_properties(
        wavelengths[0], pressure, temperature, heights=range_profile
    )
    beta_mol = atmo_data["molecular_beta"].values
    alpha_mol_emmision = atmo_data["molecular_alpha"].values
    atmo_data_raman = molecular_properties(
        wavelengths[1], pressure, temperature, heights=range_profile
    )
    beta_mol_raman = atmo_data_raman["molecular_beta"].values
    alpha_mol_raman = atmo_data_raman["molecular_alpha"].values
    
    #Remove nan and negative values from extiction
    extinction_profile = np.nan_to_num(extinction_profile)
    #extinction_profile[extinction_profile < 0] = 0 ##TODO: It led to negative values in the backscatter profile 
    logger.info('Negative values in extinction profile were removed.')
    
    alpha_part_raman = extinction_profile * (wavelengths[1] / wavelengths[0]) ** (
        -particle_angstrom_exponent
    )

    T_raman = np.exp(-cumulative_trapezoid(alpha_mol_raman + alpha_part_raman, range_profile, initial=0))  # type: ignore
    T_elastic = np.exp(-cumulative_trapezoid(alpha_mol_emmision + extinction_profile, range_profile, initial=0))  # type: ignore

    # Calibration at reference altitude
    beta_at_reference = beta_part_ref + np.nanmean(beta_mol[idx_ref])
    gain_ratio = ( (signal_emission / signal_raman) * (T_raman / T_elastic) * (beta_mol_raman / beta_at_reference) )
    gain_ratio_at_reference = np.nanmean(gain_ratio[idx_ref])
    beta_part = (
        beta_mol_raman
        * (1 / gain_ratio_at_reference)
        * ((signal_emission * T_raman) / (signal_raman * T_elastic))
        - beta_mol
    )
    return beta_part

def retrieve_extinction(
    signal: np.ndarray,
    ranges: np.ndarray,
    wavelengths: Tuple[float, float],
    pressure: np.ndarray,
    temperature: np.ndarray,
    reference: float | Tuple[float, float],
    particle_angstrom_exponent: float = 1,
    **kwargs: dict,
) -> np.ndarray:
    """Retrieve particle extinction from lidar signal.

    Args:
        signal (np.ndarray): Raman lidar signal.
        ranges (np.ndarray): Array of range values corresponding to the signal.
        wavelengths (Tuple[float, float]): Tuple of elastic and Raman wavelengths (in nm).
        pressure (np.ndarray): Atmospheric pressure profile.
        temperature (np.ndarray): Atmospheric temperature profile.
        particle_angstrom_exponent (float, optional): Particle extinction-related ??ngstr??m exponent. Defaults to 1.
        reference (float | Tuple[float, float]): Aerosol-partile free reference region.

    Raises:
        ValueError: If `reference` is out of range. 

    Returns:
        np.ndarray: Particle extinction profile.
    """    
    # --- Convertir reference float a tupla ---
    # Convertir a tupla si es float o escalar
    # --- Convertir reference float a tupla ---
    if isinstance(reference, float):
        reference = (reference, reference)
        
    
    ymin, ymax = reference

    # --- Retrieve range corrected signal ---
    rcs = signal_to_rcs(signal, ranges)


    elastic_rayleigh_beta = molecular_properties(wavelengths[0], pressure, temperature, heights=ranges)
    raman_rayleigh_beta = molecular_properties(wavelengths[1], pressure, temperature, heights=ranges)
    elastic_alpha_molecular = elastic_rayleigh_beta['molecular_alpha']
    raman_alpha_molecular = raman_rayleigh_beta['molecular_alpha']
    transmittance_twice = transmittance(elastic_alpha_molecular + raman_alpha_molecular, ranges)
    
    # Normalize rcs to attenuated molecular backscatter (elastic and raman)
    attenuated_raman_molecular_backscatter = raman_rayleigh_beta['molecular_beta'] * transmittance_twice

    # --- Selecci??n de ??ndices de referencia ---
    if ymin == ymax:
        # Caso de un solo punto: encontrar el ??ndice m??s cercano
        idx_ref = np.abs(ranges - ymin).argmin()
        reference_value = float(rcs[idx_ref])
        reference_att_mol_beta = float(attenuated_raman_molecular_backscatter[idx_ref])
    else:
        idx_ref = np.logical_and(ranges >= ymin, ranges <= ymax)
        if not idx_ref.any():
            raise ValueError("Range `reference` out of rcs size.")
        reference_value = np.nanmean(rcs[idx_ref])
        reference_att_mol_beta = np.nanmean(attenuated_raman_molecular_backscatter[idx_ref])

    # --- Calcular relaci??n atenuada ---
    rcs2attbeta = reference_att_mol_beta * (np.array(rcs) / reference_value)
    att_beta_ratio = rcs2attbeta / np.array(attenuated_raman_molecular_backscatter)
    spectral_dependence = 1 + (wavelengths[1] / wavelengths[0]) ** (-particle_angstrom_exponent)
    aod = -np.log(att_beta_ratio) / spectral_dependence

    # Extinction como gradiente de AOD
    extinction = np.gradient(aod, ranges)

    # --- Aplicar full_overlap_height si se proporciona ---
    if 'full_overlap_height' in kwargs:
        full_overlap_height = kwargs['full_overlap_height']
        idx_full = np.argmin(np.abs(ranges - full_overlap_height))
        extinction[:idx_full] = extinction[idx_full]

    return extinction
