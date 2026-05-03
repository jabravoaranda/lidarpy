from typing import Tuple
from pathlib import Path
from lidarpy.atmo.atmo import attenuated_backscatter
from lidarpy.general_utils.fitting import linear_fit, moving_linear_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import trapezoid

from lidarpy.atmo.rayleigh import molecular_properties
from lidarpy.utils.file_manager import channel2info
from lidarpy.utils.utils import signal_to_rcs

def _attenuated_backscatter(
    rcs: xr.DataArray,
    attenuated_molecular_backscatter: xr.DataArray,
    reference_range: float | Tuple[float, float],
) -> xr.DataArray:
    """Calculate the attenuated backscatter."""
    
    # Caso: referencia puntual
    if np.isscalar(reference_range):  # más robusto que isinstance(reference_range, float)
        ref_mol = attenuated_molecular_backscatter.sel(
            range=reference_range, method="nearest"
        )
        ref_rcs = rcs.sel(range=reference_range, method="nearest")
        attenuated_backscatter = ref_mol * (rcs / ref_rcs)

    # Caso: referencia en rango (ventana)
    elif isinstance(reference_range, tuple) and len(reference_range) == 2:
        ref_mol = attenuated_molecular_backscatter.sel(
            range=slice(*reference_range)
        ).mean("range")
        ref_rcs = rcs.sel(range=slice(*reference_range)).mean("range")
        attenuated_backscatter = ref_mol * (rcs / ref_rcs)

    # Fallback de seguridad
    else:
        raise ValueError(
            f"reference_range debe ser float o tuple, pero recibí {type(reference_range)}"
        )

    return attenuated_backscatter


def _get_mask_residual(
    attenuated_backscatter: xr.DataArray,
    attenuated_molecular_backscatter: xr.DataArray,
    reference_ranges: tuple[float, float],
) -> xr.DataArray:
    """Get mask for residual within the standard deviation.

    Args:
        attenuated_backscatter (xr.DataArray): Attenuated backscatter.
        attenuated_molecular_backscatter (xr.DataArray): Attenuated molecular backscatter.
        reference_ranges (tuple[float, float]): Aerosol-free reference range.

    Returns:
        xr.DataArray: Mask for residual.
    """

    def _get_window_size(
        reference_ranges: tuple[float, float], ranges: np.ndarray[float]
    ) -> int:
        """It provides the window size of a given reference range (e.g., 6000 - 7000 meters).

        Args:
            reference_ranges (tuple[float, float]): Reference range in meters.
            ranges (np.dnarray[float]): Range array in meters.

        Returns:
            int: Window size.
        """
        dz = np.median(np.diff(ranges))  # type: ignore
        reference_idxs = (
            np.floor(reference_ranges[0] / dz).astype(int),
            np.floor(reference_ranges[1] / dz).astype(int),
        )
        return reference_idxs[1] - reference_idxs[0]


    window_size_bin = _get_window_size(
        reference_ranges, attenuated_backscatter.range.values
    )

    # Get boolean variable where the residual is within the standard deviation
    att_beta_mean = attenuated_backscatter.rolling(
        range=window_size_bin, center=True
    ).mean("range")
    att_beta_mean = att_beta_mean.where(~np.isnan(att_beta_mean), drop=True)
    att_beta_std = attenuated_backscatter.rolling(
        range=window_size_bin, center=True
    ).std("range")
    att_beta_std = att_beta_std.where(~np.isnan(att_beta_std), drop=True)
    att_mol_beta_mean = attenuated_molecular_backscatter.rolling(
        range=window_size_bin, center=True
    ).mean("range")
    att_mol_beta_mean = att_mol_beta_mean.where(~np.isnan(att_mol_beta_mean), drop=True)
    att_mol_beta_std = attenuated_molecular_backscatter.rolling(
        range=window_size_bin, center=True
    ).std("range")    
    att_mol_beta_std = att_mol_beta_std.where(~np.isnan(att_mol_beta_std), drop=True)

    mask = (att_beta_mean + att_beta_std) > (att_mol_beta_mean - att_mol_beta_std)
    mask.attrs = {
        "long_name": "mask for residual within standard deviation",
        "units": "#",
    }
    return mask



def get_reference_window(
    channel: str,
    signal: xr.DataArray,
    meteo_profiles: pd.DataFrame,
    reference_candidate_limits: Tuple[float, float],
    reference_half_window: int = 500,
    debugging: bool = False,
    tolerance: float = 0.01,
    tolerance_mask: float = 0.8,
    window_size: int = 250,
) -> tuple[dict, tuple[float, float]]:
    """It provides the optimal reference range for the signal, using residual and extinction filters.

    Args:
        channel (str): Channel of the signal.
        signal (xr.DataArray): Signal to be analyzed.
        meteo_profiles (pd.DataFrame): Meteo profiles from lidarpy.atmo module. 
        reference_candidate_limits (Tuple[float, float]): Reference candidate limits.
        reference_half_window (int, optional): Half Window to split the reference_candidate_limits. Defaults to 500
        debugging (bool, optional): If True, it will plot the results. Defaults to False.
        tolerance (float, optional): Relative tolerance for the derivate filter. Defaults to 0.01.
        tolerance_mask (float, optional): Relative tolerance for the mask. Defaults to 0.8.
        window_size (int, optional): Window size for the moving linear fit. Defaults to 250.

    Returns:
        tuple[dict, tuple[float, float]]: Dictionary of candidates and the final reference range.
    """    

    wavelength, _, _, _ = channel2info(channel)

    full_ranges = signal.range.values

    # Molecular properties from meteo profiles
    mol_properties = molecular_properties(
        wavelength, meteo_profiles["pressure"], meteo_profiles["temperature"], full_ranges
    )

    # Recorte de señal y propiedades
    mol_properties = mol_properties.sel(range=slice(reference_candidate_limits[0] - 2*reference_half_window,
                                                    reference_candidate_limits[1] + 2*reference_half_window))
    signal = signal.sel(range=slice(reference_candidate_limits[0] - 2*reference_half_window,
                                    reference_candidate_limits[1] + 2*reference_half_window))
    
    # RCS
    rcs = signal_to_rcs(signal, signal.range)

    # Attenuated Molecular Backscatter
    attenuated_molecular_backscatter = mol_properties["atten_molecular_beta"]
    
    candidates = {}

    #Step 0: Calculate the first derivate of attenuated backscatter and attenuated molecular backscatter
    nrcs = ( rcs.copy()/rcs.mean('range') ) * attenuated_molecular_backscatter.mean('range') 
    derivate_nrcs = moving_linear_fit(nrcs.range.values, nrcs.values, window_size= window_size, get_intercept=True)
    derivate_attenuated_molecular_backscatter = moving_linear_fit(attenuated_molecular_backscatter.range.values, attenuated_molecular_backscatter.values, window_size= window_size, get_intercept=True)

    #filter regions with relative tolerance between derivate_nrcs['slope'] and derivate_attenuated_molecular_backscatter['slope'] less than 0.1
    mask = np.abs((derivate_nrcs['slope'] - derivate_attenuated_molecular_backscatter['slope']) / derivate_attenuated_molecular_backscatter['slope']) < tolerance
    masked_slope = derivate_nrcs['slope'].copy()
    masked_slope[~mask] = np.nan
    #get range from masked_slope[~mask]
    masked_ranges = nrcs.range.values[mask]
    
    print("Masked ranges:", masked_ranges)

    #create candidates['derivate'] with the ranges with mask true
    candidates['derivate'] = {}
    for range_ in masked_ranges:
        candidate_ = (range_ - reference_half_window, range_ + reference_half_window)

        # Attenuated Backscatter
        attenuated_backscatter = _attenuated_backscatter(rcs, attenuated_molecular_backscatter, candidate_)

        candidates['derivate'][range_] = {
            'candidate': candidate_,          
            'attenuated_backscatter': attenuated_backscatter  
        }

    # Step 1: residual filter
    candidates['residual'] = {}
    for range_ in candidates['derivate'].keys(): #np.arange(reference_candidate_limits[0], reference_candidate_limits[1], 2 * reference_half_window):        
        candidate_ = (range_ - reference_half_window, range_ + reference_half_window)

        # Attenuated Backscatter
        attenuated_backscatter = _attenuated_backscatter(rcs, attenuated_molecular_backscatter, candidate_)

        # Residual mask
        mask_res = _get_mask_residual(attenuated_backscatter, attenuated_molecular_backscatter, candidate_)
        if mask_res.sum().item() / mask_res.size > tolerance_mask:            
            candidates['residual'][range_] = {
                'candidate': candidate_,
                'attenuated_backscatter': attenuated_backscatter
            }                

    # print("Candidates after residual filter:", candidates['residual'].keys())
    
    # Step 2: Prepare final candidates
    if len(candidates['residual']) > 0:
        candidates_ = candidates['residual'].copy()
    else:
        print("No candidates passed residual filter, using derivative candidates instead.")
        candidates_ = candidates['derivate'].copy()
    
    # Step 3: extinction filter
    candidates['extinction'] = {}   
    for range_ in candidates_.keys():
        candidate_ = candidates_[range_]['candidate']
        attenuated_backscatter_ = candidates_[range_]['attenuated_backscatter'].sel(range=slice(*candidate_))
        attenuated_molecular_backscatter_ = attenuated_molecular_backscatter.sel(range=slice(*candidate_))

        stats_ = linear_fit(
            attenuated_backscatter_.range.values,
            (attenuated_backscatter_ / attenuated_molecular_backscatter_).values
        )

        extinction = -0.5 * stats_["parameters"][1]
        std_extinction = 0.5 * stats_["standard_deviation_parameters"][1]

        if std_extinction <= extinction:
            candidates['extinction'][range_] = {
                'candidate': candidate_,
                'attenuated_backscatter': candidates_[range_]['attenuated_backscatter'],
                'stats': stats_
            }

    # print("Candidates after extinction filter:", candidates['extinction'].keys())

    # if len(candidates['extinction']) == 0:
    #     return None

    # Step 3: final selection with weighting
    candidates['final'] = candidates['extinction'].copy()
    weighting_function = np.nan * np.ones(len(candidates['final']))
    for idx, range_ in enumerate(candidates['final']):
        candidate_ = candidates['final'][range_]['candidate']
        attenuated_backscatter = candidates['final'][range_]['attenuated_backscatter']
        stats = candidates['final'][range_]['stats']

        mean_ = np.mean(attenuated_backscatter.sel(range=slice(*candidate_)).values)
        std_ = np.std(attenuated_backscatter.sel(range=slice(*candidate_)).values)
        slope_ = stats["parameters"][1]
        weighting_function[idx] = np.abs(mean_) * std_ * np.abs(slope_)

    final_ranges = np.array(list(candidates['final'].keys()))
    
    final_reference_range = final_ranges[np.argmin(weighting_function)]
    final_reference_slice = candidates['final'][final_reference_range]['candidate']
    


    # print("Final reference range:", final_reference_slice)
    if debugging:
        print("Candidates after derivative filter:", candidates['derivate'].keys())
        print("Candidates after residual filter:", candidates['residual'].keys())
        
        final_keys = list(candidates['final'].keys())
        mask_final = np.isin(attenuated_backscatter.range.values, final_keys)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # --- (1) Attenuated Backscatter vs Molecular ---
        ax = axes[0]
        ax.set_title('Attenuated Backscatter vs Attenuated Molecular Backscatter')
        # ax.plot(nrcs.range, nrcs.values, label='Attenuated Backscatter')
        ax.plot(attenuated_molecular_backscatter.range, attenuated_molecular_backscatter,
                label=r'$\beta_{att}^{mol}$', color='orange', linewidth=2, zorder=1)
        # Pinta solo el candidato final que hemos seleccionado
        ax.plot(attenuated_backscatter.range, candidates['final'][final_reference_range]['attenuated_backscatter'].values,
                lw=2,  label=r'Final $\beta_{att}$', color='C0', zorder=2)
        ax.axvline(final_reference_range, color='purple', linestyle='--',
                   label='Reference range', linewidth=1, zorder=3)
        ax.scatter(attenuated_backscatter.range[mask], candidates['final'][final_reference_range]['attenuated_backscatter'].values[mask],
                   label='Derivate Candidates', color='red', s=30, zorder=4)
        ax.scatter(attenuated_backscatter.range[mask_final], candidates['final'][final_reference_range]['attenuated_backscatter'].values[mask_final],
                   label='Final Candidates', color='green', s=50, marker='x', zorder=5)
        ax.set_ylabel(r'$\beta_{att}$ (m$^{-1}$ sr$^{-1}$)')
        ax.set_yscale('log')
        ax.set_xlim(reference_candidate_limits[0] - 2 * reference_half_window, reference_candidate_limits[1] + 2 * reference_half_window)
        ax.grid()
        ax.legend(fontsize=10, loc='upper center')
        ax.set_facecolor("white")

        # --- (2) Derivative Comparison ---
        ax = axes[1]
        ax.set_title('Derivatives')
        ax.plot(attenuated_molecular_backscatter.range, derivate_attenuated_molecular_backscatter['slope'],
                label=r'$\beta_{att}^{mol}$', color='orange', linewidth=2, zorder=1)
        ax.plot(nrcs.range.values, derivate_nrcs['slope'], label=r'$\beta_{att}$', linewidth=2, color='C0', zorder=2)
        ax.axvline(final_reference_range, color='purple', linestyle='--',
                   label='Reference height', linewidth=1, zorder=3)
        ax.scatter(nrcs.range.values[mask], derivate_nrcs['slope'][mask],
                   label='Derivate Candidates', color='red', s=30, zorder=4)
        ax.scatter(nrcs.range.values[mask_final], derivate_nrcs['slope'][mask_final],
                   label='Final Candidates', color='green', s=50, marker='x', zorder=5)
        ax.set_ylabel(r'$\frac{d\beta_{att}}{dz}$')
        ax.set_xlim(reference_candidate_limits[0] - 2 * reference_half_window, reference_candidate_limits[1] + 2 * reference_half_window)
        # ax.set_xlim(0, 10000)
        ax.grid()
        ax.legend(fontsize=10, loc='upper center')
        ax.set_facecolor("white")
        
        fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'reference_range.png')

        print("Final reference range:", final_reference_slice)
    return candidates, final_reference_slice


def get_reference_range(
    channel: str,
    signal: xr.DataArray,
    meteo_profiles: pd.DataFrame,
    reference_candidate_limits: Tuple[float, float],
    reference_half_window: int = 500,
    debugging: bool = False,
    tolerance: float = 0.01,
    tolerance_mask: float = 0.8,
) -> tuple[dict, float]:
    """It provides the optimal reference range for the signal,
    using derivate + residual filters, and final selection weighting
    (without extinction filter).

    Args:
        channel (str): Channel of the signal.
        signal (xr.DataArray): Signal to be analyzed (1D in range).
        meteo_profiles (pd.DataFrame): Meteo profiles from lidarpy.atmo module. 
        reference_candidate_limits (Tuple[float, float]): Reference candidate limits.
        reference_half_window (int, optional): Half Window. Defaults to 500
        debugging (bool, optional): If True, it will plot. Defaults to False.
        tolerance (float, optional): Relative tolerance for derivate filter. Defaults to 0.01.
        tolerance_mask (float, optional): Relative tolerance for residual mask. Defaults to 0.8.
    Returns:
        tuple[dict, tuple[float, float]]: Dictionary of candidates and final reference range.
    """    

    wavelength, _, _, _ = channel2info(channel)

    full_ranges = signal.range.values

    # Molecular properties
    mol_properties_full = molecular_properties(
        wavelength, meteo_profiles["pressure"], meteo_profiles["temperature"], full_ranges
    )

    # Recorte
    mol_properties = mol_properties_full.sel(
        range=slice(reference_candidate_limits[0] - 2*reference_half_window,
                    reference_candidate_limits[1] + 2*reference_half_window)
    )
    signal_full = signal.copy(  )
    signal = signal_full.sel(
        range=slice(reference_candidate_limits[0] - 2*reference_half_window,
                    reference_candidate_limits[1] + 2*reference_half_window)
    )
    
    # RCS
    rcs = signal_to_rcs(signal, signal.range)

    # Attenuated Molecular Backscatter    
    attenuated_molecular_backscatter = mol_properties["atten_molecular_beta"]
    
    candidates = {}

    # Step 0: Derivative filter
    nrcs = (rcs.copy()/rcs.mean('range')) * attenuated_molecular_backscatter.mean('range')
    derivate_nrcs = moving_linear_fit(nrcs.range.values, nrcs.values,
                                      window_size= 2*reference_half_window, get_intercept=True)
    
    derivate_attenuated_molecular_backscatter = moving_linear_fit(
        attenuated_molecular_backscatter.range.values,
        attenuated_molecular_backscatter.values,
        window_size= 2*reference_half_window, get_intercept=True
    )

    mask = np.abs((derivate_nrcs['slope'] - derivate_attenuated_molecular_backscatter['slope']) / derivate_attenuated_molecular_backscatter['slope']) < tolerance
    masked_slope = derivate_nrcs['slope'].copy()
    masked_slope[~mask] = np.nan
    #get range from masked_slope[~mask]
    masked_ranges = nrcs.range.values[mask]
    # Create candidates from masked ranges
    candidates['derivate'] = {}
    for range_ in masked_ranges:
        attenuated_backscatter = _attenuated_backscatter(rcs, attenuated_molecular_backscatter, range_)
        candidates['derivate'][range_] = {
                'attenuated_backscatter': attenuated_backscatter,
            }
    # print("Candidates after derivative filter:", candidates['derivate'].keys())

    # Step 1: Residual filter
    candidates['residual'] = {}
    for range_ in candidates['derivate'].keys():
        candidate_ = (range_ - reference_half_window, range_ + reference_half_window)
        attenuated_backscatter = _attenuated_backscatter(rcs, attenuated_molecular_backscatter, range_)

        mask_res = _get_mask_residual(attenuated_backscatter, attenuated_molecular_backscatter, candidate_)
        if mask_res.sum().item() / mask_res.size > tolerance_mask:
            
            candidates['residual'][range_] = {
                'candidate': candidate_,
                'attenuated_backscatter': attenuated_backscatter,
            }
    # print("Candidates after residual filter:", candidates['residual'].keys())
    
    # Step 2: Prepare final candidates
    if len(candidates['residual']) > 0:
        for range_ in candidates['residual']:
            candidates['final'] = candidates['residual'].copy()
    else:
        print("No candidates passed residual filter, using derivative candidates instead.")
        candidates['final'] = candidates['derivate'].copy()


    # Step 3: Final selection (Minimum area between signals)
    areas = []

    for range_, cand in candidates['final'].items(): #range_ is the key, cand is the value

        # Recortamos ambas señales en el rango candidato (key-window, key+window)
        att_backscatter = cand['attenuated_backscatter']
        att_mol_backscatter = attenuated_molecular_backscatter

        # Calcular diferencia absoluta
        diff = att_backscatter.values - att_mol_backscatter.values
        diff[diff >0] = 0  # Solo consideramos las diferencias negativas

        # Calcular área (trapezoidal para considerar bien dz)
        area = trapezoid(np.abs(diff), att_backscatter.range.values)

        areas.append(area)
        # print(f"Candidate range: {range_}, Area: {area}")

    # Escoger el candidato con el área mínima
    final_ranges = np.array(list(candidates['final'].keys()))
    best_idx = np.argmin(areas)
    final_reference_height = final_ranges[best_idx]
    
        # --- DEBUGGING PLOTS UNIFICADOS ---
    if debugging:
        print("Candidates after derivative filter:", candidates['derivate'].keys())
        print("Candidates after residual filter:", candidates['residual'].keys())
        
        final_keys = list(candidates['final'].keys())
        mask_final = np.isin(attenuated_backscatter.range.values, final_keys)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # --- (1) Attenuated Backscatter vs Molecular ---
        ax = axes[0]
        ax.set_title('Attenuated Backscatter vs Attenuated Molecular Backscatter')
        # ax.plot(nrcs.range, nrcs.values, label='Attenuated Backscatter')
        ax.plot(attenuated_molecular_backscatter.range, attenuated_molecular_backscatter,
                label=r'$\beta_{att}^{mol}$', color='orange', linewidth=2, zorder=1)
        # Pinta solo el candidato final que hemos seleccionado
        ax.plot(attenuated_backscatter.range, candidates['final'][final_reference_height]['attenuated_backscatter'].values,
                lw=2,  label=r'Final $\beta_{att}$', color='C0', zorder=2)
        ax.axvline(final_reference_height, color='purple', linestyle='--',
                   label='Reference height', linewidth=1, zorder=3)
        ax.scatter(attenuated_backscatter.range[mask], candidates['final'][final_reference_height]['attenuated_backscatter'].values[mask],
                   label='Derivate Candidates', color='red', s=30, zorder=4)
        ax.scatter(attenuated_backscatter.range[mask_final], candidates['final'][final_reference_height]['attenuated_backscatter'].values[mask_final],
                   label='Final Candidates', color='green', s=50, marker='x', zorder=5)
        ax.set_ylabel(r'$\beta_{att}$ (m$^{-1}$ sr$^{-1}$)')
        ax.set_yscale('log')
        ax.set_xlim(reference_candidate_limits[0] - 2 * reference_half_window, reference_candidate_limits[1] + 2 * reference_half_window)
        ax.grid()
        ax.legend(fontsize=10, loc='upper center')
        ax.set_facecolor("white")

        # --- (2) Derivative Comparison ---
        ax = axes[1]
        ax.set_title('Derivatives')
        ax.plot(attenuated_molecular_backscatter.range, derivate_attenuated_molecular_backscatter['slope'],
                label=r'$\beta_{att}^{mol}$', color='orange', linewidth=2, zorder=1)
        ax.plot(nrcs.range.values, derivate_nrcs['slope'], label=r'$\beta_{att}$', linewidth=2, color='C0', zorder=2)
        ax.axvline(final_reference_height, color='purple', linestyle='--',
                   label='Reference height', linewidth=1, zorder=3)
        ax.scatter(nrcs.range.values[mask], derivate_nrcs['slope'][mask],
                   label='Derivate Candidates', color='red', s=30, zorder=4)
        ax.scatter(nrcs.range.values[mask_final], derivate_nrcs['slope'][mask_final],
                   label='Final Candidates', color='green', s=50, marker='x', zorder=5)
        ax.set_ylabel(r'$\frac{d\beta_{att}}{dz}$')
        ax.set_xlim(reference_candidate_limits[0] - 2 * reference_half_window, reference_candidate_limits[1] + 2 * reference_half_window)
        # ax.set_xlim(0, 10000)
        ax.grid()
        ax.legend(fontsize=10, loc='upper center')
        ax.set_facecolor("white")
        fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'reference_height.png')  


    print("Final reference height:", final_reference_height)
    return candidates, final_reference_height
