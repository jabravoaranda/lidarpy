from pathlib import Path
from typing import Any, Tuple
from matplotlib import pyplot as plt
import numpy as np
from loguru import logger
from scipy import integrate
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapezoid as trapz

from lidarpy.atmo.atmo import transmittance
from lidarpy.utils.types import ParamsDict
from lidarpy.utils.utils import refill_overlap, signal_to_rcs
from lidarpy.general_utils.plot import color_list

def klett_rcs(
    rcs_profile: np.ndarray[Any, np.dtype[np.float64]],
    range_profile: np.ndarray[Any, np.dtype[np.float64]],
    beta_mol_profile: np.ndarray[Any, np.dtype[np.float64]],
    reference: float | Tuple[float, float],
    lr_part: float | np.ndarray[Any, np.dtype[np.float64]] = 45.,
    lr_mol: float = 8 * np.pi / 3,
    beta_aer_ref: float = 0,
) -> np.ndarray:
    """Calculate aerosol backscattering using Classical Klett algorithm verified with Fernald, F. G.: Appl. Opt., 23, 652-653, 1984.

    Args:
        rcs_profile (np.ndarray): 1D signal profile.
        range_profile (np.ndarray): 1D range profile with the same shape as rcs_profile.
        beta_mol_profile (np.ndarray): 1D array containing molecular backscatter values.
        lr_mol (float): Molecular lidar ratio (default value based on Rayleigh scattering).
        lr_part (float, optional): Aerosol lidar ratio (default is 45 sr).
        reference (Tuple[float, float]): Range interval (ymin and ymax) for reference calculation. 
        beta_aer_ref (float, optional): Aerosol backscatter at reference range (ymin and ymax). Defaults to 0.

    Returns:
        np.ndarray: Aerosol-particle backscattering profile.
    """

    if isinstance(lr_part, float):
        lr_part = np.full(len(range_profile), lr_part)
    if isinstance(lr_mol, float):
        lr_mol = np.full(len(range_profile), lr_mol) # type: ignore
        
    # --- Convertir reference float a tupla ---
    if isinstance(reference, float):
        reference = (reference, reference)
        
    ymin, ymax = reference
    ymid = (ymin + ymax) / 2

    particle_beta = np.zeros(len(range_profile))

    ymiddle = np.abs(range_profile - ymid).argmin()

    range_resolution = np.median(np.diff(range_profile)).astype(float)

    idx_ref = np.logical_and(range_profile >= ymin, range_profile <= ymax)
    
    if ymin == ymax:
       #Get index of range_profile equal or neareast to ymin. Necesito un array de todo false menos un sitio donde es true y me debo quedar con ese
       idx_ref = np.abs(range_profile - ymin).argmin()
    else:
        idx_ref = np.logical_and(range_profile >= ymin, range_profile <= ymax)

    if not idx_ref.any():
        raise ValueError("Range [ymin, ymax] out of rcs size.")

    calib = np.nanmean(
        rcs_profile[idx_ref] / (beta_mol_profile[idx_ref] + beta_aer_ref)
    )

    # from Correct(ed) Klett–Fernald algorithm for elastic aerosol backscatter retrievals: a sensitivity analysis
    # Johannes Speidel* AND Hannes Vogelmann
    # https://doi.org/10.1364/AO.465944
    # Eq. 10
    # Reminder: BR = lr_mol, BP = lr_part

    integral_in_Y = np.flip(cumtrapz( np.flip((lr_mol[:ymiddle] - lr_part[:ymiddle]) * beta_mol_profile[:ymiddle]), dx=range_resolution, initial=0) ) # type: ignore
    exp_Y = np.exp( -2 * integral_in_Y)

    integral_in_particle_beta = np.flip(cumtrapz(np.flip(lr_part[:ymiddle] * rcs_profile[:ymiddle] * exp_Y), dx=range_resolution, initial=0)) # type: ignore

    total_beta = (rcs_profile[:ymiddle] * exp_Y) / (calib + 2 * integral_in_particle_beta)

    particle_beta[:ymiddle] = total_beta - beta_mol_profile[:ymiddle]

    return particle_beta


def quasi_beta(
    rcs_profile: np.ndarray[Any, np.dtype[np.float64]],
    calibration_factor: float,
    range_profile: np.ndarray[Any, np.dtype[np.float64]],
    params: dict | ParamsDict,
    lr_part: float | np.ndarray = 45.0,    
    full_overlap_height: float = 1000.0,    
    debug: bool = False,
) -> np.ndarray:
    """Calculate aerosol backscattering aproximation using algorithm verified with Baars, H., et al (2017). Atmospheric Measurement Techniques, 10(9), 3175-3201.

    Args:
        rcs_profile (np.ndarray): 1D signal profile.
        calibration_factor (float): Lidar calibration factor to convert RCS to attenuated backscatter.
        range_profile (np.ndarray): 1D range profile with the same shape as rcs_profile.
        params (dict | ParamsDict): Dictionary containing molecular backscatter and extinction profiles.
        lr_part (float, optional): Aerosol lidar ratio (default is 45 sr).
        full_overlap_height (float, optional): Height above which the overlap is considered full. Defaults to 1000 m.
    Returns:
        np.ndarray: Aerosol-particle backscattering profile.
    """

    #calculate beta attenuated 
    att_beta = rcs_profile / calibration_factor 
    
    star_beta = att_beta / transmittance(params["molecular_alpha"], range_profile)**2 - params["molecular_beta"]

    star_alpha = star_beta * lr_part

    #refill overlap
    star_alpha = refill_overlap(star_alpha, range_profile, full_overlap_height) # type: ignore

    # T2 = np.ones(len(range_profile))
    # for i in range(1, len(range_profile)):
    #     T2[i] = T2[i-1] * np.exp(-2 * (params["molecular_alpha"][i-1] + star_alpha[i-1]) * (range_profile[i] - range_profile[i-1]))

    # quasi_beta1 = att_beta / T2  - params["molecular_beta"]
    quasi_beta = att_beta / transmittance(params["molecular_alpha"] + star_alpha, range_profile)**2 - params["molecular_beta"]

    quasi_alpha = quasi_beta * lr_part    

    if debug:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharey=True)
        #Axis 1: plot RCS
        ax[0].plot(att_beta, range_profile, lw=2, label="attenuated beta")
        ax[0].set_xscale("log")
        ax[0].set_ylim(0, 9000)
        ax[0].set_xlabel("attenuated beta, [a.u.]")
        ax[0].legend()    
        #Axis 2: plot calibration
        ax[1].plot(1e6*quasi_beta, range_profile, lw=2, label="quasi beta")
        ax[1].plot(1e4*quasi_alpha, range_profile, lw=2, ls = '--', label="quasi alpha")
        ax[1].plot(1e4*star_alpha, range_profile, lw=2, label="star alpha")
        ax[1].set_xscale("linear")
        ax[1].set_ylim(0, 9000)
        ax[1].set_xlabel("quasi beta, [a.u.]")
        ax[1].legend()
        fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'quasi_beta.png')  

    return quasi_beta

def iterative_beta(
    rcs_profile: np.ndarray,
    calibration_factor: float,
    range_profile: np.ndarray,
    params: dict | ParamsDict,
    lr_part: float | np.ndarray = 45.0,    
    full_overlap_height: float = 1000.0,
    free_troposphere_height: float = 5000.0,    
    iterations: int = 20,
    tolerance: float = 0.01,
    debug: bool = False
) -> np.ndarray:
    """Calculate aerosol backscattering using iterative algorithm verified with Di Girolamo, P. et al.(1999). Applied optics, 38(21), 4585-4595.

    Args:
        rcs_profile (np.ndarray): 1D signal profile.
        calibration_factor (float): Lidar calibration factor to convert RCS to attenuated backscatter.
        range_profile (np.ndarray): 1D range profile with the same shape as rcs_profile.
        params (dict | ParamsDict): Dictionary containing molecular backscatter and extinction profiles.
        lr_part (float, optional): Aerosol lidar ratio (default is 45 sr).
        full_overlap_height (float, optional): Height above which the overlap is considered full. Defaults to 1000 m.
        free_troposphere_height (float, optional): Height below which the aerosol backscatter is integrated for convergence check. Defaults to 5000 m.
        You can use the reference height as free_troposphere_height.
        iterations (int, optional): Maximum number of iterations. Defaults to 10.
        tolerance (float, optional): Relative difference tolerance for convergence. Defaults to 0.01.
    Returns:
        np.ndarray: Aerosol-particle backscattering profile.
    """
    alpha_part_previous = np.zeros(len(range_profile))
    beta_part = np.zeros((iterations, len(range_profile)))
    backscattering_ratio = np.zeros((iterations, len(range_profile)))
    relative_diff_beta_previous = 1.0
    beta_part_previous = params["molecular_beta"]
    colors = color_list(iterations)
    resolution = np.median(np.diff(range_profile)).astype(float)
    
    # alpha_part_previous = np.zeros(len(range_profile))
    # beta_part = np.zeros((iterations, len(range_profile)))
    # backscattering_ratio = np.zeros((iterations, len(range_profile)))
    # relative_diff_beta_previous = 1.0
    # att_beta = rcs_profile / calibration_factor 
    # beta_part_previous = att_beta / transmittance(params["molecular_alpha"], range_profile)**2 - params["molecular_beta"]
    # colors = color_list(iterations)
    # resolution = np.median(np.diff(range_profile)).astype(float)

    for idx in range(iterations):
        #Molecular atmosphere
        signal_mol = calibration_factor * params["molecular_beta"] * transmittance(params["molecular_alpha"] + alpha_part_previous, range_profile)**2 / range_profile**2
        rcs_mol = signal_to_rcs(signal_mol, range_profile)
        
        #calculate backscattering ratio
        R = rcs_profile / rcs_mol - 1

        beta_part_current = R * params["molecular_beta"]
        
        if debug:
            beta_part[idx,:] = beta_part_current
            backscattering_ratio[idx,:] = R            

        integral_beta_previous = trapz(beta_part_previous[range_profile < free_troposphere_height], dx=resolution)
        integral_beta_current = trapz(beta_part_current[range_profile < free_troposphere_height], dx=resolution)

        relative_diff_beta_current = np.abs(integral_beta_current - integral_beta_previous) / integral_beta_previous

        if np.abs(relative_diff_beta_current) < tolerance:            
            break
        else:
            beta_part_previous = beta_part_current
            relative_diff_beta_previous = relative_diff_beta_current            
            alpha_part_previous = refill_overlap(beta_part_current * lr_part, range_profile, full_overlap_height)

    #raise if no convergence
    # if idx == iterations - 1:
    #     raise ValueError("No convergence in iterative beta retrieval.")

    if debug:
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(10, 5), sharey=True)
        #Axis 1: plot rcs mol
        ax[0].plot(rcs_mol, range_profile, lw=2, label="mol")
        ax[0].plot(rcs_profile, range_profile, lw=2, label="part")
        ax[0].set_xscale("log")
        ax[0].set_ylim(0, 9000)
        ax[0].set_xlabel("RCS, [a.u.]")
        ax[0].legend()
        #Axis 4: plot backscattering ratio
        for i in range(iterations):
            ax[1].plot(backscattering_ratio[i,:], range_profile, lw=2, color=colors[i], label=f"{i}")        
        ax[1].set_xscale("linear")
        ax[1].set_ylim(0, 9000)
        ax[1].set_xlabel("R, [a.u.]")
        ax[1].legend(fontsize=5)
        #Axis 3: plot beta part
        if "particle_beta" in params:
            ax[2].plot(1e6*params["particle_beta"], range_profile, lw=2, label="synthetic")
        for i in range(iterations):
            ax[2].plot(1e6*beta_part[i,:], range_profile, lw=2, color=colors[i], label=f"{i}")        
        ax[2].set_xscale("linear")
        ax[2].set_ylim(0, 9000)
        ax[2].set_xlabel(r"$\beta$, [a.u.]")
        ax[2].legend(fontsize=5)
        #Axis 4: plot relative diff of beta
        for i in range(iterations):
            ax[3].plot(100*(beta_part[i,range_profile < free_troposphere_height] - params['particle_beta'][range_profile < free_troposphere_height])/params['particle_beta'][range_profile < free_troposphere_height], range_profile[range_profile < free_troposphere_height], lw=2, color=colors[i], label=f"{i}")
        ax[3].set_xscale("linear")
        ax[3].set_ylim(0, 9000)
        ax[3].set_xlim(-1, 0)
        ax[3].set_xlabel(r"$\varepsilon_{\beta_{p}}$, [%]")
        ax[3].legend(fontsize=5)

        fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'iterative_beta.png')        
    
    return beta_part_current



def klett_likely_bins(
    rcs_profile: np.ndarray[Any, np.dtype[np.float64]],
    att_mol_beta: np.ndarray[Any, np.dtype[np.float64]],
    heights: np.ndarray[Any, np.dtype[np.float64]],
    min_height: float = 1000,
    max_height: float = 1010,
    window_size: int = 50,
    step: int = 1,
):
    window_size // 2
    i_bin, e_bin = np.searchsorted(heights, [min_height, max_height])

    for i in np.arange(i_bin, e_bin + 1):
        rcs_profile / rcs_profile

    return rcs_profile



def find_lidar_ratio(
    rcs: np.ndarray[Any, np.dtype[np.float64]],
    height: np.ndarray[Any, np.dtype[np.float64]],
    beta_mol: np.ndarray[Any, np.dtype[np.float64]],
    lr_mol: float,
    reference_aod: float,
    mininum_height: float = 0,
    lr_initial: float = 50,
    max_iterations: int = 100,
    rel_diff_aod_percentage_threshold: float = 1,
    debugging: bool = False,
    klett_reference: Tuple[float, float] = (7000, 8000),
) -> Tuple[float, float | None, bool]:
    """Iterative process to find the lidar ratio (lr) that minimizes the difference between the measured and the calculated aerosol optical depth (aod).

    Args:
        rcs (np.ndarray[Any, np.dtype[np.float64]]): Range Corrected Signal
        height (np.ndarray[Any, np.dtype[np.float64]]): Range profile
        beta_mol (np.ndarray[Any, np.dtype[np.float64]]): Molecular backscattering coefficient profile
        lr_mol (float): Molecular lidar ratio
        reference_aod (float): Reference aerosol optical depth
        mininum_height (float, optional): Fullover height. Defaults to 0.
        lr_initial (float, optional): _description_. Defaults to 50.
        max_iterations (int, optional): _description_. Defaults to 100.
        rel_diff_aod_percentage_threshold (float, optional): _description_. Defaults to 1.
        debugging (bool, optional): _description_. Defaults to False.
        klett_reference (Tuple[float, float], optional): _description_. Defaults to (7000, 8000).

    Returns:
        Tuple[float, float | None, bool]: _description_
    """

    # Calculate range resolution
    range_resolution = np.median(np.diff(height)).item()

    # Initialize loop
    lr_, iter_, run, success = lr_initial, 0, True, False
    rel_diff_aod = None

    while run:
        iter_ = iter_ + 1

        # Calculate aerosol backscatter
        beta_ = klett_rcs(
            rcs, height, beta_mol, lr_part=lr_, lr_mol=lr_mol, reference=klett_reference
        )

        # Refill beta profile from minimum height to surface to avoid overlap influence
        beta_ = refill_overlap(beta_, height, fulloverlap_height=mininum_height)

        # Calculate aerosol optical depth
        aod_ = integrate.simpson(beta_ * lr_, dx=range_resolution)

        # Calculate relative difference between measured and calculated aod
        rel_diff_aod = 100 * (aod_ - reference_aod) / reference_aod

        if debugging:
            logger.debug(
                "lidar_ratio: %.1f | lidar_aod: %.3f| reference_aod: %.3f | relative_difference: %.1f%%"
                % (lr_, aod_, reference_aod, rel_diff_aod)
            )

        # Check convergence
        if np.abs(rel_diff_aod) > rel_diff_aod_percentage_threshold:
            if rel_diff_aod > 0:
                if lr_ < 20:
                    run = False
                    logger.warning("No convergence. LR goes too low.")
                else:
                    lr_ = lr_ - 1
            else:
                if lr_ > 150:
                    run = False
                    logger.warning("No convergence. LR goes too high.")
                else:
                    lr_ = lr_ + 1
        else:
            logger.info("LR found: %f" % lr_)
            run = False
            success = True

        # Check maximum number of iterations
        if iter_ == max_iterations:
            run = False
            logger.warning("No convergence. Too many iterations.")

    return lr_, rel_diff_aod, success # type: ignore

def iterative_beta_forward(
    rcs_profile: np.ndarray,
    calibration_factor: float,
    range_profile: np.ndarray,
    params: dict | ParamsDict,
    lr_part: float | np.ndarray = 45.0,  
    start_height: float | None = None,
    height_top: float | None = None,                  # ← NUEVO
    initial_particle_optical_depth: float = 0.0,
    layer_tolerance: float = 1e-4,
    max_iter_per_layer: int = 30,
    debug: bool = False
) -> np.ndarray:
    """Calculate aerosol backscattering using iterative algorithm verified with Li, D., Wu, Y., Gross, B., & Moshary, F. (2021). 
    Capabilities of an Automatic Lidar Ceilometer to Retrieve Aerosol Characteristics within the Planetary Boundary Layer. Remote Sensing, 13(18), 3626.

    Args:
        rcs_profile: perfil RCS (range-corrected signal)
        calibration_factor: factor de calibración lidar
        range_profile: perfil de alturas (m, ordenado ascendente)
        params: {'molecular_alpha', 'molecular_beta'}
        lr_part: lidar ratio (Sr)
        start_height: altura inicial z0 (si None, se toma la más baja)
        initial_particle_optical_depth: aerosol optical depth accumulated
            below ``start_height``. Use 0 when starting at the first bin.
        layer_tolerance: tolerancia relativa de convergencia por capa
        max_iter_per_layer: máximo número de iteraciones por capa
        debug: imprime progreso

    Returns:
        dict con:
            'beta_a' : perfil de retrodispersión de partículas
            'alpha_a': perfil de extinción de partículas
            'tau_a'  : profundidad óptica acumulada
    """
    z = np.array(range_profile, dtype=float)
    n = len(z)
    dz = np.gradient(z)
    
    lr_input = np.asanyarray(lr_part, dtype=float)
    
    # Creamos S_r: un array de longitud n. 
    # Si lr_part es escalar, llenamos el array con ese valor.
    # Si ya es un array, lo usamos tal cual.
    if lr_input.ndim == 0:
        S_r = np.full(n, float(lr_input))
    else:
        S_r = lr_input

    # --- 0. PREPARACIÓN DE ÍNDICES Y ARRAYS ---
    # Convertir alturas (float) a índices (int)
    if start_height is not None:
        idx_start = (np.abs(z - start_height)).argmin()
    else:
        idx_start = 0  # Empezar desde el suelo si no se define
        
    if initial_particle_optical_depth < 0:
        raise ValueError("initial_particle_optical_depth must be non-negative.")
        
    # Inicializar arrays de resultado con ceros
    beta_a = np.zeros(n)
    alpha_a = np.zeros(n)
    tau_a = np.zeros(n)
        
    molecular_transmittance =  transmittance(params["molecular_alpha"], z) 
    molecular_transmittance_zero = molecular_transmittance[idx_start]
    beta_mol_zero = params["molecular_beta"][idx_start]
    
    beta_att = rcs_profile / calibration_factor
    beta_att_zero = beta_att[idx_start]
    
    particle_transmittance_zero_squared = np.exp(
        -2 * initial_particle_optical_depth
    )
    beta_a[idx_start] = (
        beta_att_zero
        / (molecular_transmittance_zero**2 * particle_transmittance_zero_squared)
        - beta_mol_zero
    )
    alpha_a[idx_start] = S_r[idx_start] * beta_a[idx_start]
    tau_a[idx_start] = initial_particle_optical_depth

    if debug:
        logger.debug(
            f"Init z0 index={idx_start} ({z[idx_start]:.1f}m): Beta_a={beta_a[idx_start]:.2e}"
        )

    # --- 2. BUCLE ITERATIVO CAPA POR CAPA (Bottom-Up) ---
    # Empezamos desde la siguiente capa (idx_start + 1) hacia arriba
    for i in range(idx_start + 1, n):
        
        # --- COMPROBACIÓN H_TOP (Diagrama: rombo zi >= htop) ---
        # Si hemos superado la altura máxima, terminamos el proceso.
        if height_top is not None and z[i] > height_top:
            if debug:
                logger.debug(f"Reached h_top at {z[i]:.1f}m. Stopping.")
            break
        
        # Variables de la capa anterior (i-1) fijas
        tau_prev = tau_a[i-1]
        alpha_prev_layer = alpha_a[i-1]
        
        # Guess inicial para la capa actual (k=0): usamos el valor de la capa anterior
        current_alpha = alpha_a[i-1] 
        current_dz = dz[i] # o z[i] - z[i-1]
        
        # Constantes para esta capa
        tm_sq_i = molecular_transmittance[i]**2
        beta_att_i = beta_att[i]
        beta_mol_i = params["molecular_beta"][i]
        
        converged = False
        
        # Bucle de convergencia 'k' (Cuadro central del diagrama)
        for k in range(max_iter_per_layer):
            
            alpha_prev_k = current_alpha
            
            # --- USO DE INTEGRAL DE PYTHON (trapz) ---
            # Creamos un pequeño array con los dos puntos del segmento actual
            # punto anterior (fijo) y punto actual (variable)
            segment_alpha = [alpha_prev_layer, current_alpha]
            
            # Calculamos el área bajo la curva solo de este segmento
            delta_tau = trapz(segment_alpha, dx=current_dz)
            
            # Sumamos al acumulado anterior
            current_tau = tau_prev + delta_tau
            
            # --------------------------------------------
            
            # [Ec 2] Transmisión^2
            ta_sq_i = np.exp(-2 * current_tau)

            # [Ec 3] Beta(zi, k)
            # Ecuación lidar invertida: beta_tot = beta_att / (Tm^2 * Ta^2)
            beta_tot_corrected = beta_att_i / (tm_sq_i * ta_sq_i)
            current_beta = beta_tot_corrected - beta_mol_i
            
            # [Ec 4] Alpha(zi, k) = Sr * Beta(zi, k)
            current_alpha = S_r[i] * current_beta
            
            # [Ec 5] Chequeo de tolerancia
            # |alpha_new - alpha_old| / alpha_new < 0.01%
            if abs(current_alpha) > 1e-12: # Evitar división por cero si alpha es 0
                rel_diff = abs(current_alpha - alpha_prev_k) / abs(current_alpha)
            else:
                rel_diff = 0.0
            
            if rel_diff < layer_tolerance:
                converged = True
                break
        
        # Al salir del bucle k (por convergencia o max_iter), guardamos valores
        beta_a[i] = current_beta
        alpha_a[i] = current_alpha
        tau_a[i] = current_tau
        
        if debug and not converged:
            logger.warning(
                f"Layer {i} ({z[i]:.1f}m) did not converge. Diff: {rel_diff:.2e}"
            )

    # Devolver el perfil de retrodispersión
    return beta_a
 
