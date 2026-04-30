import numpy as np
import xarray as xr

from lidarpy.atmo import atmo
from lidarpy.atmo.rayleigh import molecular_properties
from lidarpy.utils.types import ParamsDict
from lidarpy.utils.utils import extrapolate_beta_with_angstrom
from lidarpy.utils.utils import sigmoid


def generate_particle_properties(
    ranges: np.ndarray,
    wavelength: float,
    ae: float | tuple[float, float] = (1.5, 0),
    lr: float | tuple[float, float] = (75, 45),
    synthetic_beta: float | tuple[float, float] = (2.5e-6, 2.0e-6),
    sigmoid_edge: float | tuple[float, float] = (2500, 5000),
) -> tuple[ np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        ranges (np.ndarray): ranges
        wavelength (float): wavelength
        ae (float | tuple[float, float], optional): Angstrom exponent. ae[0] is fine mode and ae[1] is coarse mode. Float value means both modes have the same ae. 
        fine_beta532 (float, optional): fine-mode backscatter coefficient at 532 nm. 
        coarse_beta532 (float, optional): coarse-mode backscatter coefficient at 532 nm. 

    Returns:
        np.ndarray: particle backscatter coefficient profile
    """
    if isinstance(ae, tuple):
        fine_ae = ae[0]
        coarse_ae = ae[1]
    else:
        fine_ae = ae
        coarse_ae = ae

    if isinstance(lr, tuple):
        fine_lr = lr[0]
        coarse_lr = lr[1]
    else:
        fine_lr = lr
        coarse_lr = lr

    if isinstance(sigmoid_edge, tuple):
        sigmoid_edge_fine = sigmoid_edge[0]
        sigmoid_edge_coarse = sigmoid_edge[1]
    else:
        sigmoid_edge_fine = sigmoid_edge
        sigmoid_edge_coarse = sigmoid_edge

    if isinstance(synthetic_beta, tuple):
        fine_beta532 = synthetic_beta[0]
        coarse_beta532 = synthetic_beta[1]
    else:
        fine_beta532 = synthetic_beta
        coarse_beta532 = synthetic_beta

    beta_part_fine_532 = sigmoid(
        ranges, sigmoid_edge_fine, 1 / 60, coeff=-fine_beta532, offset=fine_beta532
    )
    beta_part_coarse_532 = sigmoid(
        ranges,
        sigmoid_edge_coarse,
        1 / 60,
        coeff=-coarse_beta532,
        offset=coarse_beta532,
    )

    beta_part_fine = extrapolate_beta_with_angstrom(
        beta_part_fine_532, 532, wavelength, fine_ae
    )

    beta_part_coarse = extrapolate_beta_with_angstrom(
        beta_part_coarse_532, 532, wavelength, coarse_ae
    )

    beta_total = beta_part_fine + beta_part_coarse

    alpha_part_fine = fine_lr * beta_part_fine
    alpha_part_coarse = coarse_lr * beta_part_coarse

    alpha_total = alpha_part_fine + alpha_part_coarse

    return (
        beta_part_fine,
        beta_part_coarse,
        beta_total,
        alpha_part_fine,
        alpha_part_coarse,
        alpha_total,
    )


def synthetic_signals(
    ranges: np.ndarray,
    wavelengths: float | tuple[float, float] = 532,
    wavelength_raman: float | None = None,
    overlap_midpoint: float = 500,
    overlap_slope: float = 1 / 150,
    k_lidar: float | tuple[float, float] = (1e11, 1e10), #El coeficiente de ganancias es igual que el coeficiente de k_lidar. Representan la elastica y la raman 
    ae: float | tuple[float, float] = (1.5, 0), #Modo fino y modo grueso
    lr: float | tuple[float, float] = (75, 45),
    synthetic_beta: float | tuple[float, float] = (2.5e-6, 2.0e-6), #Modo fino y modo grueso
    sigmoid_edge: float | tuple[float, float] = (2500, 5000), #Modo fino y modo grueso
    force_zero_aer_after_bin: int | None = None,
    paralell_perpendicular_ratio: float = 0.33, #despo_particle
    meteo_profiles: tuple[np.ndarray, np.ndarray] | None = None,
    apply_overlap: bool = True,
    number_of_initial_nan_values: int = 10, #Number of values to eliminate from the signal at the beginning
) -> tuple[xr.DataArray, xr.DataArray | None, ParamsDict]:
    """It generates synthetic lidar signal.

    Args:
        ranges (np.ndarray): Range
        wavelengths (float | tuple[float, float], optional): Elastic wavelength, or
            ``(elastic, Raman)`` wavelengths. Defaults to 532.
        wavelength_raman (float | None, optional): Raman wavelength used when
            ``wavelengths`` is scalar. Defaults to None, which generates only
            the elastic signal.
        overlap_midpoint (float, optional): Inflexion point of the overlap function. Defaults to 500.
        overlap_slope (float, optional): Slope of the overlap function. Defaults to 1 / 150.
        k_lidar (float | tuple[float, float], optional): Absolute lidar factor calibration (elastic, raman). Defaults to (1e11, 1e10).
        ae (float | tuple[float, float], optional): Angstrom exponent (fine, coarse). Defaults to (1.5, 0).
        lr (float | tuple[float, float], optional): Lidar ratio (fine, coarse). Defaults to (75, 45).
        synthetic_beta (float | tuple[float, float], optional): Synthetic backscatter coefficient (fine, coarse). Defaults to (2.5e-6, 2.0e-6).
        sigmoid_edge (float | tuple[float, float], optional): Transition layer in particle backscatter coefficient (fine, coarse). Defaults to (2500, 5000).
        force_zero_aer_after_bin (int | None, optional): Force zero aerosol after bin. Defaults to None.
        meteo_profiles (tuple[np.ndarray, np.ndarray] | None, optional): Meteorological profiles (pressure, temperature). Defaults to None.
        apply_overlap (bool, optional): Apply overlap function. Defaults to True.
        number_of_initial_nan_values (int, optional): Number of values to
            eliminate from the signal at the beginning. Defaults to 10.

    Returns:
        tuple[xr.DataArray, xr.DataArray | None, ParamsDict]: Elastic signal,
        Raman signal when requested, and synthetic parameters.
    """
    
    z = ranges

    # Overlap
    if apply_overlap:
        overlap = sigmoid(
            z.astype(np.float64),
            overlap_midpoint,
            overlap_slope,
            offset=0.,
        )
        overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
        overlap -= overlap[0]  # asegura que comience exactamente en 0
        

    else:
        overlap = np.ones_like(z)

    if isinstance(lr, float):
        lr = (lr, lr)
    if isinstance(ae, float):
        ae = (ae, ae)
    if isinstance(synthetic_beta, float):
        synthetic_beta = (synthetic_beta, synthetic_beta)

    if isinstance(k_lidar, float):
        k_lidar_elastic = k_lidar
        k_lidar_raman = k_lidar
    else:
        k_lidar_elastic, k_lidar_raman = k_lidar

    if isinstance(wavelengths, tuple):
        wavelength = wavelengths[0]
        wavelength_raman = wavelengths[1]
    else:
        wavelength = wavelengths

    # Check temperature and pressure profiles
    if meteo_profiles is None:
        P, T, _ = atmo.standard_atmosphere(z)
    else:
        # check length of meteo_profiles with z
        if len(meteo_profiles[0]) != len(z):
            raise ValueError("Length of meteo_profiles must be equal to length of z")
        else:
            P = meteo_profiles[0]
            T = meteo_profiles[1]

    # Generate molecular profiles for elastic wavelength
    mol_properties = molecular_properties(wavelength, P, T, heights=z)
    
    (
        _,
        _,
        beta_part,
        alpha_part_fine,
        alpha_part_coarse,
        alpha_part,
    ) = generate_particle_properties(
        ranges, wavelength, ae=ae, lr=lr, synthetic_beta=synthetic_beta, sigmoid_edge=sigmoid_edge
    )
  
    # Elastic transmittance
    T_elastic = atmo.transmittance(mol_properties["molecular_alpha"] + alpha_part, z)
    
    # Elastic signal
    P_elastic = (
        k_lidar_elastic
        * (overlap / z**2)
        * (mol_properties["molecular_beta"] + beta_part)
        * T_elastic**2
    )
    # Eliminate first N values
    P_elastic[:number_of_initial_nan_values] = np.nan
    
    clean_attenuated_molecular_beta = (
        mol_properties["molecular_beta"]
        * atmo.transmittance(mol_properties["molecular_alpha"], z) ** 2
    )

    # Save parameters to create synthetic elastic signal
    params: ParamsDict = {
        "particle_beta": beta_part,
        "particle_alpha": alpha_part,
        "molecular_beta": mol_properties["molecular_beta"],
        "molecular_alpha": mol_properties["molecular_alpha"],
        "lidar_ratio": lr,
        "attenuated_molecular_backscatter": clean_attenuated_molecular_beta,
        "transmittance_elastic": T_elastic,
        "overlap": overlap,
        "k_lidar": k_lidar,
        "particle_angstrom_exponent": ae,
        "synthetic_beta": synthetic_beta,
        "temperature": T,
        "pressure": P,
        "ranges": z,
    }

    # Raman signal
    if wavelength_raman is not None:
        # Generate molecular profiles for raman wavelength
        mol_properties_raman = molecular_properties(wavelength_raman, P, T, heights=z)

        # Alpha particle raman
        alpha_part_fine_raman = alpha_part_fine * (wavelength_raman / wavelength) ** (
            -ae[0]
        )
        alpha_part_coarse_raman = alpha_part_coarse * (
            wavelength_raman / wavelength
        ) ** (-ae[1])
        alpha_part_raman = alpha_part_fine_raman + alpha_part_coarse_raman

        # Transmittance Raman
        T_raman = atmo.transmittance(
            mol_properties_raman["molecular_alpha"] + alpha_part_raman, z
        )
        P_raman = (
            k_lidar_raman
            * (overlap / z**2)
            * mol_properties_raman["molecular_beta"]
            * T_elastic
            * T_raman
        )
        # Eliminate first N values
        P_raman[:number_of_initial_nan_values] = np.nan
        
        clean_attenuated_molecular_beta_raman: xr.DataArray = (
            mol_properties_raman["molecular_beta"]
            * atmo.transmittance(mol_properties_raman["molecular_alpha"], z)
            * atmo.transmittance(mol_properties["molecular_alpha"], z)
        )

        params["molecular_alpha_raman"] = mol_properties_raman["molecular_alpha"]
        params["molecular_beta_raman"] = mol_properties_raman["molecular_beta"]
        params["attenuated_molecular_backscatter_raman"] = (
            clean_attenuated_molecular_beta_raman,
        )
        params["transmittance_raman"] = T_raman
        params["overlap"] = overlap

    else:
        P_raman = None

    if force_zero_aer_after_bin is not None:
        alpha_part[force_zero_aer_after_bin:] = 0
        beta_part[force_zero_aer_after_bin:] = 0
        
    return P_elastic, P_raman, params


def synthetic_signals_despo(
    ranges: np.ndarray,
    wavelength: float = 532,
    overlap_midpoint: float = 500,
    overlap_slope: float = 1 / 50, 
    C_lidar_parallel: float  = 1e5, # po*tau/2*c*At. Fijamos Po * tau = 82.4 mJ. La estoy poniendo más baja de lo que debería ser
    C_lidar_perpendicular: float = 1e5, 
    phi: float = 90.0,
    reflectance_transmittance_s_path: tuple[float, float] = (0.99, 0.01),
    reflectance_transmittance_p_path: tuple[float, float] = (0.05, 0.95),
    photomultipliers_gains: tuple[float, float] = (1.4e5, 2e6), #Hamamatsu R9880U SERIES
    ae: float | tuple[float, float] = (1.5, 0),
    lr: float | tuple[float, float] = (75, 45),
    synthetic_beta: float | tuple[float, float] = (2.5e-6, 2.0e-6),
    sigmoid_edge: float | tuple[float, float] = (2500, 5000),
    force_zero_aer_after_bin: int | None = None,
    despo_particle: float = 0.33, 
    meteo_profiles: tuple[np.ndarray, np.ndarray] | None = None,
    apply_overlap: bool = True,
    number_of_initial_nan_values: int = 10, #Numero de valores para eliminar de la señal al inicio
) -> tuple[xr.DataArray, xr.DataArray, ParamsDict]:
    """It generates polarized synthetic lidar signal.

    Args:
        ranges (np.ndarray): Range
        wavelength (float, optional): Elastic wavelength. Defaults to 532.
        overlap_midpoint (float, optional):  Inflexion point of the overlap function. Defaults to 500.
        C_lidar_parallel (float, optional): Lidar factor calibration for parallel component without gain. Defaults to 1e11.
        C_lidar_perpendicular (float, optional): Lidar factor calibration for perpendicular component without. Defaults to 1e11.
        phi (float, optional): Angle between the plane of laser polarization and the incident plane of PBC. Defaults to 90.0 (s-path = parallel-path and p-path=perpendicular path).
        reflectance_transmittance_s_path (tuple[float, float], optional): Reflectance and transmittance for s-path. Defaults to (0.99, 0.01).
        reflectance_transmittance_p_path (tuple[float, float], optional): Reflectance and transmittance for p-path. Defaults to (0.05, 0.95).
        photomultipliers_gains (tuple[float, float], optional): Photomultipliers gains for signals (reflected, transmitted). Defaults to (0.9, 0.9).
        The cocient of gains (reflected/transmitted) is eta, the lidar depolarization calibration factor.
        ae (float | tuple[float, float], optional): Ángstrong exponent (fine, coarse). Defaults to (1.5, 0).
        lr (float | tuple[float, float], optional): Lidar ratio (fine, coarse). Defaults to (75, 45).
        synthetic_beta (float | tuple[float, float], optional): Synthetic backscatter coefficient (fine, coarse). Defaults to (2.5e-6, 2.0e-6).
        sigmoid_edge (float | tuple[float, float], optional): Transition layer in particle backscatter coefficient (fine, coarse). Defaults to (2500, 5000).
        force_zero_aer_after_bin (int | None, optional): _description_. Defaults to None.
        despo_particle (float, optional): Linear particle despolarization ratio. Defaults to 0.33 (dust).
        meteo_profiles (tuple[np.ndarray, np.ndarray] | None, optional): _description_. Defaults to None.
        apply_overlap (bool, optional): _description_. Defaults to True.
        N (int, optional): Number of values to eliminate from the signal at the beginning. Defaults to 10 (approx 40 m).

    Returns:
        tuple[xr.DataArray, xr.DataArray, ParamsDict]: Reflected signal,
        transmitted signal, and synthetic parameters.
    """
    
    z = ranges

    # Overlap
    if apply_overlap:
        overlap = sigmoid(
            z.astype(np.float64),
            overlap_midpoint,
            overlap_slope,
            offset=0.,
        )
        overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
        overlap -= overlap[0]  # asegura que comience exactamente en 0

    else:
        overlap = np.ones_like(z)
        

    if isinstance(lr, float):
        lr = (lr, lr)
    if isinstance(ae, float):
        ae = (ae, ae)
    if isinstance(synthetic_beta, float):
        synthetic_beta = (synthetic_beta, synthetic_beta)

    if isinstance(C_lidar_parallel, float):
        C_lidar_elastic_parallel = C_lidar_parallel
        
    if isinstance(C_lidar_perpendicular, float):
        C_lidar_elastic_perpendicular = C_lidar_perpendicular
    
    # Check if reflectance_transmittance_s_path components sum 1
    if sum(reflectance_transmittance_s_path) != 1:
        raise ValueError("Sum of reflectance_transmittance_s_path must be equal to 1")
    
    # Check if reflectance_transmittance_p_path components sum 1
    if sum(reflectance_transmittance_p_path) != 1:
        raise ValueError("Sum of reflectance_transmittance_p_path must be equal to 1") 
    
    #Identify the reflectance and transmittance components for the parallel and perpendicular components of the cube
    reflectance_s_path = reflectance_transmittance_s_path[0]
    transmittance_s_path = reflectance_transmittance_s_path[1]
    
    reflectance_p_path = reflectance_transmittance_p_path[0]
    transmittance_p_path = reflectance_transmittance_p_path[1]  
    
    #Identify the photomultipliers ganancies for the parallel and perpendicular components of the cube
    gain_reflected_path = photomultipliers_gains[0]
    gain_transmitted_path = photomultipliers_gains[1] 
      

    # Check temperature and pressure profiles
    if meteo_profiles is None:
        P, T, _ = atmo.standard_atmosphere(z)
    else:
        # check length of meteo_profiles with z
        if len(meteo_profiles[0]) != len(z):
            raise ValueError("Length of meteo_profiles must be equal to length of z")
        else:
            P = meteo_profiles[0]
            T = meteo_profiles[1]

    # Generate molecular profiles for elastic wavelength
    mol_properties = molecular_properties(wavelength, P, T, heights=z, component="cabannes")
    beta_mol_total= mol_properties['molecular_beta']
    mol_despo = mol_properties['molecular_depolarization']
    
    #Calculate the backscatter coefficient for the parallel and perpendicular components
    beta_mol_parallel = beta_mol_total/(1+mol_despo)
    beta_mol_perpendicular= beta_mol_total - beta_mol_parallel
    
    # Generate particle profiles for elastic wavelength
    (
        _,
        _,
        beta_part,
        alpha_part_fine,
        alpha_part_coarse,
        alpha_part,
    ) = generate_particle_properties(
        ranges, wavelength, ae=ae, lr=lr, synthetic_beta=synthetic_beta, sigmoid_edge=sigmoid_edge
    )
    
    #Calculate the backscatter coefficient for the parallel and perpendicular components
    beta_part_parallel = beta_part/(1+despo_particle)
    beta_part_perpendicular= beta_part - beta_part_parallel
    

    # Elastic transmittance
    # T_elastic = np.exp(-cumulative_trapezoid(alpha_mol+ alpha_part, z, initial=0))  # type: ignore
    T_elastic = atmo.transmittance(mol_properties["molecular_alpha"] + alpha_part, z)
    
    # Elastic signal. T_elastic_parallel and T_elastic_perpendicular are same (aproximation).
    P_elastic_parallel = (
        C_lidar_elastic_parallel
        * (overlap / z**2)
        * (beta_mol_parallel+ beta_part_parallel)
        * T_elastic**2 
    )
    
    P_elastic_perpendicular = (
        C_lidar_elastic_perpendicular
        * (overlap / z**2)
        * (beta_mol_perpendicular+ beta_part_perpendicular)
        * T_elastic**2
    )
    
    # Eliminate first N values
    P_elastic_parallel[:number_of_initial_nan_values] = np.nan
    P_elastic_perpendicular[:number_of_initial_nan_values] = np.nan
    
    #Calculate the signal components with respect to incident plane of the PBC
    
    Ps = P_elastic_parallel * np.sin(np.radians(phi))**2 + P_elastic_perpendicular * np.cos(np.radians(phi))**2
    Pp = P_elastic_parallel * np.cos(np.radians(phi))**2 + P_elastic_perpendicular * np.sin(np.radians(phi))**2
    
    
    #Calculate the reflected and transmitted signals
    P_elastic_reflected = (Pp*reflectance_p_path + Ps*reflectance_s_path)*gain_reflected_path
    P_elastic_transmitted = (Pp*transmittance_p_path + Ps*transmittance_s_path)*gain_transmitted_path

    #Calculate the volumic depolarization ratio
    despo_volumic= (beta_mol_perpendicular+beta_part_perpendicular)/(beta_mol_parallel+beta_part_parallel) 
    
    #Calculate the attenuation molecular backscatter
    clean_attenuated_molecular_beta_total = (
        beta_mol_total
        * atmo.transmittance(mol_properties["molecular_alpha"], z) ** 2
    )
    
    clean_attenuated_molecular_beta_parallel = (
        beta_mol_parallel
        * atmo.transmittance(mol_properties["molecular_alpha"], z) ** 2
    )
    clean_attenuated_molecular_beta_perpendicular = (
        beta_mol_perpendicular
        * atmo.transmittance(mol_properties["molecular_alpha"], z) ** 2
    )

    #Calculate the particle depolarization ratio
    despo_particle_profile = np.ones_like(beta_part_parallel)*np.nan
    np.divide(
        beta_part_perpendicular,
        beta_part_parallel,
        out=despo_particle_profile,
        where=np.logical_and.reduce([beta_part_parallel != 0, ~np.isnan(beta_part_perpendicular), np.isfinite(beta_part_perpendicular)]),
    )
    # Ensure no NaN or inf remain
    despo_particle_profile = np.nan_to_num(despo_particle_profile, nan=0.0, posinf=0.0, neginf=0.0)
    
    #Calculate the molecular depolarization ratio
    despo_molecular= mol_despo
    
    #Calculate the backscattering ratio R
    R=(beta_mol_total + beta_part)/(beta_mol_total)
    
    #Calculate the depolarization calibration factor eta
    eta= (gain_reflected_path)/(gain_transmitted_path)
    
    
    if phi == 0.0:
        reflectance_parallel_path = reflectance_p_path
        reflectance_perpendicular_path = reflectance_s_path
        transmittance_parallel_path = transmittance_p_path
        transmittance_perpendicular_path = transmittance_s_path
    
    if phi == 90.0:
        reflectance_parallel_path = reflectance_s_path
        reflectance_perpendicular_path = reflectance_p_path
        transmittance_parallel_path = transmittance_s_path
        transmittance_perpendicular_path = transmittance_p_path
        
    else :
        reflectance_parallel_path = np.nan
        reflectance_perpendicular_path = np.nan
        transmittance_parallel_path = np.nan
        transmittance_perpendicular_path = np.nan
    
    # Save parameters to create synthetic elastic signal
    params: ParamsDict = {
        "ranges": ranges,
        "particle_beta_total": beta_part,
        "particle_beta_parallel": beta_part_parallel,
        "particle_beta_perpendicular": beta_part_perpendicular,
        "particle_alpha": alpha_part,
        "molecular_beta_total": beta_mol_total,
        "molecular_beta_parallel": beta_mol_parallel,
        "molecular_beta_perpendicular": beta_mol_perpendicular,
        "molecular_alpha": mol_properties["molecular_alpha"],
        "lidar_ratio": lr,
        "attenuated_molecular_backscatter_total": clean_attenuated_molecular_beta_total,
        "attenuated_molecular_backscatter_parallel": clean_attenuated_molecular_beta_parallel,
        "attenuated_molecular_backscatter_perpendicular": clean_attenuated_molecular_beta_perpendicular,
        "transmittance_elastic": T_elastic,
        "overlap": overlap,
        "C_lidar_elastic_parallel": C_lidar_elastic_parallel,
        "C_lidar_elastic_perpendicular": C_lidar_elastic_perpendicular,
        "particle_angstrom_exponent": ae,
        "synthetic_beta": synthetic_beta,
        "temperature": T,
        "pressure": P,
        "despolarization_particle": despo_particle_profile,
        "despolarization_volumic": despo_volumic,
        "despolarization_molecular": despo_molecular,
        "backscattering_ratio" : R,
        "angle_laser_PBC": phi,
        "gain_reflected_path": gain_reflected_path,
        "gain_transmitted_path": gain_transmitted_path,
        "reflectance_parallel_path": reflectance_parallel_path,
        "reflectance_perpendicular_path": reflectance_perpendicular_path,
        "transmittance_parallel_path": transmittance_parallel_path,
        "transmittance_perpendicular_path": transmittance_perpendicular_path,
        "signal_parallel_path": P_elastic_parallel,
        "signal_perpendicular_path": P_elastic_perpendicular,
        "signal_s_path": Ps,
        "signal_p_path": Pp, 
        "eta": eta,
    }
    
    if force_zero_aer_after_bin is not None:
        alpha_part[force_zero_aer_after_bin:] = 0
        beta_part[force_zero_aer_after_bin:] = 0


    return P_elastic_reflected, P_elastic_transmitted, params


def synthetic_signals_2D(
    ranges: np.ndarray,
    time: np.ndarray,
    wavelength: float  = 532,
    overlap_midpoint: float = 500,
    overlap_slope: float = 1 / 50,
    k_lidar: float = 1e11,
    ae: float | tuple[float, float] = (1.5, 0),
    lr: float | tuple[float, float] = (75, 45),
    synthetic_beta: float | tuple[float, float] = (2.5e-6, 2.0e-6),
    sigmoid_edge: float | tuple[float, float] = (2000, 2000),
    force_zero_aer_after_bin: int | None = None,
    meteo_profiles: tuple[np.ndarray, np.ndarray] | None = None,
    apply_overlap: bool = True,
    number_of_initial_nan_values: int = 10,
    period_entrainment_zone: float = 10,
    amplitude_entrainment_zone: tuple[float, float] = (200, 200),
    abl_growth_start_time: float = 720,
    abl_day_growth_percentage: float = 0.8,
    variable_intensity: bool = False,
    variable_abl_top: bool = False,
    abl_linear_growth: bool = False,
    abl_gradual_growth: bool = False,
    apply_entrainment_zone: bool = False,

) -> tuple[xr.DataArray, ParamsDict]:
    """It generates a quicklook of synthetic elastic lidar signal 2D (time, range).
    Args:
        ranges (np.ndarray): Range
        time (np.ndarray): Time
        wavelength (float, optional): Wavelength. Defaults to 532.
        overlap_midpoint (float, optional): Inflexion point of the overlap function. Defaults to 500.
        overlap_slope (float, optional): Slope of the overlap function. Defaults to 1 / 150.
        k_lidar (float, optional): Absolute lidar factor calibration. Defaults to 1e11.
        ae (float | tuple[float, float], optional): Angstrom exponent (fine, coarse). Defaults to (1.5, 0).
        lr (float | tuple[float, float], optional): Lidar ratio (fine, coarse). Defaults to (75, 45).
        synthetic_beta (float | tuple[float, float], optional): Synthetic backscatter coefficient (fine, coarse). Defaults to (2.5e-6, 2.0e-6).
        sigmoid_edge (float | tuple[float, float], optional): Transition layer in particle backscatter coefficient (fine, coarse). Defaults to (2500, 5000).
        force_zero_aer_after_bin (int | None, optional): Force zero aerosol after bin. Defaults to None.
        meteo_profiles (tuple[np.ndarray, np.ndarray] | None, optional): Meteorological profiles (pressure, temperature). Defaults to None.
        apply_overlap (bool, optional): Apply overlap function. Defaults to True.
        number_of_initial_nan_values (int, optional): Number of values to eliminate from the signal at the beginning. Defaults to 10.
        period_entrainment_zone (float, optional): Period of the entrainment zone oscillation in minutes. Defaults to 10.
        amplitude_entrainment_zone (tuple[float, float], optional): Amplitude of the entrainment zone oscillation of the backscatter in meters (fine, coarse). Defaults to (200, 200).
        abl_growth_start_time (float, optional): Time when the ABL growth starts in minutes. Defaults to 720.
        abl_day_growth_percentage (float, optional): Percentage of the ABL growth in the day. Defaults to 0.8.
        variable_intensity (bool, optional): If True, the intensity of the aerosol backscatter coefficient increases linearly with time. Defaults to False.
        variable_abl_top (bool, optional): If True, the ABL top increases with time. Defaults to False.
        abl_linear_growth (bool, optional): If True, the ABL top increases linearly with time. Defaults to False.
        abl_gradual_growth (bool, optional): If True, the ABL top increases gradually with time following a sigmoid function. Defaults to False.
        apply_entrainment_zone (bool, optional): If True, the ABL top oscillates with time. Defaults to False.
        
        Returns:
        tuple[xr.DataArray, ParamsDict]: Elastic signal and parameters."""

    z = ranges
    t = time

    # Lógica de superposición (overlap)
    if apply_overlap:
        overlap = sigmoid(z.astype(np.float64), overlap_midpoint, overlap_slope, offset=0.)
        overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
        overlap -= overlap[0]  # asegura que comience exactamente en 0

    else:
        overlap = np.ones_like(z)

    # Manejo de parámetros
    if isinstance(lr, float):
        lr = (lr, lr)
    if isinstance(ae, float):
        ae = (ae, ae)
    if isinstance(synthetic_beta, float):
        synthetic_beta = (synthetic_beta, synthetic_beta)

    # Verificación de perfiles meteorológicos
    if meteo_profiles is None:
        P, T, _ = atmo.standard_atmosphere(z)
    else:
        if len(meteo_profiles[0]) != len(z):
            raise ValueError("Length of meteo_profiles must be equal to length of z")
        else:
            P = meteo_profiles[0]
            T = meteo_profiles[1]

    # Generación de perfiles moleculares
    mol_properties = molecular_properties(wavelength, P, T, heights=z)

    # Inicializar matrices para las propiedades de partículas
    beta_part2D = np.zeros((len(t), len(z)))
    alpha_part2D = np.zeros((len(t), len(z)))
    
    # Inicializar la matriz de señales
    P_elastic2D = np.zeros((len(t), len(z)))  
    T_elastic2D = np.zeros((len(t), len(z)))
    
    # Asegurar que overlap es 2D
    if overlap.ndim == 1:
        overlap2D = np.tile(overlap[np.newaxis, :], (len(t), 1))

    # Asegurar que molecular_beta es 2D
    # Asegurar que molecular_beta es 2D sin modificar el Dataset original
    molecular_beta = mol_properties["molecular_beta"].values
    molecular_alpha = mol_properties["molecular_alpha"].values

    if molecular_beta.ndim == 1:
        molecular_beta2D = np.tile(molecular_beta[np.newaxis, :], (len(t), 1))
        
    if molecular_alpha.ndim == 1:
        molecular_alpha2D = np.tile(molecular_alpha[np.newaxis, :], (len(t), 1))
    
    
    period = period_entrainment_zone
    amplitude = amplitude_entrainment_zone    
    omega = 2*np.pi/period    
    
    for i, t_i in enumerate(t):
    
        # Calcular intensidad
        if variable_intensity:
            beta_scale = t_i / (t[-1] * 5)
            beta_vals = (
                synthetic_beta[0] * beta_scale,
                synthetic_beta[-1] * beta_scale
            )
        else:
            beta_vals = (synthetic_beta[0], synthetic_beta[-1])

        # Calcular altura ABL (borde sigmoidal)
        if variable_abl_top:
            if abl_linear_growth:
                abl_top = (
                    sigmoid_edge[0] * t_i / t[-1],
                    sigmoid_edge[-1] * t_i / t[-1])
                
            elif abl_gradual_growth:     
                transition_factor = sigmoid(t_i, x0=abl_growth_start_time, 
                                    k=1/100, coeff=abl_day_growth_percentage, offset=1)
                
                abl_top = (
                    sigmoid_edge[0] * transition_factor,
                    sigmoid_edge[-1] * transition_factor)
                
            else:
                raise ValueError("ABL top is variable, but neither linear growth or gradual growth was selected.")
                
        else:
            abl_top = (sigmoid_edge[0], sigmoid_edge[-1])
        
        # Añadir oscilación de la cima de la ABL    
        if apply_entrainment_zone:
            abl_oscilation = (
                amplitude[0] * np.cos(omega * t_i),
                amplitude[-1] * np.cos(omega * t_i))
        else:
           abl_oscilation = (0, 0)
            
        edge_vals = (
                abl_oscilation[0] + abl_top[0],
                abl_oscilation[-1] + abl_top[-1]) 
        
        # Llamada única a la función
        (_, _, beta_part2D[i, :], _, _, alpha_part2D[i, :]) = generate_particle_properties(
            ranges,
            wavelength,
            ae,
            lr,
            beta_vals,
            edge_vals
        )


        T_elastic2D[i, :] = atmo.transmittance(molecular_alpha2D[i, :] + alpha_part2D[i, :], z) 
        
        P_elastic2D[i, :] = (
            k_lidar * (overlap2D[i, :] / z**2) * T_elastic2D[i, :] ** 2 * (beta_part2D[i, :] + molecular_beta2D[i, :])
        )
    
        

    P_elastic2D[:, :number_of_initial_nan_values] = np.nan  # Establecer a NaN los primeros N valores
    
        # Diccionario con las variables y sus nombres
    data_dict = {
        "beta_part": beta_part2D,
        "alpha_part": alpha_part2D,
        "molecular_alpha": molecular_alpha2D,
        "molecular_beta": molecular_beta2D,
        "transmittance": T_elastic2D,
        "overlap": overlap2D
    }

    # Crear los DataArray de manera más eficiente
    data_xr = {}
    for name, data in data_dict.items():
        data_xr[name] = xr.DataArray(
            data,
            dims=["time", "range"],
            coords={
                "time": t, 
                "range": z
            },
            name=name
        )

    
    # Guardar parámetros
    params: ParamsDict = {
        "particle_beta2D": data_xr["beta_part"],
        "particle_alpha2D": data_xr["alpha_part"],
        "molecular_beta2D": data_xr["molecular_beta"],
        "molecular_beta": molecular_beta,
        "molecular_alpha2D": data_xr["molecular_alpha"],
        "molecular_alpha": molecular_alpha,
        "transmittance_elastic2D": data_xr["transmittance"],
        "lidar_ratio": lr,  
        "k_lidar": k_lidar,
        "particle_angstrom_exponent": ae,
        "synthetic_beta": synthetic_beta,
        "temperature": T,
        "pressure": P,
        "overlap2D": data_xr["overlap"],
        "overlap": overlap,
        "ranges": z,
        "time": t,
    }

    if force_zero_aer_after_bin is not None:
        alpha_part2D[:, force_zero_aer_after_bin:] = 0
        beta_part2D[:, force_zero_aer_after_bin:] = 0
        
    # Convertir a xarray.DataArray
    P_elastic_xarray = xr.DataArray(
        P_elastic2D,
        dims=["time", "range"],
        coords={
            "time": t, 
            "range": z
        },
        name="LIDAR_signal"
    )

    return P_elastic_xarray, params

def synthetic_raman_signals_2D(
    ranges: np.ndarray,
    time: np.ndarray,
    wavelength: float  = 532,
    wavelength_raman: float = 531,
    overlap_midpoint: float = 500,
    overlap_slope: float = 1 / 50,
    k_lidar_raman: float = 1e10,
    ae: float | tuple[float, float] = (1.5, 0),
    lr: float | tuple[float, float] = (75, 45),
    synthetic_beta: float | tuple[float, float] = (2.5e-6, 2.0e-6),
    sigmoid_edge: float | tuple[float, float] = (2000, 2000),
    force_zero_aer_after_bin: int | None = None,
    meteo_profiles: tuple[np.ndarray, np.ndarray] | None = None,
    apply_overlap: bool = True,
    number_of_initial_nan_values: int = 10,
    period_entrainment_zone: float = 10,
    amplitude_entrainment_zone: tuple[float, float] = (200, 200),
    abl_growth_start_time: float = 720,
    abl_day_growth_percentage: float = 0.8,
    variable_intensity: bool = False,
    variable_abl_top: bool = False,
    abl_linear_growth: bool = False,
    abl_gradual_growth: bool = False,
    apply_entrainment_zone: bool = False,
    
) -> tuple[xr.DataArray, ParamsDict]:
    """It generates a quicklook of synthetic raman lidar signal 2D (time, range).
    Args:
        ranges (np.ndarray): Range
        time (np.ndarray): Time
        wavelength (float, optional): Wavelength. Defaults to 532.
        wavelength_raman (float, optional): Raman wavelength. Defaults to 531.
        overlap_midpoint (float, optional): Inflexion point of the overlap function. Defaults to 500.
        overlap_slope (float, optional): Slope of the overlap function. Defaults to 1 / 150.
        k_lidar_raman (float, optional): Absolute lidar factor calibration for raman channel. Defaults to 1e10.
        ae (float | tuple[float, float], optional): Angstrom exponent (fine, coarse). Defaults to (1.5, 0).
        lr (float | tuple[float, float], optional): Lidar ratio (fine, coarse). Defaults to (75, 45).
        synthetic_beta (float | tuple[float, float], optional): Synthetic backscatter coefficient (fine, coarse). Defaults to (2.5e-6, 2.0e-6).
        sigmoid_edge (float | tuple[float, float], optional): Transition layer in particle                 
        backscatter coefficient (fine, coarse). Defaults to (2500, 5000).
        force_zero_aer_after_bin (int | None, optional): Force zero aerosol after bin. Defaults to None.
        meteo_profiles (tuple[np.ndarray, np.ndarray] | None, optional): Meteorological profiles (pressure, temperature). Defaults to None.
        apply_overlap (bool, optional): Apply overlap function. Defaults to True.
        number_of_initial_nan_values (int, optional): Number of values to eliminate from the signal at the beginning. Defaults to 10.
        period_entrainment_zone (float, optional): Period of the entrainment zone oscillation in minutes. Defaults to 10.
        amplitude_entrainment_zone (tuple[float, float], optional): Amplitude of the entrainment zone oscillation of the backscatter in meters (fine, coarse). Defaults to (200, 200).
        abl_growth_start_time (float, optional): Time when the ABL growth starts in minutes. Defaults to 720.
        abl_day_growth_percentage (float, optional): Percentage of the ABL growth in the day. Defaults to 0.8. 
        variable_intensity (bool, optional): If True, the intensity of the aerosol backscatter coefficient increases linearly with time. Defaults to False.
        variable_abl_top (bool, optional): If True, the ABL top increases with time. Defaults to False.
        abl_linear_growth (bool, optional): If True, the ABL top increases linearly with time. Defaults to False.
        abl_gradual_growth (bool, optional): If True, the ABL top increases gradually with time following a sigmoid function. Defaults to False.
        apply_entrainment_zone (bool, optional): If True, the ABL top oscillates with time. Defaults to False.
        
        Returns:
        tuple[np.ndarray, ParamsDict]: Raman signal and parameters."""

    z = ranges
    t = time

    # Lógica de superposición (overlap)
    # Lógica de superposición (overlap)
    if apply_overlap:
        overlap = sigmoid(z.astype(np.float64), overlap_midpoint, overlap_slope, offset=0.)
        overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
        overlap -= overlap[0]  # asegura que comience exactamente en 0
        
        
    else:
        overlap = np.ones_like(z)

    # Manejo de parámetros
    if isinstance(lr, float):
        lr = (lr, lr)
    if isinstance(ae, float):
        ae = (ae, ae)
    if isinstance(synthetic_beta, float):
        synthetic_beta = (synthetic_beta, synthetic_beta)

    # Verificación de perfiles meteorológicos
    if meteo_profiles is None:
        P, T, _ = atmo.standard_atmosphere(z)
    else:
        if len(meteo_profiles[0]) != len(z):
            raise ValueError("Length of meteo_profiles must be equal to length of z")
        else:
            P = meteo_profiles[0]
            T = meteo_profiles[1]

    # Generación de perfiles moleculares
    mol_properties_elastic = molecular_properties(wavelength, P, T, heights=z)
    mol_properties_raman = molecular_properties(wavelength_raman, P, T, heights=z)

    # Asegurar que molecular_beta es 2D
    molecular_alpha_elastic = mol_properties_elastic["molecular_alpha"].values
    molecular_beta_raman = mol_properties_raman["molecular_beta"].values
    molecular_alpha_raman = mol_properties_raman["molecular_alpha"].values
    
    if molecular_alpha_elastic.ndim == 1:
        molecular_alpha_elastic2D = np.tile(molecular_alpha_elastic[np.newaxis, :], (len(t), 1))

    if molecular_beta_raman.ndim == 1:
        molecular_beta_raman2D = np.tile(molecular_beta_raman[np.newaxis, :], (len(t), 1))
        
    if molecular_alpha_raman.ndim == 1:
        molecular_alpha_raman2D = np.tile(molecular_alpha_raman[np.newaxis, :], (len(t), 1))
    
    
    # Inicializar matrices para las propiedades de partículas
    alpha_part_elastic2D = np.zeros((len(t), len(z)))
    alpha_part_fine_elastic2D = np.zeros((len(t), len(z)))
    alpha_part_coarse_elastic2D = np.zeros((len(t), len(z)))
    alpha_part_raman2D = np.zeros((len(t), len(z)))
    alpha_part_fine_raman2D = np.zeros((len(t), len(z)))
    alpha_part_coarse_raman2D = np.zeros((len(t), len(z)))
    beta_part_elastic2D = np.zeros((len(t), len(z)))
    
    # Inicializar la matriz de señales
    P_raman2D = np.zeros((len(t), len(z)))  
    T_elastic2D = np.zeros((len(t), len(z)))
    T_raman2D = np.zeros((len(t), len(z)))
    
    # Asegurar que overlap es 2D
    if overlap.ndim == 1:
        overlap2D = np.tile(overlap[np.newaxis, :], (len(t), 1))
    
    period = period_entrainment_zone
    amplitude = amplitude_entrainment_zone    
    omega = 2*np.pi/period    
    
    for i, t_i in enumerate(t):
    
        # Calcular intensidad
        if variable_intensity:
            beta_scale = t_i / (t[-1] * 5)
            beta_vals = (
                synthetic_beta[0] * beta_scale,
                synthetic_beta[-1] * beta_scale
            )
        else:
            beta_vals = (synthetic_beta[0], synthetic_beta[-1])

        # Calcular altura ABL (borde sigmoidal)
        if variable_abl_top:
            if abl_linear_growth:
                abl_top = (
                    sigmoid_edge[0] * t_i / t[-1],
                    sigmoid_edge[-1] * t_i / t[-1])
                
            elif abl_gradual_growth:     
                transition_factor = sigmoid(t_i, x0=abl_growth_start_time, 
                                    k=1/100, coeff=abl_day_growth_percentage, offset=1)
                
                abl_top = (
                    sigmoid_edge[0] * transition_factor,
                    sigmoid_edge[-1] * transition_factor)
                
            else:
                raise ValueError("ABL top is variable, but neither linear growth or gradual growth was selected.")
                
        else:
            abl_top = (sigmoid_edge[0], sigmoid_edge[-1])
            
        if apply_entrainment_zone:
            abl_oscilation = (
                amplitude[0] * np.cos(omega * t_i),
                amplitude[-1] * np.cos(omega * t_i)) 
            
        else:
           abl_oscilation = (0, 0)
            
        edge_vals = (
                abl_oscilation[0] + abl_top[0],
                abl_oscilation[-1] + abl_top[-1]) 
    
        (_, _, beta_part_elastic2D[i, :], alpha_part_fine_elastic2D[i,:], alpha_part_coarse_elastic2D[i,:], alpha_part_elastic2D[i, :]) = generate_particle_properties(
            ranges,
            wavelength,
            ae,
            lr,
            beta_vals,
            edge_vals
        )
        
        alpha_part_fine_raman2D[i, :] = alpha_part_fine_elastic2D[i, :] * (wavelength_raman / wavelength) ** (-ae[0])
        alpha_part_coarse_raman2D[i, :] = alpha_part_coarse_elastic2D[i, :] * (wavelength_raman / wavelength) ** (-ae[1])
        alpha_part_raman2D[i, :] = alpha_part_fine_raman2D[i, :] + alpha_part_coarse_raman2D[i, :]

        T_elastic2D[i, :] = atmo.transmittance(molecular_alpha_elastic2D[i, :] + alpha_part_elastic2D[i, :], z)
        T_raman2D[i, :] = atmo.transmittance(molecular_alpha_raman2D[i, :] + alpha_part_raman2D[i, :], z) 
        P_raman2D[i, :] = (
            k_lidar_raman * (overlap2D[i, :] / z**2) * T_raman2D[i, :] * T_elastic2D[i,:] * molecular_beta_raman2D[i, :]
        )
    
        

    P_raman2D[:, :number_of_initial_nan_values] = np.nan  # Establecer a NaN los primeros N valores
    
        # Diccionario con las variables y sus nombres
    data_dict = {
        "alpha_part_elastic": alpha_part_elastic2D,
        "alpha_part_raman": alpha_part_raman2D,
        "beta_part_elastic": beta_part_elastic2D,
        "molecular_alpha_elastic": molecular_alpha_elastic2D,
        "molecular_alpha_raman": molecular_alpha_raman2D,
        "molecular_beta_raman": molecular_beta_raman2D,
        "transmittance_elastic": T_elastic2D,
        "transmittance_raman": T_raman2D,
        "overlap": overlap2D
    }
    
    # Crear los DataArray de manera más eficiente
    data_xr = {}
    for name, data in data_dict.items():
        data_xr[name] = xr.DataArray(
            data,
            dims=["time", "range"],
            coords={
                "time": t, 
                "range": z
            },
            name=name
        )

    
    # Guardar parámetros
    params: ParamsDict = {
        "particle_alpha_elastic2D": data_xr["alpha_part_elastic"],
        "particle_alpha_raman2D": data_xr["alpha_part_raman"],
        "particle_beta_elastic2D": data_xr["beta_part_elastic"],
        "molecular_alpha_elastic2D": data_xr["molecular_alpha_elastic"],
        "molecular_alpha_raman2D": data_xr["molecular_alpha_raman"],
        "molecular_beta_raman2D": data_xr["molecular_beta_raman"],
        "transmittance_elastic2D": data_xr["transmittance_elastic"],
        "transmittance_raman2D": data_xr["transmittance_raman"],
        "lidar_ratio": lr,  
        "k_lidar_raman": k_lidar_raman,
        "particle_angstrom_exponent": ae,
        "synthetic_beta": synthetic_beta,
        "temperature": T,
        "pressure": P,
        "overlap": overlap,
        "overlap2D": data_xr["overlap"],
        "ranges": z,
        "time": t,
    }
    
    

    if force_zero_aer_after_bin is not None:
        alpha_part_elastic2D[:, force_zero_aer_after_bin:] = 0
        alpha_part_raman2D[:, force_zero_aer_after_bin:] = 0
        
    # Convertir a xarray.DataArray
    P_raman_xarray = xr.DataArray(
        P_raman2D,
        dims=["time", "range"],
        coords={
            "time": t, 
            "range": z
        },
        name="Raman_signal"
    )

    return P_raman_xarray, params
