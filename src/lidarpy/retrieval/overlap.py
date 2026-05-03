from pathlib import Path
from scipy.integrate import cumulative_trapezoid as cumtrapz

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

from lidarpy.preprocessing.lidar_preprocessing_tools import ff_2D_overlap_from_channels
from lidarpy.general_utils.dates import numpy_to_datetime
from lidarpy.utils.utils import refill_overlap, signal_to_rcs
from lidarpy.preprocessing.lidar_preprocessing import preprocess
from lidarpy.retrieval.klett import klett_rcs
from lidarpy.retrieval.raman import retrieve_backscatter, retrieve_extinction
from lidarpy.atmo.rayleigh import molecular_properties
from lidarpy.atmo.atmo import transmittance
from scipy.optimize import curve_fit

def retrieve_ff_overlap(
    filepath: Path | str,
    hour_range: tuple[float, float],
    output_dir: Path,
    norm_range: tuple[float, float] = (2500, 3500),
    rel_dif_threshold: float = 2.5,
    force_to_one_when_full_overlap: bool = True,
) -> Path | None:
    """Retrieve ff overlap from a near-to-far module lidar ratio

    Args:

        - filepath (Path | str): Lidar file path
        - hour_range (tuple[float, float]): Hour range to calculate the overlap
        - norm_range (tuple[float, float], optional): Range to normalized signals. Defaults to (2500, 3500).
        - rel_dif_threshold (float, optional): Relative difference threshold. Overlap function will be rejected if it causes a relative difference larger than this threshold. Defaults to 2.5 %.
        - output_dir (Path, optional): output folder path. Defaults to Path.cwd().

    Raises:

        - FileNotFoundError: Lidar file not found
        - ValueError: Hour range must be a tuple of two floats
        - ValueError: Not enough profiles in file to cover the selected time range [<50%]
        - ValueError: Could not calculate overlap for 355 nm
        - ValueError: Could not calculate overlap for 532 nm
        - ValueError: Could not calculate overlap for 1064 nm
        - ValueError: Could not calculate overlap for any channel

    Returns:

        - Path | None : Path to the overlap file
    """

    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found")

    if hour_range[0] > hour_range[1]:
        raise ValueError("hour_range[0] must be lower than hour_range[1]")

    # Check time period is within file time range
    raw = xr.open_dataset(filepath)

    # Get date from filename
    date_ = numpy_to_datetime(raw.time[2].values)
    if date_ is None:
        raise ValueError("Could not get date from filename")
    date = dt.datetime(date_.year, date_.month, date_.day)

    time_range = (
        (date + dt.timedelta(hours=hour_range[0])),
        (date + dt.timedelta(hours=hour_range[-1])),
    )
    time_resol = (raw.time[1] - raw.time[0]).item() / 1e9
    expected_profile_number = (
        time_range[1] - time_range[0]
    ).total_seconds() / time_resol
        
    if raw.sel(time=slice(*time_range)).time.size < 0.5 * expected_profile_number:
        raise ValueError(
            "Not enough profiles in file to cover the selected time range [<50%]"
        )

    raw.close()

    # Preprocess lidar file
    lidar_ = preprocess(
        filepath
    )  # FIXME: add crop_ranges and gluing_products and apply_dt as arguments

    # Select time range
    lidar = lidar_.sel(time=slice(*time_range))

    # Retrieve overlap at 355, 532, 1064 nm
    channels = []
    try:        
        overlap355 = ff_2D_overlap_from_channels( lidar, "355fpa", "355npa", norm_range=norm_range, rel_dif_threshold=rel_dif_threshold, force_to_one_when_full_overlap=force_to_one_when_full_overlap)
        channels.append("355fpa")
    except:
        overlap355 = None
        raise ValueError("Could not calculate overlap for 355 nm")
    try:
        overlap532 = ff_2D_overlap_from_channels(
            lidar,
            "532fta",
            "532npa",
            norm_range=norm_range,
            rel_dif_threshold=rel_dif_threshold,
            force_to_one_when_full_overlap=force_to_one_when_full_overlap
        )
        channels.append("532fta")
    except:
        overlap532 = None
        raise ValueError("Could not calculate overlap for 532 nm")
    try:
        overlap1064 = ff_2D_overlap_from_channels(
            lidar,
            "1064fta",
            "1064nta",
            norm_range=norm_range,
            rel_dif_threshold=rel_dif_threshold,
            force_to_one_when_full_overlap=force_to_one_when_full_overlap
        )
        channels.append("1064fta")
    except:
        overlap1064 = None
        raise ValueError("Could not calculate overlap for 1064 nm")

    # Merge overlap data
    overlap_matrix = np.zeros((len(lidar.range), len(channels)))
    if overlap355 is not None and overlap532 is not None and overlap1064 is not None:
        overlap_matrix[:, 0] = overlap355.mean('time').values
        overlap_matrix[:, 1] = overlap532.mean('time').values
        overlap_matrix[:, 2] = overlap1064.mean('time').values
    elif overlap355 is not None and overlap532 is not None:
        overlap_matrix[:, 0] = overlap355.mean('time').values
        overlap_matrix[:, 1] = overlap532.mean('time').values
    elif overlap355 is not None and overlap1064 is not None:
        overlap_matrix[:, 0] = overlap355.mean('time').values
        overlap_matrix[:, 2] = overlap1064.mean('time').values
    elif overlap532 is not None and overlap1064 is not None:
        overlap_matrix[:, 1] = overlap532.mean('time').values
        overlap_matrix[:, 2] = overlap1064.mean('time').values
    elif overlap355 is not None:
        overlap_matrix[:, 0] = overlap355.mean('time').values
    elif overlap532 is not None:
        overlap_matrix[:, 1] = overlap532.mean('time').values
    elif overlap1064 is not None:
        overlap_matrix[:, 2] = overlap1064.mean('time').values
    else:
        overlap_matrix = None
        raise ValueError("Could not calculate overlap for any channel")

    if overlap_matrix is not None:
        # Create DataArray
        overlap = xr.DataArray(
            overlap_matrix,
            dims=("range", "channel"),
            coords={"range": lidar.range.values, "channel": channels},
        )

        # Create output folder
        if not output_dir.exists() and not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        time_min = numpy_to_datetime(lidar.time[0].values)
        if time_min is None:
            raise ValueError("Could not get time from filename")
        time_min = time_min.strftime("%H%M")

        time_max = numpy_to_datetime(lidar.time[-1].values)
        if time_max is None:
            raise ValueError("Could not get time from filename")
        time_max = time_max.strftime("%H%M")

        lidar.close()

        # Create output filename
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist")

        output_path = (
            output_dir
            / f'overlap_alh_ff_{date.strftime("%Y%m%d")}_{time_min}-{time_max}.nc'
        )

        # save overlap files
        overlap.to_netcdf(output_path)
        print(f"Overlap saved in {output_path}")
    else:
        raise ValueError("Could not calculate overlap for any channel")
    
    return output_path


def overlap_function_iterative_raman(
    elastic_signal: xr.DataArray,
    raman_signal: xr.DataArray,
    meteo_profiles: pd.DataFrame,
    particle_lidar_ratio: float,
    wavelengths: tuple[float, float],
    reference: float | tuple[float, float],
    beta_aer_ref: float = 0,
    min_overlap_range: float = 200,  # minimum range of overlap function
    particle_angstrom_exponent: float = 1.0,
    iteration_limit: int = 10,
    debugging: bool = False,
    **kwargs: dict,
) -> xr.Dataset:
    """Retrieve overlap function by means of the Wandinger's method.

    Args:
        elastic_signal (xr.DataArray): Elastic lidar signal at wavelength `wavelengths[0]`.
        raman_signal (xr.DataArray): Raman lidar signal at wavelength `wavelengths[1]`.
        meteo_profiles (pd.DataFrame): from  `lidarpy.atmo.atmo.generate_meteo_profiles` with pressure and temperature data.
        particle_lidar_ratio (float): Particle lidar ratio to be used in the Klett inversion. It shall be chosen so Klett backscatter profile fits Raman backscatter profile once full overlap is reached.
        reference float | (tuple[float, float], optional): Reference range required by the Klett and Raman inversions. Defaults to (8000,8500).
        wavelengths (tuple[float, float], optional): Elastic and Raman wavelengths. Defaults to (532, 531).
        beta_aer_ref (float, optional): Particle backscatter coefficient at `reference range` in m^-1sr^-1. Defaults to 0 m^-1sr^-1.
        particle_angstrom_exponent (float, optional): Particle extinction-related Angstrom exponent. It is used in the Raman inversion. Defaults to 1.
        min_overlap_range (float, optional): Overlap function will be forced to zero below [m]. Defaults to 200 m.
        iteration_limit (int, optional): Maximum number of iterations. Defaults to 10.
        debugging (bool, optional): If True, it will return a dataset with all the intermediate variables. Defaults to False.
        **kwargs: Additional arguments such as `fulloverlap_height` and `output_dir`.

    Returns:
        xr.Dataset: overlap function. If debugging is True, it will return a dataset with all the intermediate variables.
    """
    # Check z in elastic and raman signals (the same for both signals)
    if len(elastic_signal.range) != len(raman_signal.range):
        raise ValueError(
            "elastic and Raman signals must have the same 'range' dimension."
        )

    if len(elastic_signal.range) != len(meteo_profiles["height"]):
        raise ValueError(
            "Signals range dimension and meteo_profiles['heights'] must have the same size."
        )
        
    # if isinstance(reference, (float, np.floating, int)):
    #     reference = (float(reference), float(reference))
    
    z = elastic_signal.range.values.astype(np.float64)

    mol_properties = molecular_properties(
        wavelengths[0],
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=z,
    )
    molecular_backscatter = mol_properties["molecular_beta"].values.astype(np.float64)
    pressure = meteo_profiles["pressure"].values.astype(np.float64)
    temperature = meteo_profiles["temperature"].values.astype(np.float64)

    # Beta Klett
    elastic_backscater = klett_rcs(
        signal_to_rcs(elastic_signal, z).astype(np.float64),
        z,
        molecular_backscatter,
        lr_part=particle_lidar_ratio,
        reference=reference,
    )
    
    extinction = retrieve_extinction(
        raman_signal.values.astype(np.float64),
        z,
        wavelengths,
        pressure,
        temperature,
        reference=reference,
    )

    # Refill extinction
    extinction = refill_overlap(extinction, z, kwargs.get("fulloverlap_height", 600.0))

    raman_backscatter = retrieve_backscatter(
        raman_signal.values.astype(np.float64),
        elastic_signal.values.astype(np.float64),
        extinction,
        z,
        wavelengths,
        pressure,
        temperature,
        reference,
        particle_angstrom_exponent=particle_angstrom_exponent,
    )  # Fine mode has been chosen in Amstrong exponent

    corrected_signal = np.zeros([iteration_limit, len(elastic_signal)])
    delta_overlap = np.zeros(corrected_signal.shape)
    elastic_backscatter_matrix = np.zeros(corrected_signal.shape)
    overlap_function = np.zeros(corrected_signal.shape)
    correction_factor = np.zeros(corrected_signal.shape)

    i = 0
    corrected_signal[i, :] = elastic_signal
    elastic_backscatter_matrix[i, :] = elastic_backscater
    correction_factor[i, :] = (
        2 * raman_backscatter + molecular_backscatter - elastic_backscatter_matrix[i, :]
    ) / (raman_backscatter + molecular_backscatter)

    # Iteraciones restantes
    for i in range(1, iteration_limit):
        corrected_signal[i, :] = (
            corrected_signal[i - 1, :] * correction_factor[i - 1, :]
        )
        rcs_ = signal_to_rcs(corrected_signal[i, :], z)
        elastic_backscatter_matrix[i, :] = klett_rcs(
            rcs_,
            z,
            molecular_backscatter,
            lr_part=particle_lidar_ratio,
            beta_aer_ref=beta_aer_ref,
            reference=reference,
        )

        correction_factor[i, :] = (
            2 * raman_backscatter
            + molecular_backscatter
            - elastic_backscatter_matrix[i, :]
        ) / (raman_backscatter + molecular_backscatter)

        overlap_function[i, :] = corrected_signal[0, :] / corrected_signal[i, :]

    final_overlap_function = overlap_function[-1, :]

    # Set to zero the values of overlap function below 200 m
    #final_overlap_function[: np.where(z > min_overlap_range)[0][0]] = 0

    # Search first 1 in overlap function
    first_1 = np.where(final_overlap_function >= 1)[0][0]

    # Set constant profile to 1 form first 1 up
    #final_overlap_function[first_1:] = 1

    final_overlap_function = xr.DataArray(
        final_overlap_function, dims=("range"), coords={"range": z}
    )

    if debugging:
        # Generate a xr.Dataset to save the results
        overlap_function = xr.DataArray(
            overlap_function,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        elastic_backscatter_matrix = xr.DataArray(
            elastic_backscatter_matrix,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        raman_backscatter = xr.DataArray(
            raman_backscatter, dims=("range"), coords={"range": z}
        )
        extinction = xr.DataArray(extinction, dims=("range"), coords={"range": z})
        correction_factor = xr.DataArray(
            correction_factor,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        delta_overlap = xr.DataArray(
            delta_overlap,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        corrected_signal = xr.DataArray(
            corrected_signal,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        dataset = xr.Dataset(
            {
                "overlap_function_matrix": overlap_function,
                "elastic_backscatter_matrix": elastic_backscatter_matrix,
                "raman_backscatter": raman_backscatter,
                "correction_factor": correction_factor,
                "delta_overlap": delta_overlap,
                "corrected_signal": corrected_signal,
                "extinction": extinction,
                "overlap_function": final_overlap_function,
            }
        )

    else:
        dataset = xr.Dataset(
            {
                "overlap_function": final_overlap_function,
            }
        )

    return dataset


def overlap_function_iterative_klett(
    ff_elastic_signal: xr.DataArray,
    nf_elastic_signal: xr.DataArray,
    meteo_profiles: pd.DataFrame,
    particle_lidar_ratio: float,
    wavelength: float,
    reference: float | tuple[float, float],
    beta_aer_ref: float = 0,
    min_overlap_range: float = 200,  # minimum range of overlap function
    iteration_limit: int = 10,
    debugging: bool = False,
    **kwargs: dict,
) -> xr.Dataset:
    """Retrieve overlap function by means of the iterative method with near-field Klett as reference.

    Args:
        ff_elastic_signal (xr.DataArray): Far-field elastic lidar signal.
        nf_elastic_signal (xr.DataArray): Near-field elastic lidar signal.
        meteo_profiles (pd.DataFrame): from  `lidarpy.atmo.atmo.generate_meteo_profiles` with pressure and temperature data.
        particle_lidar_ratio (float): Particle lidar ratio to be used in the Klett inversion. It shall be chosen so Klett backscatter profile fits Raman backscatter
        wavelength (float): Elastic wavelength.
        reference float | (tuple[float, float]): Reference range required by the Klett inversion.
        beta_aer_ref (float, optional): Particle backscatter coefficient at `reference range` in m^-1sr^-1. Defaults to 0 m^-1sr^-1.
        min_overlap_range (float, optional): Overlap function will be forced to zero below [m]. Defaults to 200 m.
        debugging (bool, optional): If True, it will return a dataset with all the intermediate variables. Defaults to False.

    Raises:
        ValueError: Both nf- and ff-elastic signals must have the same 'range' dimension.
        ValueError: Signals range dimension and meteo_profiles['heights'] must have the same size.

    Returns:
        xr.Dataset: overlap function. If debugging is True, it will return a dataset with all the intermediate variables.
    """    

    # Check z in elastic and raman signals (the same for both signals)
    if len(ff_elastic_signal.range) != len(nf_elastic_signal.range):
        raise ValueError(
            "Both nf- and ff-elastic signals must have the same 'range' dimension."
        )

    if len(ff_elastic_signal.range) != len(meteo_profiles["height"]):
        raise ValueError(
            "Signals range dimension and meteo_profiles['heights'] must have the same size."
        )

    z = ff_elastic_signal.range.values.astype(np.float64)

    mol_properties = molecular_properties(
        wavelength,
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=z,
    )
    molecular_backscatter = mol_properties["molecular_beta"].values.astype(np.float64)

    # Beta Klett
    ff_elastic_backscater = klett_rcs(
        signal_to_rcs(ff_elastic_signal, z).astype(np.float64),
        z,
        molecular_backscatter,
        lr_part=particle_lidar_ratio,
        reference=reference,
    )

    nf_elastic_backscater = klett_rcs(
        signal_to_rcs(nf_elastic_signal, z).astype(np.float64),
        z,
        molecular_backscatter,
        lr_part=particle_lidar_ratio,
        reference=reference,
    )

    corrected_signal = np.zeros([iteration_limit, len(ff_elastic_signal)])
    delta_overlap = np.zeros(corrected_signal.shape)
    ff_elastic_backscatter_matrix = np.zeros(corrected_signal.shape)
    overlap_function = np.zeros(corrected_signal.shape)
    correction_factor = np.zeros(corrected_signal.shape)

    i = 0
    corrected_signal[i, :] = ff_elastic_signal
    ff_elastic_backscatter_matrix[i, :] = ff_elastic_backscater
    correction_factor[i, :] = (
        2 * nf_elastic_backscater + molecular_backscatter - ff_elastic_backscatter_matrix[i, :]
    ) / (nf_elastic_backscater + molecular_backscatter)

    # Iteraciones restantes
    for i in range(1, iteration_limit):
        corrected_signal[i, :] = (
            corrected_signal[i - 1, :] * correction_factor[i - 1, :]
        )
        rcs_ = signal_to_rcs(corrected_signal[i, :], z)
        ff_elastic_backscatter_matrix[i, :] = klett_rcs(
            rcs_,
            z,
            molecular_backscatter,
            lr_part=particle_lidar_ratio,
            beta_aer_ref=beta_aer_ref,
            reference=reference,
        )

        correction_factor[i, :] = (
            2 * nf_elastic_backscater
            + molecular_backscatter
            - ff_elastic_backscatter_matrix[i, :]
        ) / (nf_elastic_backscater + molecular_backscatter)

        overlap_function[i, :] = corrected_signal[0, :] / corrected_signal[i, :]

    final_overlap_function = overlap_function[-1, :]

    # Set to zero the values of overlap function below 200 m
    final_overlap_function[: np.where(z > min_overlap_range)[0][0]] = 0

    # Search first 1 in overlap function
    first_1 = np.where(final_overlap_function >= 1)[0][0]

    # Set constant profile to 1 form first 1 up
    final_overlap_function[first_1:] = 1

    final_overlap_function = xr.DataArray(
        final_overlap_function, dims=("range"), coords={"range": z}
    )

    if debugging:
        # Generate a xr.Dataset to save the results
        overlap_function = xr.DataArray(
            overlap_function,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        ff_elastic_backscatter_matrix = xr.DataArray(
            ff_elastic_backscatter_matrix,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        nf_elastic_backscatter = xr.DataArray(
            nf_elastic_backscater, dims=("range"), coords={"range": z}
        )
        correction_factor = xr.DataArray(
            correction_factor,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        delta_overlap = xr.DataArray(
            delta_overlap,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        corrected_signal = xr.DataArray(
            corrected_signal,
            dims=("iteration", "range"),
            coords={"iteration": np.arange(iteration_limit), "range": z},
        )
        dataset = xr.Dataset(
            {
                "overlap_function_matrix": overlap_function,
                "elastic_backscatter_matrix": ff_elastic_backscatter_matrix,
                "nf_elastic_backscatter": nf_elastic_backscatter,
                "correction_factor": correction_factor,
                "delta_overlap": delta_overlap,
                "corrected_signal": corrected_signal,
                "overlap_function": final_overlap_function,
            }
        )

    else:
        dataset = xr.Dataset(
            {
                "overlap_function": final_overlap_function,
            }
        )

    return dataset


def overlap_function_explicit(
    elastic_rcs: xr.DataArray,
    raman_rcs: xr.DataArray,
    wavelengths: tuple[float, float],
    reference_heights: float | tuple[float, float],
    particle_lidar_ratio: float,
    meteo_profiles: pd.DataFrame,
    debugging=False,
    **kwargs: dict,
) -> xr.Dataset:
    """Retrieve overlap function by means of the explicit method.

    Args:

        - elastic_rcs (xr.DataArray): Elastic range corrected signal at wavelength `wavelengths[0]`.
        - raman_rcs (xr.DataArray): Raman range corrected signal at wavelength `wavelengths[1]`.
        - wavelengths (tuple[float, float]): Elastic and Raman wavelengths.
        - reference_height (float | tuple[float, float]): Aerosol-free region reference height in meters.
        - particle_lidar_ratio (float): Assumed particle lidar ratio.
        - meteo_profiles (pd.DataFrame): Meteo profiles with pressure and temperature data from `lidarpy.atmo.atmo.generate_meteo_profiles`.

    Returns:

        - xr.Dataset: overlap function. If debugging is True, it will return a dataset with all the intermediate variables.

    References:

        - Comerón et al., 2023: https://doi.org/10.5194/amt-16-3015-2023
    """

        # --- Convertir reference float a tupla ---
    if isinstance(reference_heights, float):
        reference_heights = (reference_heights, reference_heights)
        
    r = elastic_rcs.range.values.astype(np.float64)
    dr = r[1] - r[0]

    # Create an array full of 1 of size r
    array_ones = np.ones(r.shape)
    elastic_mol_properties = molecular_properties(
        wavelengths[0],
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=r,
    )
    raman_mol_properties = molecular_properties(
        wavelengths[1],
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=r,
    )

    # Computation of beta molecular for elastic and Raman signal
    beta_mol_elastic = elastic_mol_properties["molecular_beta"]  # it has to be in m-1*sr-1
    beta_mol_raman = raman_mol_properties["molecular_beta"]

    # Cálculo inverso de beta molecular, a partir de la señal Rayleigh
    XE = elastic_rcs.values  # elastic signal
    X_Ray = XE  # molecular backscatter normalized to RCS

    XR = raman_rcs.values  # raman signal
    XR_Ray = XR  # molecular backscatter normalized to RCS
    
    Sm_el_array = elastic_mol_properties["molecular_lidar_ratio"] 
    Sa_array = np.full_like(Sm_el_array, particle_lidar_ratio)# Assumed Lidar Ratio
    Sm_ra_array = raman_mol_properties['molecular_lidar_ratio']

    ref_height = np.mean(np.array(reference_heights))  # Reference height (m)
    idx = np.round(ref_height / dr).astype(int)
    idx_min = np.round(reference_heights[0] / dr).astype(int)
    idx_max = np.round(reference_heights[1] / dr).astype(int)
    
    if idx_min != idx_max:
        X_ref = np.nanmean(X_Ray[idx_min:idx_max])
        XR_ref = np.nanmean(XR_Ray[idx_min:idx_max])
        beta_mol_ref = np.nanmean(beta_mol_elastic[idx_min:idx_max])
    
    elif idx_min == idx_max:
        X_ref = X_Ray[idx]
        XR_ref = XR_Ray[idx]
        beta_mol_ref = beta_mol_elastic.values[idx]

    norm_XR = XR / XR_ref
    nbeta_molecular = beta_mol_elastic.values / beta_mol_ref

    beta_mol_el = beta_mol_elastic[:idx]
    Sa = Sa_array[:idx]
    Sm_el = Sm_el_array[:idx]
    Sm_ra = Sm_ra_array[:idx]
    beta_mol_ra = beta_mol_raman[:idx]
    nbeta_mol = nbeta_molecular[:idx]
    nXR = norm_XR[:idx]
    X = XE[:idx]
    
    fexp = np.exp(
        -2
        * (Sa - Sm_el.values)
        * np.flip(cumtrapz(np.flip(beta_mol_el), initial=0, dx=dr))
    )

    # Not used because its efect is negligible
    fexp_dif_ext_mol = np.exp(- np.flip(cumtrapz(np.flip(Sm_el.values * beta_mol_el.values - Sm_ra.values * beta_mol_ra.values), initial=0, dx=dr)))

    g = (nbeta_mol / nXR) * fexp * fexp_dif_ext_mol

    phi = (2 * beta_mol_el / (nXR * X_ref)) * fexp * fexp_dif_ext_mol

    psi = (X * Sa) / fexp

    phi_psi = phi * psi

    fexp_phi_psi = np.exp(np.flip(cumtrapz(np.flip(phi_psi), initial=0, dx=dr)))

    fint_g_psi = np.flip(cumtrapz(np.flip(g * psi / fexp_phi_psi), initial=0, dx=dr))

    f = g + phi * fexp_phi_psi * fint_g_psi

    Oe = 1 / f

    # Paste Oe into the array of ones so that it has the same length as r
    Oe = np.concatenate((Oe, array_ones[idx:]))

    # Create a dataset with all the intermediate parameters
    Oe = xr.DataArray(Oe, dims=("range"), coords={"range": r})

    if debugging:
        f_ = np.empty_like(r)
        f_[:idx] = f
        g_ = np.empty_like(r)
        g_[:idx] = g
        phi_ = np.empty_like(r)
        phi_[:idx] = phi
        psi_ = np.empty_like(r)
        psi_[:idx] = psi
        phi_psi_ = np.empty_like(r)
        fexp_ = np.empty_like(r)
        fexp_[:idx] = fexp
        phi_psi_[:idx] = phi_psi
        fexp_phi_psi_ = np.empty_like(r)
        fexp_phi_psi_[:idx] = fexp_phi_psi
        fint_g_psi_ = np.empty_like(r)
        fint_g_psi_[:idx] = fint_g_psi

        f = xr.DataArray(f_, dims=("range"), coords={"range": r})
        g = xr.DataArray(g_, dims=("range"), coords={"range": r})
        phi = xr.DataArray(phi_, dims=("range"), coords={"range": r})
        psi = xr.DataArray(psi_, dims=("range"), coords={"range": r})
        fexp = xr.DataArray(fexp_, dims=("range"), coords={"range": r})
        phi_psi = xr.DataArray(phi_psi_, dims=("range"), coords={"range": r})
        fexp_phi_psi = xr.DataArray(fexp_phi_psi_, dims=("range"), coords={"range": r})
        fint_g_psi = xr.DataArray(fint_g_psi_, dims=("range"), coords={"range": r})
        beta_mol_el = xr.DataArray(beta_mol_elastic, dims=("range"), coords={"range": r})
        beta_mol_ra = xr.DataArray(beta_mol_raman, dims=("range"), coords={"range": r})
        nbeta_mol = xr.DataArray(nbeta_molecular, dims=("range"), coords={"range": r})
        nXR = xr.DataArray(norm_XR, dims=("range"), coords={"range": r})
        X = xr.DataArray(XE, dims=("range"), coords={"range": r})
        XR = xr.DataArray(XR, dims=("range"), coords={"range": r})

        dataset = xr.Dataset(
            {
                "overlap_function": Oe,
                "f": f,
                "g": g,
                "phi": phi,
                "psi": psi,
                "phi_psi": phi_psi,
                "fexp_phi_psi": fexp_phi_psi,
                "fint_g_psi": fint_g_psi,
                "beta_mol_el": beta_mol_el,
                "beta_mol_ra": beta_mol_ra,
                "nbeta_mol": nbeta_mol,
                "nXR": nXR,
                "X": X,
                "fexp": fexp,
                "XR": XR,
                "XR_Ray": XR_Ray,
                "X_Ray": X_Ray,
            }
        )
    else:
        dataset = xr.Dataset({"overlap_function": Oe})

    return dataset


def overlap_function_explicit_using_particle_extinction(
    elastic_rcs: xr.DataArray,
    raman_rcs: xr.DataArray,
    wavelengths: tuple[float, float],
    reference_heights: float | tuple[float, float],
    particle_lidar_ratio: float,
    meteo_profiles: pd.DataFrame,
    particle_angstrom_exponent: float = 1.0,
    debugging=False,
    **kwargs: dict,
) -> xr.Dataset:
    """Retrieve overlap function by means of the explicit method including the raman inversion.

    Args:

        - elastic_rcs (xr.DataArray): Elastic range corrected signal at wavelength `wavelengths[0]`.
        - raman_rcs (xr.DataArray): Raman range corrected signal at wavelength `wavelengths[1]`.
        - wavelengths (tuple[float, float]): Elastic and Raman wavelengths.
        - reference_height (float | tuple[float, float]): Aerosol-free region reference height in meters.
        - particle_lidar_ratio (float): Assumed particle lidar ratio.
        - meteo_profiles (pd.DataFrame): Meteo profiles with pressure and temperature data from `lidarpy.atmo.atmo.generate_meteo_profiles`.
        - particle_angstrom_exponent (float, optional): Particle extinction-related Angstrom exponent. It is used in the Raman inversion. Defaults to 1. 
        This version includes the effect of the differential aerosol transmission term. For these reason, it is necessary to provide the particle_angstrom_exponent.

    Returns:

        - xr.Dataset: overlap function. If debugging is True, it will return a dataset with all the intermediate variables.

    References:

        - Comerón et al., 2023: https://doi.org/10.5194/amt-16-3015-2023
    """

        # --- Convertir reference float a tupla ---
    if isinstance(reference_heights, float):
        reference_heights = (reference_heights, reference_heights)
        
    r = elastic_rcs.range.values.astype(np.float64)
    dr = r[1] - r[0]

    # Create an array full of 1 of size r
    array_ones = np.ones(r.shape)
    elastic_mol_properties = molecular_properties(
        wavelengths[0],
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=r,
    )
    raman_mol_properties = molecular_properties(
        wavelengths[1],
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=r,
    )

    # Computation of beta molecular for elastic and Raman signal
    beta_mol_elastic = elastic_mol_properties["molecular_beta"]  # it has to be in m-1*sr-1
    beta_mol_raman = raman_mol_properties["molecular_beta"]

    # Cálculo inverso de beta molecular, a partir de la señal Rayleigh
    XE = elastic_rcs.values  # elastic signal
    X_Ray = XE  # molecular backscatter normalized to RCS

    XR = raman_rcs.values  # raman signal
    XR_Ray = XR  # molecular backscatter normalized to RCS
    
    
    # Computation of elastic extinction using the raman algorithm
    pressure = meteo_profiles["pressure"].values.astype(np.float64)
    temperature = meteo_profiles["temperature"].values.astype(np.float64)
    raman_signal = raman_rcs / (r**2)
    
    extinction = retrieve_extinction(
        raman_signal.values.astype(np.float64),
        r,
        wavelengths,
        pressure,
        temperature,
        reference=reference_heights,
    )
    
    Sm_el_array = elastic_mol_properties["molecular_lidar_ratio"] 
    Sa_array = np.full_like(Sm_el_array, particle_lidar_ratio)# Assumed Lidar Ratio
    Sm_ra_array = raman_mol_properties['molecular_lidar_ratio']


    ref_height = np.mean(np.array(reference_heights))  # Reference height (m)
    idx = np.round(ref_height / dr).astype(int)
    idx_min = np.round(reference_heights[0] / dr).astype(int)
    idx_max = np.round(reference_heights[1] / dr).astype(int)
    
    if idx_min != idx_max:
        X_ref = np.nanmean(X_Ray[idx_min:idx_max])
        XR_ref = np.nanmean(XR_Ray[idx_min:idx_max])
        beta_mol_ref = np.nanmean(beta_mol_elastic[idx_min:idx_max])
    elif idx_min == idx_max:
        X_ref = X_Ray[idx]
        XR_ref = XR_Ray[idx]
        beta_mol_ref = beta_mol_elastic.values[idx]

    norm_XR = XR / XR_ref
    nbeta_molecular = beta_mol_elastic.values / beta_mol_ref

    beta_mol_el = beta_mol_elastic[:idx]
    Sa = Sa_array[:idx]
    Sm_el = Sm_el_array[:idx]
    Sm_ra = Sm_ra_array[:idx]
    beta_mol_ra = beta_mol_raman[:idx]
    nbeta_mol = nbeta_molecular[:idx]
    nXR = norm_XR[:idx]
    X = XE[:idx]
    extinction_ref = extinction[:idx]
    
    fexp = np.exp(
        -2
        * (Sa - Sm_el.values)
        * np.flip(cumtrapz(np.flip(beta_mol_el), initial=0, dx=dr))
    )

    # Not used because its efect is negligible
    fexp_dif_ext_mol = np.exp(- np.flip(cumtrapz(np.flip(Sm_el.values * beta_mol_el.values - Sm_ra.values * beta_mol_ra.values), initial=0, dx=dr)))
    
    # Differential aerosol transmission term
    fexp_dif_ext_aer = np.exp(- (1 - (wavelengths[0]/wavelengths[1])**(particle_angstrom_exponent)) * np.flip(cumtrapz(np.flip(extinction_ref),
                                                                                                                                  initial=0, dx=dr)))

    g = (nbeta_mol / nXR) * fexp * fexp_dif_ext_mol * fexp_dif_ext_aer

    phi = (2 * beta_mol_el / (nXR * X_ref)) * fexp * fexp_dif_ext_mol * fexp_dif_ext_aer

    psi = (X * Sa) / fexp

    phi_psi = phi * psi

    fexp_phi_psi = np.exp(np.flip(cumtrapz(np.flip(phi_psi), initial=0, dx=dr)))

    fint_g_psi = np.flip(cumtrapz(np.flip(g * psi / fexp_phi_psi), initial=0, dx=dr))

    f = g + phi * fexp_phi_psi * fint_g_psi

    Oe = 1 / f

    # Paste Oe into the array of ones so that it has the same length as r
    Oe = np.concatenate((Oe, array_ones[idx:]))

    # Create a dataset with all the intermediate parameters
    Oe = xr.DataArray(Oe, dims=("range"), coords={"range": r})

    if debugging:
        f_ = np.empty_like(r)
        f_[:idx] = f
        g_ = np.empty_like(r)
        g_[:idx] = g
        phi_ = np.empty_like(r)
        phi_[:idx] = phi
        psi_ = np.empty_like(r)
        psi_[:idx] = psi
        phi_psi_ = np.empty_like(r)
        fexp_ = np.empty_like(r)
        fexp_[:idx] = fexp
        phi_psi_[:idx] = phi_psi
        fexp_phi_psi_ = np.empty_like(r)
        fexp_phi_psi_[:idx] = fexp_phi_psi
        fint_g_psi_ = np.empty_like(r)
        fint_g_psi_[:idx] = fint_g_psi

        f = xr.DataArray(f_, dims=("range"), coords={"range": r})
        g = xr.DataArray(g_, dims=("range"), coords={"range": r})
        phi = xr.DataArray(phi_, dims=("range"), coords={"range": r})
        psi = xr.DataArray(psi_, dims=("range"), coords={"range": r})
        fexp = xr.DataArray(fexp_, dims=("range"), coords={"range": r})
        phi_psi = xr.DataArray(phi_psi_, dims=("range"), coords={"range": r})
        fexp_phi_psi = xr.DataArray(fexp_phi_psi_, dims=("range"), coords={"range": r})
        fint_g_psi = xr.DataArray(fint_g_psi_, dims=("range"), coords={"range": r})
        beta_mol_el = xr.DataArray(beta_mol_elastic, dims=("range"), coords={"range": r})
        beta_mol_ra = xr.DataArray(beta_mol_raman, dims=("range"), coords={"range": r})
        nbeta_mol = xr.DataArray(nbeta_molecular, dims=("range"), coords={"range": r})
        nXR = xr.DataArray(norm_XR, dims=("range"), coords={"range": r})
        X = xr.DataArray(XE, dims=("range"), coords={"range": r})
        XR = xr.DataArray(XR, dims=("range"), coords={"range": r})

        dataset = xr.Dataset(
            {
                "overlap_function": Oe,
                "f": f,
                "g": g,
                "phi": phi,
                "psi": psi,
                "phi_psi": phi_psi,
                "fexp_phi_psi": fexp_phi_psi,
                "fint_g_psi": fint_g_psi,
                "beta_mol_el": beta_mol_el,
                "beta_mol_ra": beta_mol_ra,
                "nbeta_mol": nbeta_mol,
                "nXR": nXR,
                "X": X,
                "fexp": fexp,
                "XR": XR,
                "XR_Ray": XR_Ray,
                "X_Ray": X_Ray,
            }
        )
    else:
        dataset = xr.Dataset({"overlap_function": Oe})

    return dataset


def overlap_function_homogeneous_backscatter(
    elastic_rcs: xr.DataArray,
    meteo_profiles: pd.DataFrame,
    particle_lidar_ratio: float,
    wavelength: float,
    reference: float | tuple[float, float],
    fulloverlap_height: float,
    absolute_lidar_calibration_factor: float,
    beta_aer_ref: float = 0,
    ) -> xr.DataArray:
    """Retrieve overlap function by means of the homogeneous backscatter method. It is not necessary to provide a Raman signal.
    Args:
    
        elastic_rcs (xr.DataArray): Elastic range corrected signal.
        meteo_profiles (pd.DataFrame): from  `lidarpy.atmo.atmo.generate_meteo_profiles` with pressure and temperature data.
        particle_lidar_ratio (float): Particle lidar ratio to be used in the Klett inversion.
        wavelength (float): Elastic wavelength.
        reference float | (tuple[float, float]): Reference range required by the Klett inversion.
        fulloverlap_height (float): Height above which the overlap function is considered to be 1.
        absolute_lidar_calibration_factor (float): Absolute lidar calibration factor.
        beta_aer_ref (float, optional): Particle backscatter coefficient at `reference range` in m^-1sr^-1. Defaults to 0 m^-1sr^-1.
    Returns:
        xr.DataArray: overlap function.
        """
    
    z = elastic_rcs.range.values.astype(np.float64)
    
    mol_properties = molecular_properties(
        wavelength,
        meteo_profiles["pressure"],
        meteo_profiles["temperature"],
        heights=z,
    )
    molecular_backscatter = mol_properties["molecular_beta"].values.astype(np.float64)
    molecular_extinction = mol_properties["molecular_alpha"].values.astype(np.float64)

    # Beta Klett
    particle_backscater = klett_rcs(
        elastic_rcs,
        z,
        molecular_backscatter,
        lr_part=particle_lidar_ratio,
        reference=reference,
    )
    
    particle_backscater_refilled = refill_overlap(particle_backscater, z, fulloverlap_height)

    particle_extinction_refilled = particle_backscater_refilled * particle_lidar_ratio

    transmittance_elastic_refilled = transmittance(molecular_extinction + particle_extinction_refilled, z)


    overlap = elastic_rcs/(absolute_lidar_calibration_factor * (molecular_backscatter + particle_backscater_refilled) * transmittance_elastic_refilled**2)

    return overlap


def overlap_function_correction_for_temperature_dependent_variations(range_vals, beta_raw, fit_range) -> tuple[np.ndarray, float, float]:
    """
    Computes the overlap correction function f_c(r) to correct distortions in lidar backscatter data.

    Parameters:
    - range_vals: 1D array of range values (r) in meters.
    - beta_raw: 1D array of raw backscatter values beta_raw(r).
    - fit_range: Tuple (R_OK_min, R_OK_max) defining the valid fitting range.

    Returns:
    - fc: Interpolated correction function to match beta_raw's length.
    - A, B: Coefficients from the linear fit.

    References:
    - Hervo et al., 2016: https://doi.org/10.5194/amt-9-2947-2016
    """
    # Convert inputs to NumPy arrays if they are xarray DataArray
    range_vals = np.asarray(range_vals)
    beta_raw = np.asarray(beta_raw)

    # Ensure all values in beta_raw are positive (avoid log issues)
    valid_mask = beta_raw > 0
    range_vals_valid = range_vals[valid_mask]
    beta_raw_valid = beta_raw[valid_mask]

    # Select data within the fitting range
    mask_fit = (range_vals_valid >= fit_range[0]) & (range_vals_valid <= fit_range[1])
    if np.sum(mask_fit) < 2:
        raise ValueError("Not enough valid points in the fitting range.")

    # Extract fitting data
    r_fit = range_vals_valid[mask_fit]
    log_beta_raw_fit = np.log(beta_raw_valid[mask_fit])

    # Perform linear fit: log(beta_raw) = A + B * r
    linear_function = lambda r, A, B: A + B * r
    popt, _ = curve_fit(linear_function, r_fit, log_beta_raw_fit, maxfev=10000)
    A, B = popt

    # Compute log(f_c) for all valid range values
    log_fc = np.full_like(range_vals, np.nan)  # Initialize with NaN
    log_fc[valid_mask] = np.log(beta_raw_valid) - (A + B * range_vals_valid)

    # Interpolate missing values in f_c(r) for full coverage
    nan_mask = np.isnan(log_fc)
    log_fc[nan_mask] = np.interp(
        range_vals[nan_mask],
        range_vals_valid,
        log_fc[valid_mask],
        left=log_fc[valid_mask][0],
        right=log_fc[valid_mask][-1],
    )

    # Convert back from log scale
    fc = np.exp(-log_fc)

    return fc, A, B
