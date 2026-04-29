from pathlib import Path
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
import xarray as xr

from lidarpy.atmo.atmo import transmittance
from lidarpy.utils.utils import refill_overlap
import matplotlib.dates as mdates


def get_calibration_factor_profile(    
    rcs_elastic: np.ndarray,
    particle_backscatter: np.ndarray,
    range_profile: np.ndarray,
    mol_properties: pd.DataFrame,
    lr_part: float,
    full_overlap_height: float,
    ) -> np.ndarray:
    """Compute the calibration factor profile for the elastic channel.
    Args:
        rcs_elastic (np.ndarray): Profile of the elastic range-corrected signal (time, range).
        particle_backscatter (np.ndarray): Profile of the particle backscatter coefficient (time, range).
        mol_properties (pd.DataFrame): Molecular properties with columns ['molecular_beta', 'molecular_alpha'].
        lr_part (float): Lidar ratio of particles (sr).
        full_overlap_height (float): Height above which the overlap is 1.
    Returns:
        np.ndarray: Calibration factor profile (range).
    """

    # Check that beta_part and rcs_elastic have the same dimensions
    if rcs_elastic.shape != particle_backscatter.shape:
        raise ValueError("rcs_elastic and particle_backscatter must have the same shape.")
    
    #fill overlap
    beta_part_refill = refill_overlap(particle_backscatter, range_profile, full_overlap_height)
    alpha_part_refill = beta_part_refill * lr_part
    
    transmittance_elastic_refill = transmittance(mol_properties["molecular_alpha"] + alpha_part_refill, range_profile)
    
    calibration_factor_profile = rcs_elastic / ((mol_properties["molecular_beta"] + beta_part_refill) * transmittance_elastic_refill**2)
         
    
    return calibration_factor_profile # type: ignore

def get_calibration_factor_direct(
    rcs_elastic: xr.DataArray,
    particle_backscatter: xr.DataArray,
    mol_properties: pd.DataFrame,
    lr_part: float,
    full_overlap_height: float,
    debug: bool = False,
    **kwargs
) -> xr.Dataset:
    """
    Compute the lidar calibration factor for the elastic channel (direct method) over a selected time period (e.g., full day, night, hour).

    Args:
        rcs_elastic (xr.DataArray): Quicklook of the elastic range-corrected signal (time, range).
        particle_backscatter (xr.DataArray): Quicklook of the particle backscatter coefficient (time, range).
        mol_properties (pd.DataFrame): Molecular properties with columns ['molecular_beta', 'molecular_alpha'].
        lr_part (float): Lidar ratio of particles (sr).
        full_overlap_height (float): Height above which the overlap is 1.
        debug (bool): If True, plot K over time for selected heights.
        kwargs: calibration_heights (list) = List of height(s) (in meters) to compute the calibration factor if debug is true. Defaults to None.

    Returns:
        xr.DataArray | dict:
            - If debug=False → Calibration factor (xr.DataArray). K is independent of height (full 2D field).
            - If debug=True → dict with:
                {
                    "K_sel": xr.DataArray (selected height time series),
                    "K_mean": float (mean value of the time series),
                    "K_std": float (standard deviation of the time series),
                    "K_all": xr.DataArray (full 2D field)
                }
    """
    # Check that beta_part and rcs_elastic have the same dimensions
    if rcs_elastic.shape != particle_backscatter.shape:
        raise ValueError("rcs_elastic and particle_backscatter must have the same shape.")

    # --- Coordinates ---
    range_profile = rcs_elastic.range.values
    time_profile = rcs_elastic.time.values

    # channel_name = rcs_elastic.channel.attrs.get("long_name", rcs_elastic.channel.values)
    # date_label = pd.to_datetime(str(rcs_elastic.time.values[0])).strftime("%Y-%m-%d")

    Nt, Nz = len(time_profile), len(range_profile)

    # --- Expand molecular properties to 2D ---
    molecular_beta2D = np.tile(mol_properties["molecular_beta"].values[np.newaxis, :], (Nt, 1))
    molecular_alpha2D = np.tile(mol_properties["molecular_alpha"].values[np.newaxis, :], (Nt, 1))

    # --- Initialize matrices ---
    beta_part = np.zeros((Nt, Nz))
    beta_part_refill = np.zeros((Nt, Nz))
    alpha_part_refill = np.zeros((Nt, Nz))
    transmittance_elastic = np.zeros((Nt, Nz))

    # --- Refill backscatter in overlap region and compute alpha ---
    for i in range(Nt):
        beta_part_refill[i, :] = refill_overlap(
            particle_backscatter.isel(time=i).values,
            range_profile,
            fulloverlap_height=full_overlap_height,
        )
        alpha_part_refill[i, :] = beta_part_refill[i, :] * lr_part

    for i in range(Nt):
        total_alpha = molecular_alpha2D[i, :] + alpha_part_refill[i, :]
        transmittance_elastic[i, :] = transmittance(total_alpha, range_profile)

    calibration_factor = rcs_elastic.values / (
        (molecular_beta2D + beta_part_refill) * transmittance_elastic**2
    )

    calibration_factor_da = xr.DataArray(
        calibration_factor,
        coords={"time": time_profile, "range": range_profile},
        dims=["time", "range"],
        name="calibration_factor",
        attrs={
            "description": "Calibration factor",
            "units": "a.u.",
        },
    )
    calibration_dataset = calibration_factor_da.to_dataset(name="calibration_factor")
    
    # --- If debug, plot K over time for selected heights ---
    if debug:
        plt.figure(figsize=(15, 6))
        calibration_heights = kwargs.get("calibration_heights", [2000, 3000, 4000])  # in meters
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(calibration_heights)))
        fontsize = 18

        for idx, range_ in enumerate(calibration_heights):
            k_sel = calibration_factor_da.sel(range=range_, method="nearest")
            k_sel.plot(label=f"K @ {range_} m", lw=2, color=colors[idx])
            plt.scatter(k_sel.time, k_sel.values, color="black", s=40)

        # Calcular estadísticas en el último nivel
        k_last = calibration_factor_da.sel(range=calibration_heights[-1], method="nearest")
        k_mean = float(k_last.mean(dim="time").values)
        k_std = float(k_last.std(dim="time").values)

        exp = int(np.floor(np.log10(abs(k_mean)))) if k_mean != 0 else 0
        mean_norm = k_mean / 10**exp
        std_norm = k_std / 10**exp

        plt.axhline(k_mean, color="gray", lw=2, ls="--", label="Mean")

        plt.text(
            0.01, 0.2,
            r'Mean$\pm$STD:' f'\n({mean_norm:.2f}' + r'$\pm$' + f'{std_norm:.2f})e{exp} a.u.',
            transform=plt.gca().transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"),
        )

        plt.title(f"Calibration factor K | Direct method", fontsize=fontsize)
        plt.xlabel("Time [UTC]", fontsize=fontsize)
        plt.ylabel("K [a.u.]", fontsize=fontsize)
        plt.legend(fontsize=fontsize - 2)
        plt.grid(True)
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
        plt.ylim(k_mean-0.01*k_mean, k_mean+0.01*k_mean)
        plt.tight_layout()        
           
        # --- Return dictionary ---
        calibration_dataset = xr.Dataset({
            "K_time_series": k_last,          # Time series of the selected height
            "K_mean": k_mean,         # Mean
            "K_std": k_std,           # Standard deviation
            "K_quicklook": calibration_factor_da,  # Full field
        })

    # --- Return calibration dataset ---
    return calibration_dataset


def get_calibration_factor_attenuated_backscatter_fit(
    rcs_elastic: xr.DataArray,
    particle_backscatter: xr.DataArray,
    mol_properties: pd.DataFrame,
    lr_part: float,
    full_overlap_height: float = 1000.,
    reference_height: float = 7000.,    
    debug: bool = False,
    **kwargs
) -> xr.Dataset:
    """
    Compute the lidar calibration factor for the elastic channel using the attenuated backscatter fitting method over a selected height range.
    Args:
        rcs_elastic (xr.DataArray): Quicklook of the elastic range-corrected signal (time, range).
        particle_backscatter (xr.DataArray): Quicklook of the particle backscatter coefficient (time, range).
        mol_properties (pd.DataFrame): Molecular properties with columns ['molecular_beta', 'molecular_alpha'].
        lr_part (float): Lidar ratio of particles (sr).
        full_overlap_height (float): Height above which the overlap is 1. Default is 1000 m.
        reference_height (float): Upper height limit for calibration. Default is 7000 m.        
        debug (bool): If True, plot K and b over time. Default is False.
        kwargs: plot_specific_time (str) = Specific time to plot the fit (format: 'YYYY-MM-DD HH:MM:SS'). Default is None.
    Returns:
        xr.Dataset: Dataset containing the calibration factor (K), intercept (b), correlation (R), and their errors over time.
    """

    
    # Check that beta_part and rcs_elastic have the same dimensions
    if rcs_elastic.shape != particle_backscatter.shape:
        raise ValueError("rcs_elastic and particle_backscatter must have the same shape.")
    
    # Define height range for calibration
    height_min = full_overlap_height 
    height_max = reference_height
    
    # Cut signal and backscatter profiles to the desired height range
    rcs_elastic = rcs_elastic.sel(range=slice(height_min, height_max))
    beta_part = particle_backscatter.sel(range=slice(height_min, height_max))
    
    #Define range and time profiles
    range_profile = rcs_elastic.range.values
    time_profile = rcs_elastic.time.values
    
    # Mask molecular properties to the desired height range
    mol_properties = {
    "range": np.array(mol_properties["range"].values),
    "molecular_beta": np.array(mol_properties["molecular_beta"].values),
    "molecular_alpha": np.array(mol_properties["molecular_alpha"].values),
}  # es un valor único}
    
    range_mask = (mol_properties["range"] >= height_min) & (mol_properties["range"] <= height_max)
    mol_properties = {
    key: val[range_mask] if val.shape == range_mask.shape else val
    for key, val in mol_properties.items()}
    
    #Repeat the molecular properties for each time
    molecular_beta2D = np.tile(mol_properties["molecular_beta"][np.newaxis, :], (len(time_profile), 1))
    
    molecular_alpha2D = np.tile(mol_properties["molecular_alpha"][np.newaxis, :], (len(time_profile), 1))

    alpha_part_klett = beta_part * lr_part

    #Calculate transmittance
    transmittance_elastic = np.zeros((len(time_profile), len(range_profile)))
    
    for i, t_i in enumerate(time_profile):
        transmittance_elastic[i, :] = transmittance(molecular_alpha2D[i, :] + alpha_part_klett[i, :], range_profile) 
        
    
    #Calculate attenuated backscatter
    beta_att = (beta_part + molecular_beta2D) * transmittance_elastic**2

    # Convert to xarray
    beta_att = xr.DataArray(beta_att, coords={'time': time_profile, 'range': range_profile}, dims=['time', 'range'])
    rcs_elastic = xr.DataArray(rcs_elastic.values, coords={'time': time_profile, 'range': range_profile}, dims=['time', 'range'])
 

    calibration_factors = []
    intercepts = []
    correlations = []
    K_errors = []
    b_errors = []
   
    # Height linear fit for each time step
    for t in time_profile:
        beta_ = beta_att.sel(time=t, method='nearest', range=range_profile)

        rcs_ = rcs_elastic.sel(time=t, method='nearest', range=range_profile)

        mask =  ~np.isnan(beta_) & ~np.isnan(rcs_)

        if mask.sum() >= 2:
            # Ajuste lineal con matriz de covarianza
            (m, b), cov = np.polyfit(beta_[mask], rcs_[mask], deg=1, cov=True)

            # Incertidumbres (desviación estándar de m y b)
            sigma_m = np.sqrt(cov[0, 0])
            sigma_b = np.sqrt(cov[1, 1])

            # Correlación
            R_corr = np.corrcoef(beta_[mask], rcs_[mask])[0, 1]
        else:
            m, b, R_corr = np.nan, np.nan, np.nan

        calibration_factors.append(m)
        intercepts.append(b)
        correlations.append(R_corr)
        K_errors.append(sigma_m)
        b_errors.append(sigma_b)
        

    # Plotting for a specific time if plot_specific_time is provided
        plot_specific_time = kwargs.get('plot_specific_time', None)
        if plot_specific_time is not None and np.datetime64(t) == np.datetime64(plot_specific_time):
            fontsize = 19
            plt.figure(figsize=(15, 6))
            plt.title(f'{plot_specific_time} ', fontsize=fontsize)
            heights = beta_.coords['range'].values
            heights_masked = heights[mask.values]
            
            plt.plot(beta_[mask], m * beta_[mask] + b, color='black', lw=2, label='Linear fit')
            sc = plt.scatter(beta_[mask], rcs_[mask], c=heights_masked, cmap='jet')

            cbar = plt.colorbar(sc, label='Height [m]')

            # Aumenta el tamaño de la fuente de los ticks de la colorbar
            cbar.ax.tick_params(labelsize=fontsize)

            # También puedes aumentar el tamaño de la etiqueta de la colorbar si quieres
            cbar.set_label('Height [m]', fontsize=fontsize)

            # Aumenta el tamaño de las etiquetas del eje x e y
            plt.tick_params(labelsize=fontsize)
            
            
            plt.xlabel(r'$\beta_{att}$, [m$^{-1}$ sr$^{-1}$]', fontsize=fontsize)
            plt.ylabel("RCS, [a.u.]", fontsize=fontsize)
            plt.grid()
            plt.legend(loc='upper left', fontsize=fontsize)
            plt.gca().set_facecolor('white')

            # Escalar K (pendiente) y su error
            exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
            m_scaled = m / 10**exp_m
            sigma_m_scaled = sigma_m / 10**exp_m

            # Escalar b (intercepto) y su error
            exp_b = int(np.floor(np.log10(abs(b)))) if b != 0 else 0
            b_scaled = b / 10**exp_b
            sigma_b_scaled = sigma_b / 10**exp_b

            # Mostrar en texto con misma potencia
            plt.text(
                0.02, 0.7,
                rf'$K = ({m_scaled:.4f} \pm {sigma_m_scaled:.4f})e{{{exp_m}}}$ a.u.'
                f'\n$R_f = {R_corr:.2f}$\n'
                rf'$b = ({b_scaled:.2f} \pm {sigma_b_scaled:.2f})e{{{exp_b}}}$ a.u.',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                fontsize=fontsize
            )

            plt.tight_layout()
            plt.show()
            
    calibration_factors = np.array(calibration_factors)
    intercepts = np.array(intercepts)
    correlations = np.array(correlations)
    K_errors = np.array(K_errors)
    b_errors = np.array(b_errors)

    calibration_dataset = xr.Dataset({
            "K_time_series": calibration_factors,  
            "b_time_series": intercepts,
            "R_time_series": correlations,
            "K_errors": K_errors,
            "b_errors": b_errors
        })

    if debug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
            fontsize = 19

            # --- Subplot 1: K (pendiente)
            K_mean = calibration_factors.mean()
            std = calibration_factors.std()

            mean_val = K_mean
            std_val = std

            exp = int(np.floor(np.log10(mean_val)))
            mean_norm = mean_val / (10 ** exp)
            std_norm = std_val / (10 ** exp)

            ax1.errorbar(
                time_profile,
                calibration_factors,
                yerr=K_errors,
                fmt='o-',
                color='darkred',
                ecolor='darkred',
                elinewidth=4,
                capsize=7,
                markersize=7,
                label='Experimental points'
            )
            ax1.axhline(K_mean, color='black', lw=2, linestyle='--', label='Mean')
            ax1.set_title(r'$K$ from $\beta_{att}$ fitting' , fontsize=fontsize)
            ax1.set_ylabel(r'$K$, [a.u.]', fontsize=fontsize)
            ax1.grid()
            ax1.set_facecolor('white')
            ax1.tick_params(axis='both', labelsize=fontsize-2)
            ax1.legend(fontsize=fontsize-2)

            # Añadir texto de la media
            ax1.text(
                0.01, 0.2,
                r'Mean $\pm$ STD:' f'\n({mean_norm:.2f}' + r'$\pm$' + f' {std_norm:.2f})e{exp} a.u.',
                transform=ax1.transAxes,
                fontsize=fontsize,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
            )

            # --- Subplot 2: b (intercepto)
            ax2.errorbar(
                time_profile,
                intercepts,
                yerr=b_errors,
                fmt='o-',
                color='darkgreen',
                ecolor='darkgreen',
                elinewidth=2,
                capsize=5,
                markersize=7
            )
            ax2.set_title(r'Intercept $b$ from $\beta_{att}$ fitting', fontsize=fontsize)
            ax2.set_xlabel('Time, [UTC]', fontsize=fontsize)
            ax2.set_ylabel('b, [a.u.]', fontsize=fontsize)
            ax2.grid()
            ax2.set_facecolor('white')
            ax2.tick_params(axis='both', labelsize=fontsize-2)
            
            calibration_dataset = xr.Dataset({
                "K_time_series": calibration_factors, # Time series of K calculated with a linear fit each time step
                "b_time_series": intercepts, # Time series of intercepts calculated with a linear fit each time step
                "R_time_series": correlations, # Time series of correlations calculated with a linear fit each time step
                "K_errors": K_errors, # Time series of K errors calculated with a linear fit each time step
                "b_errors": b_errors, # Time series of intercept errors calculated with a linear fit each time step
                "K_mean": K_mean, # Mean of K time series
                "K_std": std # Standard deviation of K time series
            })
           
    return calibration_dataset




def get_calibration_factor_aod_fit(
    rcs_elastic: xr.DataArray,
    mol_properties: pd.DataFrame,
    lr_part: float,
    particle_backscatter: xr.DataArray,
    reference_height: float = 6000.,
    debug: bool = False,
) -> xr.Dataset:
    """
    Compute the lidar calibration factor for the elastic channel using the aerosol optical depth fitting method over a selected time range.
    Args:
        rcs_elastic (xr.DataArray): Quicklook of the elastic range-corrected signal (time, range).
        particle_backscatter (xr.DataArray): Quicklook of the particle backscatter coefficient (time, range).
        mol_properties (pd.DataFrame): Molecular properties with columns ['molecular_beta', 'molecular_alpha'].
        lr_part (float): Lidar ratio of particles (sr).
        reference_height (float): Upper height limit for calibration. Default is 7000 m.
        debug (bool): If True, plot the temporal linear fit. Default is False.
    Returns:
        xr.Dataset: Dataset containing the calibration factor (K), slope (m), correlation (R), and their errors.
    """

    
    # Check that beta_part and rcs_elastic have the same dimensions
    if rcs_elastic.shape != particle_backscatter.shape:
        raise ValueError("rcs_elastic and particle_backscatter must have the same shape.")
    
    # Select time and range profiles
    range_profile = rcs_elastic.range.values
    time_profile = rcs_elastic.time.values
 
   # Repeat molecular properties for each time
    mol_properties_dict = {
    "range": np.array(mol_properties["range"].values),
    "molecular_beta": np.array(mol_properties["molecular_beta"].values),
    "molecular_alpha": np.array(mol_properties["molecular_alpha"].values),
}  # es un valor único
    
    molecular_beta2D = np.tile(mol_properties_dict["molecular_beta"][np.newaxis, :], (len(time_profile), 1))
    molecular_alpha2D = np.tile(mol_properties_dict["molecular_alpha"][np.newaxis, :], (len(time_profile), 1))

    # Calculate alpha part using lidar ratio
    alpha_part_klett = particle_backscatter * lr_part

    # Calculate molecular transmittance
    transmittance_molecular = np.zeros((len(time_profile), len(range_profile)))
    
    for i, t_i in enumerate(time_profile):
        transmittance_molecular[i, :] = transmittance(molecular_alpha2D[i, :], range_profile) 

    # Calculate RCS mod y AOD
    rcs_mod = rcs_elastic / ((molecular_beta2D + particle_backscatter) * transmittance_molecular**2)
    aod = integrate.trapezoid(alpha_part_klett, x=range_profile, axis=1)

    # Convert to xarray
    rcs_mod = xr.DataArray(rcs_mod, coords={'time': time_profile, 'range': range_profile}, dims=['time', 'range'])

    # Select at reference height and eviter log of zero or negative
    rcs_mod_sel = rcs_mod.sel(range=reference_height, method='nearest')
    rcs_mod_sel = rcs_mod_sel.where(rcs_mod_sel > 0)

    # Calculate log
    log_rcs_mod_sel = np.log(rcs_mod_sel)

    # Filter valid times
    idx = np.logical_and(~np.isnan(log_rcs_mod_sel), ~np.isnan(aod))

    # Linear fit
    (m, b), cov = np.polyfit(aod[idx], log_rcs_mod_sel[idx], deg=1, cov=True)

    # Uncertainties (standard deviation of m and b)
    sigma_m = np.sqrt(cov[0, 0])
    sigma_b = np.sqrt(cov[1, 1])

    R_correlation = np.corrcoef(aod[idx], log_rcs_mod_sel[idx])[0, 1]
    calibration_factor = np.exp(b)
    sigma_K = sigma_b * calibration_factor  # Error en K
    
    calibration_dataset = xr.Dataset({
        "K_value": calibration_factor,
        "K_error": sigma_K,
        "slope": m,
        "slope_error": sigma_m,
        "R_correlation": R_correlation
    })
    
    # Gráfico de diagnóstico
    if debug:
        plt.figure(figsize=(15, 6))
        fontsize = 19
        fig, ax = plt.subplots(figsize=(15, 6))
        # Repetir time_profile a lo largo del eje vertical, usar para colorear puntos
        time_valid = time_profile[idx]
        time_numeric = mdates.date2num(time_valid)

        sc = ax.scatter(aod[idx], log_rcs_mod_sel[idx], lw=0, marker='o', c=time_numeric, cmap='jet')
        cbar = plt.colorbar(sc, ax=ax, label='Time [UTC]')
        cbar.set_label('Time [UTC]', fontsize=fontsize) 
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        cbar.ax.tick_params(labelsize=fontsize)

        ax.plot(aod[idx], m * aod[idx] + b, color='black', lw=2, label='Linear fit')
        ax.set_title(f'Linear fit at {reference_height:.0f} m ', fontsize=fontsize)
        ax.set_xlabel(r'$AOD_{lidar}$ [#]', fontsize=fontsize)
        ax.set_ylabel(r'$ln\left(\frac{RCS}{\beta \; T_m^2} \right)$ [#]', fontsize=fontsize)
        ax.grid()
        ax.tick_params(axis='both', labelsize=fontsize-2)
        ax.legend(loc='upper right', fontsize=fontsize)
        ax.set_facecolor('white')
        
        exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
        m_scaled = m / 10**exp_m
        sigma_m_scaled = sigma_m / 10**exp_m

        # Escalar b (intercepto) y su error
        exp_K = int(np.floor(np.log10(abs(calibration_factor)))) if calibration_factor != 0 else 0
        K_scaled = calibration_factor / 10**exp_K
        sigma_K_scaled = sigma_K / 10**exp_K
        
        
        plt.text(
                0.02, 0.3,
                rf'$K = ({K_scaled:.3f} \pm {sigma_K_scaled:.3f})e{{{exp_K}}}$ a.u.'
                f'\n$R_f = {R_correlation:.2f}$\n'
                rf'$m = ({m_scaled:.2f} \pm {sigma_m_scaled:.2f})e{{{exp_m}}}$',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                fontsize=fontsize
        )
        plt.tight_layout()
        
        if "K_lidar" in mol_properties:
            K_lidar_synt = mol_properties["K_lidar"].iloc[0]  # tomar solo un valor
            relative_diff = (calibration_factor - K_lidar_synt) / K_lidar_synt * 100
            plt.text(0.05, 0.5,
                      r'$K_{synt}$= ' + f'({K_lidar_synt:.3e}) a.u. \n Relative difference: {relative_diff:.3e} %',
                      transform=plt.gca().transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                      fontsize=fontsize
                      )
            fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'test_lidar_aod_calibration.png')
        # plt.show()
    return calibration_dataset







def get_calibration_factor_aod_photometer_fit(
    rcs_elastic: xr.DataArray,
    mol_properties: pd.DataFrame,
    aod_photometer: np.ndarray,
    reference_height: float = 6000.,
    debug: bool = False,
) -> xr.Dataset:
    """
    Compute the lidar calibration factor for the elastic channel using the aerosol optical depth fitting method over a selected time range.
    Args:
        rcs_elastic (xr.DataArray): Quicklook of the elastic range-corrected signal (time, range).
        mol_properties (pd.DataFrame): Molecular properties with columns ['molecular_beta', 'molecular_alpha'].
        aod_photometer (np.ndarray): Array of AOD values from a reference AERONET photometer (time,).
        lr_part (float): Lidar ratio of particles (sr).
        reference_height (float): Upper height limit for calibration. Default is 7000 m.
        debug (bool): If True, plot the temporal linear fit. Default is False.
    Returns:
        xr.Dataset: Dataset containing the calibration factor (K), slope (m), correlation (R), and their errors.
    """
    
    
    # Check that beta_part and rcs_elastic have the same dimensions
    if rcs_elastic.time.shape != aod_photometer.shape:
        raise ValueError("rcs_elastic and aod_photometer must have the same shape.")

    # Select time and range profiles
    range_profile = rcs_elastic.range.values
    time_profile = rcs_elastic.time.values

    # Expand molecular properties to 2D
    mol_properties_dict = {
    "range": np.array(mol_properties["range"].values),
    "molecular_beta": np.array(mol_properties["molecular_beta"].values),
    "molecular_alpha": np.array(mol_properties["molecular_alpha"].values),
}  # es un valor único
    
    molecular_beta2D = np.tile(mol_properties_dict["molecular_beta"][np.newaxis, :], (len(time_profile), 1))
    molecular_alpha2D = np.tile(mol_properties_dict["molecular_alpha"][np.newaxis, :], (len(time_profile), 1))

    #Calculate molecular transmittance
    transmittance_molecular = np.zeros((len(time_profile), len(range_profile)))
    
    for i, t_i in enumerate(time_profile):
        transmittance_molecular[i, :] = transmittance(molecular_alpha2D[i, :], range_profile) 

    # Calculate RCS mod 
    rcs_mod = rcs_elastic / (molecular_beta2D * transmittance_molecular**2)
    
    # Convert to xarray
    rcs_mod = xr.DataArray(rcs_mod, coords={'time': time_profile, 'range': range_profile}, dims=['time', 'range'])

    # Select at reference height and eviter log of zero or negative
    rcs_mod_sel = rcs_mod.sel(range=reference_height, method='nearest')
    rcs_mod_sel = rcs_mod_sel.where(rcs_mod_sel > 0)

    # Calcular log 
    log_rcs_mod_sel = np.log(rcs_mod_sel)

    # Enmascarar datos inválidos
    AOD_photometer = np.asarray(aod_photometer)
    idx = np.logical_and(~np.isnan(log_rcs_mod_sel), ~np.isnan(AOD_photometer))

    # Ajuste lineal
    (m, b), cov = np.polyfit(AOD_photometer[idx], log_rcs_mod_sel[idx], deg=1, cov=True)

    # Incertidumbres (desviación estándar de m y b)
    sigma_m = np.sqrt(cov[0, 0])
    sigma_b = np.sqrt(cov[1, 1])

    R_correlation = np.corrcoef(AOD_photometer[idx], log_rcs_mod_sel[idx])[0, 1]
    calibration_factor = np.exp(b)
    sigma_K = sigma_b * calibration_factor  # Error en K
    
    calibration_dataset = xr.Dataset({
        "K_value": calibration_factor,
        "K_error": sigma_K,
        "slope": m,
        "slope_error": sigma_m,
        "R_correlation": R_correlation
    })
    
    # Gráfico si se activa debug
    if debug:
        plt.figure(figsize=(15, 6))

        fontsize = 19
        fig, ax = plt.subplots(figsize=(15, 6))
        # Repetir time_profile a lo largo del eje vertical, usar para colorear puntos
        # Tiempos válidos directamente
        time_valid = time_profile[idx]
        time_numeric = mdates.date2num(time_valid)

        sc = ax.scatter(AOD_photometer[idx], log_rcs_mod_sel[idx], lw=2, marker='o', c=time_numeric, cmap='jet')
        cbar = plt.colorbar(sc, ax=ax, label='Time [UTC]')
        cbar.set_label('Time [UTC]', fontsize=fontsize) 
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        cbar.ax.tick_params(labelsize=fontsize)

        ax.plot(AOD_photometer[idx], m * AOD_photometer[idx] + b, color='black', lw=2, label='Linear fit')

        ax.set_title(f'Linear fit at {reference_height:.0f} m ', fontsize=fontsize)
        ax.set_xlabel(r'$AOD_{photo}$ [#]', fontsize=fontsize)
        ax.set_ylabel(r'$ln\left(\frac{RCS}{\beta \; T_m^2} \right)$ [#]', fontsize=fontsize)
        ax.grid()

        ax.legend(loc='upper right', fontsize=fontsize)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', labelsize=fontsize-2)
        exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
        m_scaled = m / 10**exp_m
        sigma_m_scaled = sigma_m / 10**exp_m

        # Escalar b (intercepto) y su error
        exp_K = int(np.floor(np.log10(abs(calibration_factor)))) if calibration_factor != 0 else 0
        K_scaled = calibration_factor / 10**exp_K
        sigma_K_scaled = sigma_K / 10**exp_K

        plt.text(
                0.02, 0.3,
                rf'$K = ({K_scaled:.3f} \pm {sigma_K_scaled:.3f})e{{{exp_K}}}$ a.u.'
                f'\n$R_f = {R_correlation:.2f}$\n'
                rf'$m = ({m_scaled:.2f} \pm {sigma_m_scaled:.2f})e{{{exp_m}}}$',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), 
                fontsize=fontsize
        )
        plt.tight_layout()
        if "K_lidar" in mol_properties:
            K_lidar_synt = mol_properties["K_lidar"].iloc[0]  # tomar solo un valor
            relative_diff = (calibration_factor - K_lidar_synt) / K_lidar_synt * 100
            plt.text(0.05, 0.5,
                      r'$K_{synt}$= ' + f'({K_lidar_synt:.3e}) a.u. \n Relative difference: {relative_diff:.3e} %',
                      transform=plt.gca().transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                      fontsize=fontsize
                      )
            fig.savefig(Path(__file__).parent.parent.parent.parent / 'tests' / 'figures' / 'test_photo_aod_calibration.png')

        plt.show()

    return calibration_dataset
