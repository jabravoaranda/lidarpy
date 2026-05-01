import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

def gluing_book(
    dataset,
    analog_var,
    pc_var,
    min_alt=1000,
    max_alt=8000,
    smoothing_window=11,
    plot=True
):
    # cuerpo de la función...

    signal_analog = dataset[analog_var].values
    signal_pc = dataset[pc_var].values
    range_array = dataset["range"].values

    merged_signal = np.empty_like(signal_analog)
    merge_infos = []

    for t in range(signal_analog.shape[0]):
        sig_an = signal_analog[t]
        sig_pc = signal_pc[t]
        
        # ... el código que tienes dentro, adaptado para sig_an y sig_pc ...
        # Usar el mismo código que tienes pero con estas señales 1D.

        # Aquí por simplicidad copio la lógica para 1D:
        valid_mask = (sig_pc > 0) & (sig_an > 0)
        ratio = np.full_like(sig_an, np.nan)
        ratio[valid_mask] = sig_an[valid_mask] / sig_pc[valid_mask]
        ratio_smooth = uniform_filter1d(ratio, size=smoothing_window, mode='nearest')

        idx_range = np.where((range_array >= min_alt) & (range_array <= max_alt))[0]
        alt_slice = slice(idx_range[0], idx_range[-1] + 1)

        grad = np.gradient(ratio_smooth[alt_slice])
        grad_abs = np.abs(grad)

        low_gradient_idx = np.argsort(grad_abs)[:100]
        selected_idx = idx_range[low_gradient_idx]

        crossover_start = range_array[selected_idx].min()
        crossover_end = range_array[selected_idx].max()
        merge_point = (crossover_start + crossover_end) / 2.0

        idx_crossover = np.where((range_array >= crossover_start) & (range_array <= crossover_end))[0]
        rho = np.nanmean(sig_an[idx_crossover] / sig_pc[idx_crossover])
        signal_pc_scaled = rho * sig_pc

        merge_idx = np.argmin(np.abs(range_array - merge_point))

        merged_signal[t, :merge_idx] = sig_an[:merge_idx]
        merged_signal[t, merge_idx:] = signal_pc_scaled[merge_idx:]

        merge_infos.append({
            'scale_factor': rho,
            'crossover_start': crossover_start,
            'crossover_end': crossover_end,
            'merge_point': merge_point
        })

    # Opcional: plot del último perfil
    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        axs[0].semilogx(signal_analog[-1], range_array, label='Analog', alpha=0.7)
        axs[0].semilogx(signal_pc_scaled, range_array, label='PC (scaled)', alpha=0.7)
        axs[0].semilogx(merged_signal[-1], range_array, label='Merged', lw=2)
        axs[0].axhline(merge_point, color='gray', linestyle='--', label='Merge point')
        axs[0].set_xlabel('Signal (a.u.)')
        axs[0].set_ylabel('Altitude [m]')
        axs[0].set_title('Merged Signal (last time step)')
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(ratio_smooth, range_array, label='Smoothed ratio (last time step)')
        axs[1].axhline(crossover_start, color='green', linestyle='--', label='Crossover start')
        axs[1].axhline(crossover_end, color='red', linestyle='--', label='Crossover end')
        axs[1].set_xlabel('Analog / PC ratio')
        axs[1].set_title('Ratio Profile')
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show()

    return merged_signal, merge_infos

def linear_model(Va, a, b):
    """
    Linear model to fit the analog signal (Va) to the photon-counting signal (Vpc):
    V'_a = a * Va + b
    """
    return a * Va + b


def compute_scaling_params(Va, Vpc):
    """
    Fits the analog signal (Va) to the photon-counting signal (Vpc).
    """
    Va = np.asarray(Va).flatten()
    Vpc = np.asarray(Vpc).flatten()
    popt, _ = curve_fit(linear_model, Va, Vpc)
    return popt

def compute_error_norm(Va, Vpc, a, b):
    """
    Computes the squared error norm between the photon-counting signal and
    the scaled-and-offset analog signal:

        ε² = || Vpc - (a * Va + b) ||²         [Licel, 2007 Eq. (1)]

    Parameters:
    - Va: analog signal
    - Vpc: photon-counting signal
    - a, b: scaling and offset parameters

    Returns:
    - error norm (float)
    """
    V_adjusted = a * Va + b
    return np.linalg.norm(Vpc - V_adjusted)


def find_optimal_interval(range_array, Va, Vpc, window_length=1000, step=100):
    """
    Implements the optimized gluing method from Lange et al. (2011).

    Parameters:
    - range_array: array-like, altitude values [m]
    - Va: analog signal (same size as range_array)
    - Vpc: photon-counting signal (same size as range_array)
    - window_length: size of fitting window in meters (default: 1000 m)
    - step: sliding step in meters (default: 100 m)

    Returns:
    - R_A_opt, R_B_opt: optimal fitting range boundaries
    - R_opt: optimal center of the fitting interval
    - centers: array of window centers
    - scaling_factors: array of optimal 'a' values for each window
    """
    centers = []
    scaling_factors = []
    errors = []

    range_array = np.asarray(range_array)
    Va = np.asarray(Va)
    Vpc = np.asarray(Vpc)

    min_range = range_array[0] + window_length / 2
    max_range = range_array[-1] - window_length / 2

    if max_range <= min_range:
        raise ValueError("Range array too small for the given window length.")

    for center in np.arange(min_range, max_range, step):
        i_start = np.argmin(np.abs(range_array - (center - window_length / 2)))
        i_end = np.argmin(np.abs(range_array - (center + window_length / 2)))

        if i_end <= i_start:
            continue

        Va_win = Va[i_start:i_end]
        Vpc_win = Vpc[i_start:i_end]

        if len(Va_win) < 10 or len(Vpc_win) < 10:
            continue

        try:
            a, b = compute_scaling_params(Va_win, Vpc_win)
            err = compute_error_norm(Va_win, Vpc_win, a, b)

            centers.append(center)
            scaling_factors.append(a)
            errors.append(err)
        except Exception as e:
            continue  # skip problematic windows

    if not scaling_factors:
        raise RuntimeError("No valid fitting windows found. Adjust window size or range.")

    centers = np.array(centers)
    scaling_factors = np.array(scaling_factors)
    errors = np.array(errors)

    opt_idx = np.argmin(scaling_factors)
    R_opt = centers[opt_idx]
    a_opt = scaling_factors[opt_idx]
    threshold = 1.01 * a_opt

    # Define interval where ai <= 1.01 * a_opt
    within_threshold = scaling_factors <= threshold
    valid_centers = centers[within_threshold]

    if len(valid_centers) == 0:
        raise RuntimeError("No valid fitting interval found within threshold.")

    R_A_opt = valid_centers[0]
    R_B_opt = valid_centers[-1]

    return R_A_opt, R_B_opt, R_opt, centers, scaling_factors


def gluing_V3(Va, Vpc, range_array, R_A, R_B):
    """
    Applies the final gluing using the optimal fitting range [R_A, R_B]:

    Parameters:
    - Va: analog signal
    - Vpc: photon-counting signal
    - range_array: altitude values [m]
    - R_A, R_B: limits of the fitting interval

    Returns:
    - V_glued: final glued signal
    - a, b: optimal fitting parameters
    """
    i_start = np.argmin(np.abs(range_array - R_A))
    i_end = np.argmin(np.abs(range_array - R_B))

    a, b = compute_scaling_params(Va[i_start:i_end], Vpc[i_start:i_end])
    V_glued = a * Va + b

    return V_glued, a, b
