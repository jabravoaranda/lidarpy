import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

def gluing_licel(height_bins, analog_signal, photon_counting_signal, 
                 lower_toggle_rate=0.5e6, upper_toggle_rate=10e6, 
                 toggle_rate=None, plot=False, channel_name=None):
    """
    Apply gluing between analog and photon counting signals using linear regression.

    Parameters
    ----------
    height_bins : ndarray
        Vertical range bins (e.g., height in meters).
    analog_signal : ndarray
        Analog signal (arbitrary units or counts/s).
    photon_counting_signal : ndarray
        Photon counting signal (counts/s).
    lower_toggle_rate : float
        Lower toggle rate threshold [Hz], typically 0.5 MHz.
    upper_toggle_rate : float
        Upper toggle rate threshold [Hz], typically 10 MHz.
    toggle_rate : ndarray, optional
        Actual toggle rate per bin (same shape as signals). If None, photon counting signal is used.
    plot : bool
        If True, display diagnostic plots.
    channel_name : str, optional
        Name of the channel being processed (used for logging).

    Returns
    -------
    glued_signal : ndarray or np.nan
        Final signal after gluing. np.nan if gluing failed.
    coefficients : tuple or None
        Linear regression coefficients (slope, intercept), or None if failed.
    """

    try:
        toggle_rate = photon_counting_signal if toggle_rate is None else toggle_rate

        # 1. Identify the valid region for gluing
        valid_mask = (toggle_rate > lower_toggle_rate) & (toggle_rate < upper_toggle_rate)

        if np.sum(valid_mask) < 2:
            msg = f"[{channel_name}] Not enough valid points in toggle region for regression."
            warnings.warn(msg)
            return np.nan, None

        # 2. Linear regression: PC ≈ a * Analog + b
        slope, intercept, *_ = linregress(
            analog_signal[valid_mask], photon_counting_signal[valid_mask]
        )

        # 3. Scale the analog signal
        analog_scaled = slope * analog_signal + intercept

        # 4. Combine both signals using toggle threshold
        glued_signal = np.where(toggle_rate < upper_toggle_rate,
                                photon_counting_signal,
                                analog_scaled)

        if plot:
            plt.figure(figsize=(8, 5))
            if photon_counting_signal.ndim == 2:
                pc_mean = np.mean(photon_counting_signal, axis=0)
                an_scaled_mean = np.mean(analog_scaled, axis=0)
                glued_mean = np.mean(glued_signal, axis=0)
            else:
                pc_mean = photon_counting_signal
                an_scaled_mean = analog_scaled
                glued_mean = glued_signal

            plt.plot(pc_mean, height_bins, '--', label='Photon Counting', alpha=0.6)
            plt.plot(an_scaled_mean, height_bins, '--', label='Scaled Analog', alpha=0.6)
            plt.plot(glued_mean, height_bins, label='Glued Signal', linewidth=2)

            plt.gca().invert_yaxis()
            plt.xlabel("Signal (counts/s or arbitrary units)")
            plt.ylabel("Height (m)")
            plt.title("Gluing Between Analog and Photon Counting Signals")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return glued_signal, (slope, intercept)

    except Exception as e:
        msg = f"[{channel_name}] Gluing failed: {str(e)}"
        warnings.warn(msg)
        return np.nan, None
