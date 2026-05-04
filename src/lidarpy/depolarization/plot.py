from pathlib import Path
from pdb import set_trace
import matplotlib.pyplot as plt
import xarray as xr

from lidarpy.utils.utils import signal_to_rcs


def plot_eta_star_calib(
    d: xr.Dataset,
    wavelength: int = 532,
    output_dir: Path | str | None = None,
    telescope: str = "x",
    **kwargs,
) -> Path | None:
    """It plots the calibration factors of a specific channel (i.e., wavelength and telescope) from dataset load with `lidarpy.depolarization.calibration.eta_star_reader`. 

    Args:
    
        d (xr.Dataset): calibration factor dataset.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif output_dir is None:
        output_dir = Path.cwd()

    if not output_dir.exists() or not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} not found.")

    try:
        channel_str = f"{wavelength}{telescope}"

        # FIXME: This search channels should be provisional and shoud be make by
        # passing a channels array with the eta star dataset
        variables: list[str] = list(d.variables.keys())  # type: ignore
        search_channels = filter(lambda v: v.startswith("eta_star_profile_"), variables)

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10), sharey=True) # type: ignore
        y_lim = kwargs.get("y_lim", (0, 5000))

        # Title
        date_str, hour_str = d.attrs['calibration_datetime'].split('_')        
        system = d.attrs['system'].upper()
        fig_title_1 = f"{system} calibration depolarization analysis | {date_str} - {hour_str} UTC | {channel_str}"
        cal_height_an_min, cal_height_an_max = d[f'eta_star_mean_{channel_str}a'].attrs['min_range_m'], d[f'eta_star_mean_{channel_str}a'].attrs['max_range_m']
        cal_height_pc_min, cal_height_pc_max = d[f'eta_star_mean_{channel_str}p'].attrs['min_range_m'], d[f'eta_star_mean_{channel_str}p'].attrs['max_range_m']
        eta_star_an = d[f"eta_star_mean_{channel_str}a"].values.item()
        eta_star_pc = d[f"eta_star_mean_{channel_str}p"].values.item()
        std_eta_star_an = d[f"eta_star_standard_deviation_{channel_str}a"].values.item()
        std_eta_star_pc = d[f"eta_star_standard_deviation_{channel_str}p"].values.item()
        fig_title_2 = rf"Analog calib. factor [{cal_height_an_min:3.1f} - {cal_height_an_max:3.1f} m] = {eta_star_an:3.2f} $\pm$ {std_eta_star_an:3.2f}"
        fig_title_3 = rf"Photocounting calib. factor [{cal_height_pc_min:3.1f} - {cal_height_pc_max:3.1f} m] = {eta_star_pc:3.2f} $\pm$ {std_eta_star_pc:3.2f}"

        plt.suptitle(f"{fig_title_1} \n {fig_title_2} \n {fig_title_3}", fontsize=16)

        # ANALOG
        signal_to_rcs(d[f"signal_T_P45_{channel_str}a"], d.range).plot(
            ax=ax1, y="range", lw=2, c="lime", label=r"RCS$^T_{+45}$. AN"
        )  # type: ignore
        signal_to_rcs(d[f"signal_R_N45_{channel_str}a"], d.range).plot(
            ax=ax1, y="range", lw=2, c="darkgreen", label=r"RCS$^R_{-45}$. AN"
        )  # type: ignore
        signal_to_rcs(d[f"signal_T_N45_{channel_str}a"], d.range).plot(
            ax=ax1, y="range", lw=2, c="r", label=r"RCS$^T_{-45}$. AN"
        )  # type: ignore
        signal_to_rcs(d[f"signal_R_P45_{channel_str}a"], d.range).plot(
            ax=ax1, y="range", lw=2, c="darkred", label=r"RCS$^R_{+45}$. AN"
        )  # type: ignore

        # PHOTONCOUNTING
        signal_to_rcs(d[f"signal_T_P45_{channel_str}p"], d.range).plot(ax=ax1, y="range", lw=2, c="deepskyblue", label=r"RCS$^T_{+45}$. PC")  # type: ignore
        signal_to_rcs(d[f"signal_R_N45_{channel_str}p"], d.range).plot(
            ax=ax1, y="range", lw=2, c="b", label=r"RCS$^R_{-45}$. PC"
        )  # type: ignore
        signal_to_rcs(d[f"signal_T_N45_{channel_str}p"], d.range).plot(
            ax=ax1, y="range", lw=2, c="magenta", label=r"RCS$^T_{-45}$. PC"
        )  # type: ignore
        signal_to_rcs(d[f"signal_R_P45_{channel_str}p"], d.range).plot(
            ax=ax1, y="range", lw=2, c="darkmagenta", label=r"RCS$^R_{+45}$. PC"
        )  # type: ignore

        ax1.grid()
        ax1.axes.set_xlabel(r"RCS [a.u.]")
        ax1.axes.set_ylabel(r"Height [km, agl]")
        ax1.axes.set_ylim(y_lim)  # type: ignore
        ax1.axes.set_xscale("log")
        ax1.legend(fontsize="small", loc="upper left")

        (d[f"signal_R_P45_{channel_str}a"] / d[f"signal_T_P45_{channel_str}a"]).plot(
            ax=ax2, y="range", lw=2, c="blue", label=r"AN at +45"
        )  # type: ignore
        (d[f"signal_R_N45_{channel_str}a"] / d[f"signal_T_N45_{channel_str}a"]).plot(
            ax=ax2, y="range", lw=2, c="red", label=r"AN at -45"
        )  # type: ignore
        (d[f"signal_R_P45_{channel_str}p"] / d[f"signal_T_P45_{channel_str}p"]).plot(
            ax=ax2, y="range", lw=2, c="green", label=r"PC at +45"
        )  # type: ignore
        (d[f"signal_R_N45_{channel_str}p"] / d[f"signal_T_N45_{channel_str}p"]).plot(
            ax=ax2, y="range", lw=2, c="black", label=r"PC at -45"
        )  # type: ignore

        ax2.grid()
        ax2.axes.set_xlabel(r"Signal Ratio [R / T]")
        ax2.axes.set_ylabel(r"")
        ax2.legend(fontsize="small", loc="upper left")
        ax2.axes.set_xlim(kwargs.get("signal_ratio_x_lim", (0, 10)))

        d[f"eta_star_profile_{channel_str}a"].plot(
            ax=ax3, y="range", c="g", label=r"AN"
        )  # type: ignore
        d[f"eta_star_profile_{channel_str}p"].plot(
            ax=ax3, y="range", c="m", label=r"PC"
        )  # type: ignore
        ax3.grid()
        ax3.axes.set_xlabel(r"Calibration Factor")
        ax3.axes.set_ylabel(r"")
        ax3.axes.set_xlim(kwargs.get("calib_factor_x_lim", (0, 5)))
        ax3.legend(fontsize="small", loc="upper left")
        # ax3.axes.set_xlim((0, 1))

        plt.tight_layout()
    except ValueError:
        output_file = None
        raise ValueError

    if output_dir.exists() and output_dir.is_dir():
        calibration_datetime = d.attrs["calibration_datetime"]
        output_file = (
            output_dir / f"calibration_{channel_str}_{calibration_datetime}.png"
        )   

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        output_file = None
        raise FileNotFoundError("`output_dir` not found.")
    return output_file
