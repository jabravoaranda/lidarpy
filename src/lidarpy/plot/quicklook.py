import numpy as np
import xarray as xr
from pathlib import Path
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import colors
from typing import List, Tuple

from lidarpy.nc_convert.measurement import merge_measurements_by_date
from lidarpy.utils.types import MeasurementType
from lidarpy.general_utils import plot
from lidarpy.general_utils.io import read_yaml_from_info
from lidarpy.general_utils.dates import parse_datetime
from lidarpy.utils.file_manager import filename2info
from lidarpy.utils.utils import LIDAR_INFO, signal_to_rcs
from lidarpy.preprocessing.lidar_preprocessing import preprocess
from lidarpy.plot.utils import get_norm, BoundsType


def quicklook_xarray(
    data_array: xr.DataArray,
    lidar_name: str,
    location: str = "Granada",
    is_rcs: bool = True,
    scale_bounds: BoundsType = "auto",
    colormap: str | colors.Colormap = "jet",
    ylims: Tuple[float, float] = (0.0, 14000.0),
    xlims: Tuple[datetime, datetime] | None = None,
) -> Tuple[Figure, Axes]:
    """Plot a quicklook of a lidar signal.

    Args:

        - data_array (xr.DataArray): Lidar signal or rcs from `lidar.preprocess`
        - is_rcs (bool, optional): To indicate if the data is RCS or not. Defaults to True.
        - scale_bounds (BoundsType, optional): scale bounds for the colorbar. Defaults to "auto".
        - colormap (str | matplotlib.colors.Colormap, optional): colormap of the colorbar. Defaults to "jet".

    Returns:

        - tuple[Figure, Axes]: Figure and Axes objects
    """

    # Convert cmap str to matplotlib colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    cmap = colormap.copy()
    cmap.set_extremes(over="white")

    # Define channel
    channel = data_array.name.split("_")[1]

    try:
        if is_rcs:
            rcs = data_array.values
        else:
            rcs = signal_to_rcs(data_array.values, data_array.range.values)

        fig, ax = plt.subplots(figsize=(15, 5))
        norm, bounds = get_norm(
            rcs, scale_bounds, lidar_name=lidar_name, channel=channel
        )

        q = ax.pcolormesh(
            data_array.time,
            data_array.range / 1000.0,
            rcs.T,
            cmap=cmap,
            norm=norm,
        )

        ax.set_xlabel(r"Time, $[UTC]$")
        ax.set_ylabel(r"Height, $[km, \, agl]$")
        ax.set_ylim(ylims[0] / 1000.0, ylims[1] / 1000.0)

        # Set xlims as in the original xarray.Dataset dims (time)
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1]) #type: ignore
        else:
            ax.set_xlim(data_array.time[0].values, data_array.time[-1].values)

        plot.watermark(ax, zoom=0.6, alpha=0.6)

        datestr = datetime.strftime(
            parse_datetime(data_array.time[1].values), "%Y-%m-%d"
        )
        ax.set_title(f"{location} | {lidar_name.upper()} | {datestr}")

        plot.apply_gap_size(ax, data_array=data_array)

        # Agregar la magnitud y unidades a la barra de color
        magnitud = f"RCS@{channel[:-3]}nm"
        unidades = "a.u."
        ticks = np.linspace(bounds.min(), bounds.max(), 9)
        colorbar = fig.colorbar(q, ticks=ticks, extend="max", pad=0.02)
        # fontsize colorbar
        colorbar.ax.tick_params(labelsize=12)

        # Crear el texto de magnitud y unidades
        text = f"{magnitud} [{unidades}]"

        # Calcular la mitad de la altura de la barra de color
        mid_height = (colorbar.ax.get_ylim()[0] + colorbar.ax.get_ylim()[1]) / 2

        # Añadir el texto a la barra de color girado 90 grados y 3 cm a la derecha de los valores
        x_position = 1.2  # Cambiar la coordenada x según sea necesario
        y_position = mid_height
        x_offset = 4  # 3 cm a la derecha (puede ajustarse)
        colorbar.ax.text(
            x_position + x_offset,
            y_position,
            text,
            rotation=90,
            va="center",
            ha="left",
            fontsize=12,
        )

        q.cmap.set_over("white")  # type: ignore
    except Exception as e:
        raise e
    return fig, ax


def quicklook_dataset(
    dataset: xr.Dataset,
    channel: str,
    /,
    scale_bounds: BoundsType = "auto",
    colormap: str | colors.Colormap = "jet",
    ylims: Tuple[float, float] = (0.0, 14000.0),
) -> Tuple[Figure, Axes]:
    """Plot a quicklook of a lidar signal.

    Args:

        - data_array (xr.DataArray): Lidar signal or rcs from `lidar.preprocess`
        - is_rcs (bool, optional): To indicate if the data is RCS or not. Defaults to True.
        - scale_bounds (BoundsType, optional): scale bounds for the colorbar. Defaults to "auto".
        - colormap (str | matplotlib.colors.Colormap, optional): colormap of the colorbar. Defaults to "jet".

    Returns:

        - tuple[Figure, Axes]: Figure and Axes objects
    """

    # Convert cmap str to matplotlib colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    cmap = colormap.copy()
    cmap.set_extremes(over="white")

    data_array = dataset[f"signal_{channel}"]

    # Get lidar name from dataset
    lidar_name = dataset.attrs["location"].lower()

    quicklook_xarray(
        data_array,
        lidar_name=lidar_name,
        location="Granada",
        is_rcs=False,
        scale_bounds=scale_bounds,
        colormap=colormap,
        ylims=ylims,
    )

    try:
        rcs = signal_to_rcs(data_array.values, data_array.range.values)

        fig, ax = plt.subplots(figsize=(15, 5))
        norm, bounds = get_norm(
            rcs, scale_bounds, lidar_name=lidar_name, channel=channel
        )

        q = ax.pcolormesh(
            data_array.time,
            data_array.range / 1000.0,
            rcs.T,
            cmap=cmap,
            norm=norm,
        )

        ax.set_xlabel(r"Time, $[UTC]$")
        ax.set_ylabel(r"Height, $[km, \, agl]$")
        ax.set_ylim(ylims[0] / 1000.0, ylims[1] / 1000.0)

        datestr = datetime.strftime(parse_datetime(data_array.time[1].values), "%Y%m%d")
        plot.title1(f"{data_array.name} | {datestr}", 2)

        plot.watermark(ax, zoom=0.6, alpha=0.6)
        plot.apply_gap_size(ax, data_array=data_array)

        # Agregar la magnitud y unidades a la barra de color
        magnitud = "RCS"
        unidades = "a.u."
        ticks = np.linspace(bounds.min(), bounds.max(), 9)
        colorbar = fig.colorbar(q, ticks=ticks, extend="max")

        # Crear el texto de magnitud y unidades
        text = f"{magnitud} [{unidades}]"

        # Calcular la mitad de la altura de la barra de color
        mid_height = (colorbar.ax.get_ylim()[0] + colorbar.ax.get_ylim()[1]) / 2

        # Añadir el texto a la barra de color girado 90 grados y 3 cm a la derecha de los valores
        x_position = 1.2  # Cambiar la coordenada x según sea necesario
        y_position = mid_height
        x_offset = 4  # 3 cm a la derecha (puede ajustarse)
        colorbar.ax.text(
            x_position + x_offset,
            y_position,
            text,
            rotation=90,
            va="center",
            ha="left",
            fontsize=12,
        )

        q.cmap.set_over("white")  # type: ignore
    except Exception as e:
        raise e
    return fig, ax


def quicklook_from_file(
    filepath: Path | str,
    channels: List[str],
    output_dir: Path | str | None = None,
    scale_bounds: BoundsType = "auto",
    **kwargs,
):
    # Define output directory
    if output_dir is None:
        output_dir = Path.cwd()
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
    else:
        if not output_dir.exists():
            raise ValueError(f"Output directory {output_dir} does not exist.")

    # Check filepath
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    # Get lidar name from filepath
    nickname, _, _, _, _, target_date = filename2info(filepath.name)
    lidar_name = LIDAR_INFO["metadata"]["nick2name"][nickname]

    # Get INFO
    INFO = read_yaml_from_info(nickname, target_date)

    if kwargs.get("apply_ov", False):
        channels2load = channels.copy()
        for channel_ in channels2load:
            if "f" in channel_:
                overlap_channel_ = INFO["overlap_channels"][channel_]
                if overlap_channel_ not in channels2load:
                    channels2load.append(overlap_channel_)
    else:
        channels2load = None

    lidar = preprocess(
        filepath,
        channels=channels2load,
        apply_bg=kwargs.get("apply_bg", True),
        apply_dc=kwargs.get("apply_dc", False),
        apply_dt=kwargs.get("apply_dt", False),
        apply_bz=kwargs.get("apply_bz", True),
        apply_ov=kwargs.get("apply_ov", False),
    )

    if "ghk_dir" in kwargs and "depo_calib_dir" in kwargs:
        lidar.add_depolarization_products(
            lidar, ghk_dir=kwargs["ghk_dir"], depo_calib_dir=kwargs["depo_calib_dir"]
        )

    # Get location from dataset
    location = lidar.attrs["location"]

    # Get date string
    datestr = datetime.strftime(parse_datetime(lidar.time[1].values), "%Y%m%d_%H%M")

    for channel_ in channels:
        fig, _ = quicklook_xarray(
            lidar[f"signal_{channel_}"],
            lidar_name=lidar_name,
            location=location,
            is_rcs=False,
            scale_bounds=scale_bounds,
        )
        fig.savefig(
            output_dir / f"quicklook_{nickname}_{channel_}_{datestr}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)
    lidar.close()


def quicklook_from_date(
    lidar_name: str,
    channels: List[str],
    target_date: date | str,
    product_dir: Path | str,
    output_dir: Path | str | None = None,
    scale_bounds: BoundsType = "auto",
    **kwargs,
) -> List[Path]:

    if isinstance(target_date, str):
        target_date = parse_datetime(target_date).date()

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif output_dir is None:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create xlim for target date from 0h to 0h of the next day
    xlims = (
        parse_datetime(target_date),
        parse_datetime(target_date + timedelta(days=1)),
    )

    lidar = merge_measurements_by_date(
        measurement_type=MeasurementType("RS"),
        lidar_name=lidar_name,
        target_date=target_date,
        product_dir=product_dir,
        channels=channels,
    )

    nickname = LIDAR_INFO["metadata"]["name2nick"][lidar_name]
    datestr = target_date.strftime("%Y%m%d")
    quicklook_paths = []
    for channel_ in channels:
        if channel_ not in lidar.channel.values:
            continue
        fig, _ = quicklook_xarray(
            lidar[f"signal_{channel_}"],
            lidar_name=lidar_name,
            is_rcs=False,
            scale_bounds=scale_bounds,
            ylims=kwargs.get("ylims", (0.0, 14000.0)),
            xlims=xlims,
        )
        quicklook_path = output_dir / f"quicklook_{nickname}_{channel_}_{datestr}.png"
        fig.savefig(
            quicklook_path,
            dpi=kwargs.get("dpi", 300),
            bbox_inches="tight",
        )
        plt.close(fig)
        quicklook_paths.append(quicklook_path)
    lidar.close()
    return quicklook_paths

def quicklook_lpdr(
    lpdr: xr.DataArray,    
    scale_bounds: BoundsType = (0, 0.5),
    colormap: str | colors.Colormap = "jet",
    ylims: Tuple[float, float] = (0.0, 15000.0),
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot a quicklook of a lidar LPDR signal.

    Args:
        - lpdr (dict[int, xr.DataArray]): Dictionary containing LPDR data for different channels.
        - lidar_time (xr.DataArray): Time coordinate associated with the lidar data.
        - channel (int): Wavelength channel (e.g., 532 or 355).
        - scale_bounds (BoundsType, optional): Scale bounds for the colorbar. Defaults to (0, 0.5).
        - colormap (str | matplotlib.colors.Colormap, optional): Colormap for the colorbar. Defaults to "jet".
        - ylims (Tuple[float, float], optional): Y-axis limits in meters. Defaults to (0.0, 14000.0).

    Returns:
        - tuple[Figure, Axes]: Figure and Axes objects.
    """

    # Convert cmap str to matplotlib colormap
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    
    cmap = colormap.copy() #tye: ignore
    cmap.set_extremes(over="white")

    fig, ax = plt.subplots(figsize=(15, 5))
    norm, _ = get_norm(lpdr.values, scale_bounds)

    q = ax.pcolormesh(
        lpdr.time,  # Time coordinate
        lpdr.range / 1000.0,
        lpdr.values.T,  # Ensure correct shape for pcolormesh
        cmap=cmap,
        norm=norm,
        shading="auto"  # Allow automatic shading correction
    )

    ax.set_xlabel(r"Time, $[UTC]$")
    ax.set_ylabel(r"Range, $[km]$")
    ax.set_ylim(ylims[0] / 1000.0, ylims[1] / 1000.0)

    ticks = np.linspace(0, 0.5, 9)
    colorbar = fig.colorbar(q, ticks=ticks, extend="max")
    
    mid_height = (colorbar.ax.get_ylim()[0] + colorbar.ax.get_ylim()[1]) / 2
    x_position = 1.2  # Adjust as needed
    x_offset = 4
    colorbar.ax.text(
        x_position + x_offset, mid_height, "LPDR, [#]", rotation=90, va="center", ha="left", fontsize=12
    )

    q.cmap.set_over("white")  # type: ignore
    
    return fig, ax
