import os
import sys
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from lidarpy.scc.utils import get_scc_config
from lidarpy.utils.utils import signal_to_rcs
from lidarpy.plot import PLOT_INFO
from lidarpy.general_utils.io import find_nearest_filepath
from lidarpy.general_utils.plot import color_list

mpl.rcParams['figure.max_open_warning'] = 100 

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class SCC_product(ABC):
    """A simple class to manage generic SCC products."""

    def __init__(self, path: Path):
        self.path = self.check_path(path)

    def check_path(self, path: Path) -> Path:
        """Check if the path exists and is a netCDF file.

        Args:
            path (Path): Path to check.

        Raises:
            ValueError: path does not exist.
            ValueError: Filepath is not a netCDF file.

        Returns:
            Path: Path to the netCDF file.
        """
        if not path.exists():
            raise ValueError(f"{path} does not exist.")
        # check zip
        if path.suffix != ".nc":
            raise ValueError("Filepath is not a netCDF file.")
        return path

    def open_dataset(self) -> xr.Dataset:
        """Open the netCDF file.

        Returns:
            xr.Dataset: NetCDF file.
        """
        data = xr.open_dataset(self.path)
        return data


class SCC_raw(SCC_product):
    """A simple class to manage SCC raw products based on SCC_product."""

    def __init__(self, path: Path, range_resolution: float = 3.75):
        super().__init__(path)
        (
            self.station,
            self.datetime,
            self.measurementID,
            self.type,
        ) = self._get_info(self.path)
        self.range_resolution = range_resolution
        self.channels_ID = self._get_channel_ID()

    def _get_info(self, path: Path) -> tuple[str, datetime, str, str]:
        """Get information from SCC raw netcdf files.

        Args:
            path (Path): Path to the SCC raw netcdf file.

        Returns:
            tuple[str, datetime, int]: Information from the SCC raw netcdf file. In order: station, datetime, measurementID, scc_product_type.
        """
        # raw: raw_202307060159_202307060259_20230706gra0200_elpp_v5.2.7.nc
        measurementID = path.name.split("_")[-1].split(".")[0]
        ini_date, station, ini_hour = (
            measurementID[:8],
            measurementID[8:11],
            measurementID[11:],
        )
        ini_datetime = datetime.strptime(f"{ini_date}{ini_hour}", "%Y%m%d%H%M")
        return (
            station,
            ini_datetime,
            measurementID,
            "raw",
        )

    def _get_channel_ID(self) -> np.ndarray:
        """Get the channel ID from the netCDF file.

        Returns:
            np.ndarray: Channel ID.
        """
        with xr.open_dataset(self.path) as data:
            return data.channel_ID.values

    def channel2str(self, ID: int) -> str:
        """Provides the string of the channel/product ID.

        Args:
            ID (int): ID channel/product

        Returns:
            str: string describing the channel/product.
        """
        strid = str(ID)
        if strid is None:
            raise ValueError(f"ID {ID} not found in the configuration.")
        return strid

    def _subtrack_dark_measurement(
        self, data: xr.Dataset, fill_value=9.969209968386869e36
    ) -> xr.Dataset:
        """It substract the dark measurment `Background_profile`.

        Args:
            data (xr.Dataset): data from self.data
            fill_value (_type_, optional): Fill value used in the netcdf. Defaults to EARLINET fill value: 9.969209968386869e+36.

        Returns:
            xr.Dataset: same Dataset but including `Raw_Lidar_Data_dc`.
        """
        if "Background_Profile" not in data:
            logger.warning(
                "Background_Profile not found in data. Dark measurement subtraction will not be performed."
            )
            return data
        channel_IDs = [
            channel_
            for channel_ in self.channels_ID
            if data["DAQ_Range"].sel(channels=channel_).values != fill_value
        ]
        data["Raw_Lidar_Data_dc"] = data["Raw_Lidar_Data"].copy()

        for channel_ in channel_IDs:
            # find the index of the channel channel_ID in data['channels']
            idx_ = np.where(data["channels"].values == channel_)[0][0]
            data["Raw_Lidar_Data_dc"][:, idx_, :] = data["Raw_Lidar_Data"].sel(
                channels=channel_
            ) - data["Background_Profile"].sel(channels=channel_).mean("time_bck")
        return data

    def _subtract_background(self, data: xr.Dataset) -> xr.Dataset:
        """It substract the background according to `Background_Mode`.

        Args:
            data (xr.Dataset): data from self.data.

        Returns:
            xr.Dataset: Same data including variable `Raw_Lidar_Data_bckg` or `Raw_Lidar_Data_dc__bckg`.
        """
        if "Raw_Lidar_Data_dc" not in data:
            raw_string = "Raw_Lidar_Data"
        else:
            raw_string = "Raw_Lidar_Data_dc"
        data[f"{raw_string}_bckg"] = data[raw_string].copy(
            data=np.nan * np.ones(data[raw_string].shape)
        )

        for channel_idx, channel_ in enumerate(self.channels_ID):
            if data["Background_Mode"].sel(channels=channel_).item() == 0:
                low_background = int(
                    data["Background_Low"].sel(channels=channel_).item()
                )
                high_background = int(
                    data["Background_High"].sel(channels=channel_).item()
                )
            else:
                low_background = int(
                    data["Background_Low"].sel(channels=channel_).item()
                    / self.range_resolution
                )
                high_background = int(
                    data["Background_High"].sel(channels=channel_).item()
                    / self.range_resolution
                )
            bckgrd_ = (
                data[raw_string]
                .sel(channels=channel_, points=slice(low_background, high_background))
                .mean("points")
            )
            data[f"{raw_string}_bckg"][:, channel_idx, :] = (
                data[raw_string].sel(channels=channel_) - bckgrd_
            )

        return data

    def _zerobin_correction(self, data: xr.Dataset) -> xr.Dataset:
        """It corrects from zero bin using `First_Signal_Rangebin`.

        Args:
            data (xr.Dataset): data from self.data

        Raises:
            ValueError: Background subtraction must be performed before zero bin correction.

        Returns:
            xr.Dataset: same data including `Raw_Lidar_Data_dc_bckg_zb` or `Raw_Lidar_Data_bckg_zb`.
        """
        raw_string = "Raw_Lidar_Data_dc_bckg"
        if raw_string not in data:
            raw_string = "Raw_Lidar_Data_bckg"
        if raw_string not in data:
            raise ValueError(
                "Background subtraction must be performed before zero bin correction."
            )

        shifted_data = np.nan * np.ones(data[raw_string].shape)
        topbin = data[raw_string].shape[-1]
        for channel_idx, channel_ in enumerate(data.channels.values):
            zerobin_ = data["First_Signal_Rangebin"].sel(channels=channel_).item()
            shifted_data[:, channel_idx, : (topbin - zerobin_)] = data[raw_string].sel(
                channels=channel_, points=slice(zerobin_, topbin)
            )

        # add shifted data to dataset as Raw_Lidar_Data_dc_bckg_zb
        data[f"{raw_string}_zb"] = (("time", "channels", "points"), shifted_data)
        return data

    def scc_preprocessing(self) -> xr.Dataset:
        """It preprocesses the raw data: 1) Dark measurement, 2) Background, 3) Zero bin. Background correction is mandatory.

        Returns:
            xr.Dataset: same data including 'Raw_Lidar_Data_dc_bckg_zb' or 'Raw_Lidar_Data_bckg_zb'
        """
        data = self.open_dataset()
        data["channels"] = data["channel_ID"].values
        data = self._subtrack_dark_measurement(data=data)
        data = self._subtract_background(data=data)
        data = self._zerobin_correction(data=data)
        return data

    def plot(
        self,
        rcs_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
        output_dir: Path | None = None,
        dpi: int = 300,
        savefig=False,
        **kwargs,
    ) -> tuple[list[Figure], list[Path | None]]:

        if kwargs.get("preprocess", True):
            data = self.scc_preprocessing()
        else:
            data = self.open_dataset()
        data["points"] = (
            self.range_resolution * np.arange(1, data.sizes["points"] + 1) / 1e3
        )
        figs, outputpaths = [], []
        for channel_ in data.channels.values:
            DAQ = data["DAQ_Range"].sel(channels=channel_).values.item()  # mV
            fill_value = 9.969209968386869e36
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            if isinstance(ax, Axes):
                ax = [ax]
            string_ID = self.channel2str(channel_)
            wavelength = int(string_ID.split("_")[0][:-3])
            
            colors = color_list(
                data.sizes["time"], cmap='jet'
            )
            raw_colors = color_list(data.sizes["time"], cmap="Greys")
            for time_idx, time_ in enumerate(
                data["Raw_Data_Start_Time"].values.flatten()
            ):
                try:
                    if "Raw_Lidar_Data_dc_bckg_zb" in data:
                        signal_to_rcs(
                            data.sel(time=time_idx, channels=channel_)[
                                "Raw_Lidar_Data_dc_bckg_zb"
                            ],
                            data["points"] * 1e3,
                        ).plot(
                            x="points", ax=ax[0], color=colors[time_idx], linewidth=0.25
                        )
                        data.sel(time=time_idx, channels=channel_)[
                            "Raw_Lidar_Data_dc_bckg_zb"
                        ].plot(
                            x="points", ax=ax[1], color=colors[time_idx], linewidth=0.25
                        )
                        data.sel(time=time_idx, channels=channel_)[
                            "Raw_Lidar_Data"
                        ].plot(
                            x="points",
                            ax=ax[1],
                            color=raw_colors[time_idx],
                            linewidth=0.25,
                        )
                    elif "Raw_Lidar_Data_bckg_zb" in data:
                        signal_to_rcs(
                            data.sel(time=time_idx, channels=channel_)[
                                "Raw_Lidar_Data_bckg_zb"
                            ],
                            data["points"] * 1e3,
                        ).plot(
                            x="points", ax=ax[0], color=colors[time_idx], linewidth=0.25
                        )
                        data.sel(time=time_idx, channels=channel_)[
                            "Raw_Lidar_Data_bckg_zb"
                        ].plot(
                            x="points", ax=ax[1], color=colors[time_idx], linewidth=0.25
                        )
                        data.sel(time=time_idx, channels=channel_)[
                            "Raw_Lidar_Data"
                        ].plot(
                            x="points",
                            ax=ax[1],
                            color=raw_colors[time_idx],
                            linewidth=0.25,
                        )
                    else:
                        data.sel(time=time_idx, channels=channel_)[
                            "Raw_Lidar_Data"
                        ].plot(
                            x="points",
                            ax=ax[1],
                            color=raw_colors[time_idx],
                            linewidth=0.25,
                        )
                except:
                    raise ValueError(
                        f"Error plotting {self.path.name} | {string_ID} | {time_}"
                    )
            fig.suptitle(
                f"{self.station} | {string_ID} | {channel_} | {self.measurementID}"
            )
            ax[0].set_ylabel("Uncal. RCS, [a.u.]")
            if rcs_limits is not None:
                ax[0].set_ylim(rcs_limits)
            if data["DAQ_Range"].sel(channels=channel_).values != fill_value:
                if kwargs.get("analog_limits", None) is not None:
                    ax[1].set_ylim(kwargs.get("analog_limits"))
                ax[1].set_ylabel("Signal, [mV]")
                ax[1].axhline(y=DAQ / 20000, color="r", linestyle="--", linewidth=1)
            else:
                if kwargs.get("photoncounting_limits", None) is not None:
                    ax[1].set_ylim(kwargs.get("photoncounting_limits"))
                ax[1].axhline(y=20, color="b", linestyle="--", linewidth=1)
                ax[1].set_ylabel("Signal, [Counts]")
            if range_limits is not None:
                ax[0].set_xlim(range_limits)
            else:
                ax[0].set_xlim(0, 15)
            # Remove xlabel from ax[0]
            ax[0].set_xlabel("")
            ax[1].set_xlabel("Range, [km]")
            ax[0].set_yscale("log")
            for ax_ in ax:
                ax_.minorticks_on()
                ax_.set_title("")
                ax_.grid()
                # ax_.legend().remove()
            try:
                fig.tight_layout()
            except Exception as e:
                print(e)
                logger.warning(
                    f"Error in tight_layout for {self.path.name} | {string_ID}"
                )
                continue
            if savefig:
                # save
                if isinstance(output_dir, Path):
                    output_dir.mkdir(parents=True, exist_ok=True)
                elif output_dir is None:
                    output_dir = Path.cwd()
                outputpath = output_dir.joinpath(
                    self.path.name.replace(".nc", f"_{string_ID}.png")
                )
                fig.savefig(outputpath, dpi=dpi)                
                outputpaths.append(outputpath)
            figs.append(fig)
        data.close()
        return figs, outputpaths


class SCC_elpp(SCC_product):
    """A simple class to manage SCC elpp products based on SCC_product."""

    def __init__(self, path: Path):
        super().__init__(path)
        (
            self.station,
            self.number1,
            self.productID,
            self.datetime_ini,
            self.datetime_end,
            self.measurementID,
            self.type,
            self.scc_version,
            self.hoi_configuration_ID,
        ) = self._get_info(self.path)
        self.string_ID = str(self.productID)

    def _get_info(
        self, path: Path
    ) -> tuple[str, int, int, datetime, datetime, str, str, str, int]:
        """Get information from SCC ELPP netcdf files.

        Args:
            path (Path): Path to the SCC ELPP netcdf file.

        Returns:
            tuple[str, int, int, datetime, datetime, str, str, str, int]: Information from the SCC ELPP netcdf file. In order: station, number1, productID, ini_datetime, end_datetime, measurementID, scc_product_type, scc_version_str, hoi_configuration_ID.
        """
        # elpp: gra_003_0002223_202307060159_202307060259_20230706gra0200_elpp_v5.2.7.nc
        (
            station,
            number1,
            productID,
            ini_datestr,
            end_datestr,
            measurementID,
            scc_product_type,
            scc_version_str,
        ) = path.name.split("_")
        ini_datetime = datetime.strptime(ini_datestr, "%Y%m%d%H%M")
        end_datetime = datetime.strptime(end_datestr, "%Y%m%d%H%M")
        with xr.open_dataset(self.path) as ds:
            hoi_configuration_ID = ds.attrs["hoi_configuration_ID"]

        return (
            station,
            int(number1),
            int(productID),
            ini_datetime,
            end_datetime,
            measurementID,
            scc_product_type,
            scc_version_str[:-3],
            hoi_configuration_ID,
        )

    def get_id_to_channel(self, scc_id, scc_config_dir= Path(__file__).parent.parent.absolute() / 'scc_configFiles') -> dict:
        
        scc_config_fn = find_nearest_filepath(
            scc_config_dir,
            f"alh_parameters_scc_{scc_id}_*.py",
            4,
            self.datetime_ini,
            and_previous=True,
        )
        scc_config_dict = get_scc_config(scc_config_fn)

        id_to_channel = {params["channel_ID"]: params["channel_string_ID"] for _, params in scc_config_dict['channel_parameters'].items()}
        return id_to_channel

    def rayleight_fit(
        self,
        scc_id: int,
        normalization_range: tuple[float, float] = (6, 7),
        range_limits: tuple[float, float] = (0, 25),
        attenuated_backscatter_limits: tuple[float, float] | None = None,
        output_dir: Path | None = None,
        dpi: int = 300,
        combine_channels: bool = False,
        savefig: bool = False,
    ) -> tuple[Figure, Path | None]:
        """plot Rayleigh fit from SCC ELPP netcdf files.

        Args:
            normalization_range (tuple[float, float], optional): Minimum and maximum ranges to normalized signals. Defaults to (6, 7).
            range_limits (tuple[float, float], optional): Range limits. Defaults means (0, 25).
            attenuated_backscatter_limits (tuple[float, float] | None, optional): Attenuated backscatter limits. Defaults to None means automatic limits from `lidarpy.plot.PLOT_INFO`.
            savefig (bool, optional): If True save figure. Defaults to False.
            output_dir (Path | None, optional): Output directory to save figure. Defaults to None means current working directory.
            dpi (int, optional): _description_. Defaults to 300.
            combine_channels (bool, optional): Combine in the same figure all the signals linked to a givel ELPP product. Defaults to False.

        Returns:
            list[Figure]: List of figures.
        """

        station, hoi_configuration_ID, string_ID = (
            self.station,
            self.hoi_configuration_ID,
            self.string_ID,
        )
        raw_data = self.open_dataset()
        raw_data["altitude"] = raw_data["altitude"] / 1e3
        attenuated_molecular_backscatter = (
            (raw_data.molecular_extinction / raw_data.molecular_lidar_ratio)
            * raw_data.molecular_transmissivity_at_detection_wavelength
            * raw_data.molecular_transmissivity_at_emission_wavelength
        )
        if combine_channels:
            fig, ax = plt.subplots(figsize=(10, 4))
        for idx, channel_ in enumerate(raw_data.channel):
            if not combine_channels:
                fig, ax = plt.subplots(figsize=(8, 4))
            IDs = raw_data.range_corrected_signal_channel_id.sel(
                channel=channel_
            ).values.flatten()
            channels = [self.get_id_to_channel(scc_id=scc_id).get(int(ID_)) for ID_ in IDs]            
            comb_channel = os.path.commonprefix(channels) #take common part of the channel. e.g. from 532fta and 532ftp provides 532ft
            wavelength = int(channels[0][:-3])
            if len(raw_data.time) > 1:
                colors = color_list(
                    raw_data.sizes["time"],
                    cmap=PLOT_INFO["wavelength2cmaps"][wavelength],
                )
            else:
                if string_ID is not None and "b" in string_ID:
                    colors = [
                        tuple(
                            np.asarray(
                                list(
                                    mcolors.to_rgba(
                                        PLOT_INFO["backscatter2color"][self.string_ID]
                                    )
                                )
                            )[:3]
                            / (idx + 1)
                        )
                        + tuple([1])
                    ]
                elif string_ID is not None and ("lre" in string_ID or "eo" in string_ID):
                    colors = [
                        tuple(
                            np.asarray(
                                list(
                                    mcolors.to_rgba(
                                        PLOT_INFO["extinction2color"][self.string_ID]
                                    )
                                )
                            )[:3]
                            / (idx + 1)
                        )
                        + tuple([1])
                    ]
                else:
                    colors = ["orange"]
            for idx, time_ in enumerate(raw_data.time.values):
                range_resol = (
                    raw_data.altitude.sel(time=time_)[1]
                    - raw_data.altitude.sel(time=time_)[0]
                )
                normalization_index = [
                    int(i / range_resol) for i in normalization_range
                ]
                norm_attbetamol = attenuated_molecular_backscatter.sel(
                    level=slice(*normalization_index)
                ).mean("level")
                norm_rcs = (
                    raw_data["range_corrected_signal"]
                    .sel(level=slice(*normalization_index))
                    .mean("level")
                )
                normalized_rcs = raw_data["range_corrected_signal"] * (
                    norm_attbetamol / norm_rcs
                )
                (1e6 * normalized_rcs.sel(time=time_, channel=channel_)).plot(
                    x="altitude",
                    ax=ax,
                    label=r"$\beta_{RCS}^{att}$"
                    + f"[{comb_channel}] | {time_.astype(str).split('.')[0]}",
                    color=colors[idx],
                    linewidth=1.0,
                )
                (
                    1e6
                    * attenuated_molecular_backscatter.sel(time=time_, channel=channel_)
                ).plot(
                    x="altitude",
                    ax=ax,
                    label=r"$\beta_{mol}^{att}$"
                    + f"[{wavelength} nm] | {time_.astype(str).split('.')[0]}",
                    color="orange",
                    linewidth=1.0,
                )
            ax.set_xlabel("Range, [km]")
            ax.set_ylabel("Att. backscatter, [Mm$^{-1}$sr$^{-1}$]")
            ax.set_title(
                f"{station} | Lidar ID: {hoi_configuration_ID} | Product ID/name: {self.productID}/{string_ID} \n Norm. range: {normalization_range} km"
            )
            ax.set_yscale("log")
            ax.set_xlim(*range_limits)
            if attenuated_backscatter_limits is not None:
                ax.set_ylim(*attenuated_backscatter_limits)
            else:
                ax.set_ylim(*PLOT_INFO["limits"]["attenuated_backscatter"][wavelength])
            ax.legend(fontsize=10, loc="upper right")
            ax.minorticks_on()
            ax.grid()
            ax.set_facecolor("white")
            fig.tight_layout()
            if savefig:
                if output_dir is None:
                    output_dir = Path.cwd()
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True)
                outputpath = output_dir / self.path.name.replace("nc", "png")
                fig.savefig(
                    outputpath,
                    dpi=dpi,
                    bbox_inches="tight",
                )
            else:
                outputpath = None
        raw_data.close()
        return fig, outputpath

    def plot(
        self,
        rcs_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
        output_dir: Path | None = None,
        dpi: int = 300,
        savefig=False,
    ) -> tuple[Figure, Path | None]:
        """Plot SCC ELPP netcdf files.

        Args:
            rcs_limits (tuple[float, float] | None, optional): Range Corrected Signal limits. Defaults to None. None means automatic limits.
            range_limits (tuple[float, float] | None, optional): Range limits. Defaults to None. None means automatic limits.
            output_dir (Path | None, optional): Output directory. Defaults to None. None means current working directory.
            dpi (int, optional): Plot resolution. Defaults to 300.

        Raises:
            ValueError: Filepath does not exist.
            ValueError: Error plotting.

        Returns:
            Path: Path to the plot.
        """

        if not Path(self.path).exists():
            raise ValueError("Filepath does not exist.")

        # Load netcdf
        elpp = xr.open_dataset(self.path)
        elpp["altitude"] = elpp["altitude"] / 1e3
        hoi_configuration_ID = elpp.attrs["hoi_configuration_ID"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        for product_ in elpp.channel.values:
            IDs = elpp.range_corrected_signal_channel_id.sel(
                channel=product_
            ).values.flatten()
            string_list = [str(int(ID)) for ID in IDs]
            comb_channel = os.path.commonprefix(string_list)
            wavelength = int(string_list[0].split("_")[0][:-3])
            colors = color_list(
                elpp.sizes["time"], cmap=PLOT_INFO["wavelength2cmaps"][wavelength]
            )
            for idx, time_ in enumerate(elpp.time.values):
                try:
                    elpp.sel(time=time_, channel=product_)[
                        "range_corrected_signal"
                    ].plot(
                        x="altitude",
                        ax=ax,
                        label=f"{comb_channel} | {time_.astype(str).split('.')[0]}",
                        color=colors[idx],
                        linewidth=1.0,
                    )
                except:
                    raise ValueError(
                        f"Error plotting {self.path.name} | {product_} | {time_}"
                    )
        ax.set_title(f"{self.station} | {hoi_configuration_ID} | {self.string_ID}")
        if rcs_limits is not None:
            ax.set_ylim(rcs_limits)
        if range_limits is not None:
            ax.set_xlim(range_limits)
        else:
            ax.set_xlim(0, 15)
        ax.set_yscale("log")
        ax.legend(fontsize=10, loc="upper right")
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel("Range, [km]")
        ax.set_ylabel("Uncal. RCS, [a.u.]")
        fig.tight_layout()
        if savefig:
            # save
            if isinstance(output_dir, Path):
                output_dir.mkdir(parents=True, exist_ok=True)
            elif output_dir is None:
                output_dir = Path.cwd()
            outputpath = output_dir.joinpath(self.path.name.replace(".nc", ".png"))
            fig.savefig(outputpath, dpi=dpi)
            elpp.close()
            plt.close(fig)
        else:
            outputpath = None
        return fig, outputpath


class SCC_elda(SCC_product):
    """A simple class to manage SCC elda products based on SCC_product."""

    def __init__(self, path: Path):
        super().__init__(path)
        (
            self.station,
            self.number1,
            self.wavelength,
            self.productID,
            self.datetime_ini,
            self.datetime_end,
            self.measurementID,
            self.type,
            self.scc_version,
            self.hoi_configuration_ID,
        ) = self._get_info(self.path)
        self.string_ID = str(self.productID)
        self.earlinet_product_type = self._earlinet_product_type()

    def _get_info(
        self, path: Path
    ) -> tuple[str, int, int, int, datetime, datetime, str, str, str, int]:
        """Get information from SCC ELDA netcdf files.

        Args:
            path (Path): Path to the SCC ELDA netcdf file.

        Returns:
            tuple[str, int, int, int, datetime, datetime, str, str, str, int]: Information from the SCC ELDA netcdf file. In order: station, number1, wavelength, productID, ini_datetime, end_datetime, measurementID, scc_product_type, scc_version_str, hoi_configuration_ID.
        """
        # elda: gra_001_0532_0002366_202302220029_202302220059_20230222gra0000_elda_v5.2.7
        (
            station,
            number1,
            wavelength,
            productID,
            ini_datestr,
            end_datestr,
            measurementID,
            scc_product_type,
            scc_version_str,
        ) = path.name.split("_")
        ini_datetime = datetime.strptime(ini_datestr, "%Y%m%d%H%M")
        end_datetime = datetime.strptime(end_datestr, "%Y%m%d%H%M")
        with xr.open_dataset(self.path) as ds:
            hoi_configuration_ID = ds.attrs["hoi_configuration_ID"]

        return (
            station,
            int(number1),
            int(wavelength),
            int(productID),
            ini_datetime,
            end_datetime,
            measurementID,
            scc_product_type,
            scc_version_str[:-3],
            hoi_configuration_ID,
        )

    def _earlinet_product_type(self) -> dict:
        """Return the EARLINET product flags.

        Args:
            elda_product (SCC_elda): SCC elda object.

        Returns:
            dict: Dictionary with numbers [int] as keys and its meaning [string] as value.
        """
        with xr.open_dataset(self.path) as elda:
            flag_dict = dict(
                zip(
                    elda.earlinet_product_type.flag_values,
                    elda.earlinet_product_type.flag_meanings.split(" "),
                )
            )

        return flag_dict[elda.earlinet_product_type.values.item()]

    def plot_backscatter(
        self,
        ax: Axes,
        backscatter_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot backscatter from SCC elda netcdf files.

        Args:
            ax (Axes): Axes to plot.
            backscatter_limits (tuple[float, float] | None, optional): Backcatter coefficient limits. Defaults to None. None means automatic limits.
            range_limits (tuple[float, float] | None, optional): Range limits. Defaults to None. None means automatic limits.

        Raises:
            ValueError: Error plotting.

        Returns:
            Axes: Axes with the plot.
        """

        # Load netcdf        
        elda = xr.open_dataset(self.path)
        elda["altitude"] = elda["altitude"] / 1e3
        for time_ in elda.time.values:            
            try:
                (elda.sel(time=time_)["backscatter"] * 1e6).plot(
                    y="altitude",
                    ax=ax,
                    label=self.string_ID,
                    color=PLOT_INFO["backscatter2color"][self.wavelength],
                    linewidth=1.0,
                )
            except:
                raise ValueError(f"Error plotting {self.path.name} | {time_}")
        ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
        if backscatter_limits is not None:
            ax.set_xlim(*backscatter_limits)
        if range_limits is not None:
            ax.set_xlim(*PLOT_INFO["limits"]["backscatter"])
        ax.set_xlabel(r"$\beta_{part}$, [$Mm^{-1}$]")
        elda.close()
        return ax

    def plot_extinction(
        self,
        ax: Axes,
        extinction_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot extinction from SCC elda netcdf files.

        Args:
            ax (Axes): Axes to plot.
            extinction_limits (tuple[float, float] | None, optional): Extinction limits. Defaults to None. None means automatic limits.
            range_limits (tuple[float, float] | None, optional): Range limits. Defaults to None. None means automatic limits.

        Raises:
            ValueError: Error plotting.

        Returns:
            Axes: Axes with the plot.
        """

        # Load netcdf
        elda = xr.open_dataset(self.path)
        elda["altitude"] = elda["altitude"] / 1e3
        color = PLOT_INFO["extinction2color"][self.wavelength]
        for time_ in elda.time.values:
            try:
                (elda.sel(time=time_)["extinction"] * 1e6).plot(
                    y="altitude",
                    ax=ax,
                    label=self.string_ID,
                    color=color,
                    linewidth=1.0,
                )
            except:
                raise ValueError(f"Error plotting {self.path.name} | {time_}")
        # plot vertical line at zero
        ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
        if extinction_limits is not None:
            ax.set_xlim(*extinction_limits)
        else:
            ax.set_ylim(*PLOT_INFO["limits"]["extinction"])
        if range_limits is not None:
            ax.set_ylim(range_limits)
        ax.set_xlabel(r"$\alpha_{part}$, [$Mm^{-1}$]")
        elda.close()
        return ax

    def plot_depo(
        self,
        ax: Axes,
        depo_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot depolarization from SCC elda netcdf files.

        Args:
            ax (Axes): Axes to plot.
            depo_limits (tuple[float, float] | None, optional): Depolarization limits. Defaults to None. . None means limits `from lidarpy.plot.info.yml`.
            range_limits (tuple[float, float] | None, optional): Range limits. Defaults to None. None means automatic limits.

        Raises:
            ValueError: Error plotting.

        Returns:
            Axes: Axes with the plot.
        """

        # Load netcdf
        elda = xr.open_dataset(self.path)
        elda["altitude"] = elda["altitude"] / 1e3
        color = PLOT_INFO["depolarization2color"][self.wavelength]
        for time_ in elda.time.values:
            try:
                elda.sel(time=time_)["particledepolarization"].plot(
                    y="altitude",
                    ax=ax,
                    label=self.string_ID,
                    color=color,
                    linewidth=2.5,
                )
                elda.sel(time=time_)["volumedepolarization"].plot(
                    y="altitude",
                    ax=ax,
                    label=self.string_ID,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                )
            except:
                raise ValueError(f"Error plotting {self.path.name} | {time_}")
        # plot vertical line at zero
        ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
        if depo_limits is not None:
            ax.set_xlim(*depo_limits)
        else:
            ax.set_xlim(*PLOT_INFO["limits"]["depolarization"])
        if range_limits is not None:
            ax.set_ylim(range_limits)
        ax.set_xlabel(r"$\delta_{part}$, [#]")
        elda.close()
        return ax

    def plot_lidar_ratio(
        self,
        ax: Axes,
        lidar_ratio_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot lidar ratio from SCC elda netcdf files.

        Args:
            ax (Axes): Axes to plot.
            lidar_ratio_limits (tuple[float, float] | None, optional): Lidar ratio limits. Defaults to None. None means limits `from lidarpy.plot.info.yml`.
            range_limits (tuple[float, float] | None, optional): Range limits. Defaults to None. None means automatic limits.

        Raises:
            ValueError: Error plotting.

        Returns:
            Axes: Axes with the plot.
        """

        # Load netcdf
        elda = xr.open_dataset(self.path)
        elda["altitude"] = elda["altitude"] / 1e3
        color = PLOT_INFO["lidar_ratio2color"][self.wavelength]
        for time_ in elda.time.values:
            try:
                (
                    elda.sel(time=time_)["extinction"]
                    / elda.sel(time=time_)["backscatter"]
                ).plot(
                    y="altitude",
                    ax=ax,
                    label=self.string_ID,
                    color=color,
                    linewidth=1.0,
                )
            except:
                raise ValueError(f"Error plotting {self.path.name} | {time_}")
        # plot vertical line at zero
        ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
        if lidar_ratio_limits is not None:
            ax.set_xlim(*lidar_ratio_limits)
        else:
            ax.set_xlim(*PLOT_INFO["limits"]["lidar_ratio"])
        if range_limits is not None:
            ax.set_ylim(range_limits)
        ax.set_xlabel(r"$LR_{part}$, [sr]")
        elda.close()
        return ax
