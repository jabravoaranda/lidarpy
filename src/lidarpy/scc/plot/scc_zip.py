from codecs import ignore_errors
from datetime import datetime, date
import shutil
import sys
import logging
from pathlib import Path
import tempfile
import xarray as xr

import matplotlib.pyplot as plt
from lidarpy.scc.plot.utils import plot_angstrom
from lidarpy.scc.plot.retrieval import angstrom_exponent

from lidarpy.general_utils.io import unzip_file
from lidarpy.scc.plot.product import SCC_elpp, SCC_elda, SCC_raw

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class SCC_zipfile:
    """A simple class to manage SCC products."""
    _path: Path
    def __init__(self, path: Path):
        """Initialize SCC_zipfile class.

        Args:
            path (Path): Path of SCC zip file.
        """
        self._path = path
        self.type, self.measurementID = self._get_info(self.path)
        self._unzipped_dir = None
        self._is_zip = None
        self.products = self.get_products()
        self.hoi_configuration_ID = self.get_hoi_configuration_ID()
        self.date = self.get_date_from_measurementID(measurementID=self.measurementID)

    @property
    def path(self) -> Path:
        """Check if the path exists and is a zip file.

        Args:
            path (Path): Path of SCC zip file.

        Raises:
            ValueError: Filepath does not exist.
            ValueError: Filepath is not a zip file.

        Returns:
            Path: Path of SCC zip file.
        """        
        if not self._path.exists():
            raise ValueError("Filepath does not exist.")
        # check zip
        if not self._path.suffix.endswith("zip"):
            raise ValueError(f"Filepath {self._path} is not a zip file.")
        return self._path

    @property
    def is_zip(self) -> bool:
        if self._is_zip is None:
            self._is_zip = self.path.suffix.endswith("zip")
        return self._is_zip

    def _get_info(self, path: Path) -> tuple[str, str]:
        """Get info from SCC zip file.

        Args:
            path (Path): Path of SCC zip file.

        Returns:
            tuple[str: str]: SCC product type (scc type as elpp, elda, ...) and measurementID.
        """
        # split pathfilename elpp_20230706gra0200.zip
        pathfilename = path.name
        type, filename = pathfilename.split("_")
        if type == "optical":
            type = "elda"
        if type == "preprocessed":
            type = "elpp"
        return type, filename.split(".")[0]
    
    @property
    def unzipped_dir(self) -> Path:
        if self._unzipped_dir is None:            
            self._unzipped_dir = self.unzip()
            return Path(self._unzipped_dir.name)
        else:
            return Path(self._unzipped_dir.name)        

    def unzip(
        self, pattern_or_list: str = r"*.*", destination: Path | None = None
    ) -> tempfile.TemporaryDirectory | None:
        """Extract the zip file

        Args:

            - pattern_or_list (str, optional): pattern or list of patterns. Defaults to r'\\.\\d+$'.
            - destination (Path | None, optional): Directory to extract files. Defaults to None (extract to the same directory as the zip file).
        """
        if self._unzipped_dir is None:
            self._unzipped_dir = unzip_file(
                self.path, pattern_or_list=pattern_or_list, destination=destination
            )                
        return self._unzipped_dir
    
    def remove_unzipped_dir(self):
        """Remove unzipped directory."""
        if self._unzipped_dir and self.unzipped_dir.is_dir():
            try:
                self._unzipped_dir.cleanup()
            except Exception:                
                shutil.rmtree(self.unzipped_dir, ignore_errors=True)  
            self._unzipped_dir = None
        else:
            logger.warning(f"Directory {self.unzipped_dir} does not exist.")
            

    def get_date_from_measurementID(self, measurementID: str) -> date:
        """Get date from measurementID.

        Args:
            measurementID (str): measurementID as 20230706gra0200

        Returns:
            datetime: datetime object.
        """
        return datetime.strptime(measurementID[0:8], "%Y%m%d").date()

    def get_products(self) -> list[SCC_raw | SCC_elpp | SCC_elda]:
        """Get products from unzipped folder.

        Returns:
            list[SCC_elpp | SCC_elda]: List of SCC_elpp or SCC_elda objects.
        """

        if self.type == "raw":
            return [SCC_raw(file_) for file_ in self.unzipped_dir.rglob(f"*.nc")]

        if self.type == "elpp":
            return [
                SCC_elpp(file_)
                for file_ in self.unzipped_dir.rglob(f"*{self.type}*.nc")
            ]

        if self.type == "elda":
            return [
                SCC_elda(file_)
                for file_ in self.unzipped_dir.rglob(f"*{self.type}*.nc")
            ]

    def angstrom_exponent(self) -> dict:
        """Retrieve angstrom exponent from elda products.

        Returns:
            dict[str: xr.DataArray]: Dictionary with angstrom exponents.
        """
        dict_angstrom_exponent = {}
        for product1 in self.products:
            if product1.earlinet_product_type[0] != "b":
                continue
            for product2 in self.products:
                if product2.earlinet_product_type[0] != "b":
                    continue
                if product1.wavelength >= product2.wavelength:
                    continue
                name_ = f"{product1.wavelength}-{product2.wavelength}"
                dict_angstrom_exponent[name_] = angstrom_exponent(product1, product2)
        return dict_angstrom_exponent

    def get_hoi_configuration_ID(self) -> int | None:
        """Get hoi_configuration_ID from global attributes.

        Returns:
            int: hoi_configuration_ID.
        """
        # get hoi_configuration_ID from global attributes

        if self.type == "raw":
            return None

        with xr.open_dataset(self.products[0].path) as ds_:
            return ds_.attrs["hoi_configuration_ID"]

    def plot_raw(
        self,
        output_dir: Path | None = None,
        dpi: int = 300,
        raw_limits: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
        combine_channels: bool = False,
        savefig: bool = True,
    ) -> list[Path | None]:
        """Plot raw products.

        Args:
            output_dir (Path | None, optional): Path to save plots. Defaults to None means current working directory.
            dpi (int, optional): Dots per inch. Defaults to 300.

        Returns:
            list[Path]: Paths of the plots.
        """
        return [
            product.plot(
                output_dir=output_dir,
                dpi=dpi,
                raw_limits=raw_limits,
                range_limits=range_limits,
                savefig=savefig,
                combine_channels=combine_channels,
            )[-1]
            for product in self.products
        ]

    def plot_elpp(
        self,
        output_dir: Path | None = None,
        dpi: int = 300,
        attenuated_backscatter_limits: tuple[float, float] | None = None,
        normalization_range: tuple[float, float] | None = None,
        range_limits: tuple[float, float] | None = None,
        combine_channels: bool = False,
        savefig: bool = True,
    ) -> list[Path | None]:
        """Plot elpp products.

        Args:
            output_dir (Path | None, optional): Path to save plots. Defaults to None means current working directory.
            dpi (int, optional): Dots per inch. Defaults to 300.

        Returns:
            list[Path]: Paths of the plots.
        """
        return [
            product.rayleight_fit(
                output_dir=output_dir,
                dpi=dpi,
                attenuated_backscatter_limits=attenuated_backscatter_limits,
                normalization_range=normalization_range,
                range_limits=range_limits,
                savefig=savefig,
                combine_channels=combine_channels,
            )[-1]
            for product in self.products
        ]

    def plot_elda(
        self,
        output_dir: Path | None = None,
        dpi: int = 300,
        range_limits: tuple[float, float] | None = None,
        backscatter_limits: tuple[float, float] | None = None,
        extinction_limits: tuple[float, float] | None = None,
        lidar_ratio_limits: tuple[float, float] | None = None,
        ae_limits: tuple[float, float] | None = None,
        depo_limits: tuple[float, float] | None = None,
    ) -> Path:

        """Plot elda products

        Args:
            output_dir (Path | None, optional): Path to save plots. Defaults to None means current working directory.
            dpi (int, optional): Dots per inch. Defaults to 300.

        Returns:
            Path: Path of the png image.
        """
        fig, (
            ax_backscatter,
            ax_angstrom,
            ax_depo,
            ax_lidar_ratio,
            ax_extinction,
        ) = plt.subplots(nrows=1, ncols=5, figsize=[15, 7.5], sharey=True)

        for product in self.products:
            optical_product = product.earlinet_product_type
            if optical_product[0] == "b":
                ax_backscatter = product.plot_backscatter(
                    ax=ax_backscatter,
                    backscatter_limits=backscatter_limits,
                    range_limits=range_limits,
                )
                if "dr" in product.string_ID:
                    ax_depo = product.plot_depo(
                        ax=ax_depo, depo_limits=depo_limits, range_limits=range_limits
                    )
            # elif optical_product[0] == "e":
            #     ax_extinction = product.plot_extinction(
            #         ax=ax_extinction,
            #         extinction_limits=extinction_limits,
            #         range_limits=range_limits,
            #     )
            #     if "lr" in product.string_ID:
            #         ax_lidar_ratio = product.plot_lidar_ratio(
            #             ax=ax_lidar_ratio,
            #             lidar_ratio_limits=lidar_ratio_limits,
            #             range_limits=range_limits,
            #         )
        ax_angstrom = plot_angstrom(
            self.angstrom_exponent(),
            ax=ax_angstrom,
            ae_limits=ae_limits,
            range_limits=range_limits,
        )

        # Fix y label. All empty but the first with "Range, [km]"
        for ax in [ax_backscatter, ax_angstrom, ax_depo, ax_lidar_ratio, ax_extinction]:
            ax.set_ylabel("")
            ax.set_title("")
            ax.legend(fontsize=10, loc="upper right")
            ax.grid()
            ax.minorticks_on()
        ax_backscatter.set_ylabel("Range, [km]")
        fig.suptitle(
            f"SCC config: {self.products[0].hoi_configuration_ID} | Measurement ID: {self.measurementID} | Averaged Time: {self.products[0].datetime_ini.strftime('%H:%M')} - {self.products[0].datetime_end.strftime('%H:%M')}"
        )
        fig.tight_layout()
        # save
        if isinstance(output_dir, Path):
            output_dir.mkdir(parents=True, exist_ok=True)
        elif output_dir is None:
            output_dir = Path.cwd()
        outputpath = output_dir.joinpath(self.path.name.replace(".zip", ".png"))
        fig.savefig(outputpath, dpi=dpi)
        plt.close(fig)
        return outputpath

    def plot(
        self,
        output_dir: Path | None = None,
        dpi: int = 300,
        range_limits: tuple[float, float] = (0, 25),
        raw_limits: tuple[float, float] | None = None,
        attenuated_backscatter_limits: tuple[float, float] | None = None,
        backscatter_limits: tuple[float, float] | None = None,
        extinction_limits: tuple[float, float] | None = None,
        lidar_ratio_limits: tuple[float, float] | None = None,
        ae_limits: tuple[float, float] | None = None,
        depo_limits: tuple[float, float] | None = None,
        elpp_combine_channels: bool = False,
        elpp_normalization_range: tuple[float, float] = (6, 7),
        savefig: bool = True,
    ) -> Path | list[Path]:
        """Plot content of SCC zip file.

        Args:
            output_dir (Path | None, optional): Directory to save plots. Defaults to None means current working directory.
            dpi (int, optional): Dots per inch. Defaults to 300.
            range_limits (tuple, optional): Set range limit in km. Defaults to None means (0,14).
            raw_limits (tuple, optional): Set raw limit in arbitrary units. Defaults to None means autofix.
            backscatter_limits (tuple, optional): Set backscatter limit in Mm^-1 sr^-1. Defaults to None means autofix.
            extinction_limits (tuple, optional): Set extinction limit in Mm^-1. Defaults to None means autofix.
            lidar_ratio_limits (tuple, optional): Set lidar ratio limit in sr. Defaults to None means values from lidarpy.plot.info.yml
            ae_limits (tuple, optional): Set angstrom exponent limit. Defaults to None means values from lidarpy.plot.info.yml
            depo_limits (tuple, optional): Set depolarization limit. Defaults to None means values from lidarpy.plot.info.yml

        Returns:
            Path | list[Path]: List of plots or single plot.
        """

        if self.type == "raw":
            return self.plot_raw(
                output_dir=output_dir,
                dpi=dpi,
                raw_limits=raw_limits,
                range_limits=range_limits,
                combine_channels=True,
                savefig=savefig,
            )

        if self.type == "elpp":
            return self.plot_elpp(
                output_dir=output_dir,
                dpi=dpi,
                attenuated_backscatter_limits=attenuated_backscatter_limits,
                normalization_range=elpp_normalization_range,
                range_limits=range_limits,
                combine_channels=elpp_combine_channels,
                savefig=savefig,
            )

        elif self.type == "elda":
            return self.plot_elda(
                output_dir=output_dir,
                dpi=dpi,
                range_limits=range_limits,
                backscatter_limits=backscatter_limits,
                extinction_limits=extinction_limits,
                lidar_ratio_limits=lidar_ratio_limits,
                ae_limits=ae_limits,
                depo_limits=depo_limits,
            )
