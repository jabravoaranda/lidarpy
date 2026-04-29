import xarray as xr
import re
import tempfile
import zipfile
import numpy as np
from pathlib import Path
from loguru import logger
from linc import get_config, write_nc_legacy
from linc.config.read import Config
from datetime import datetime as dt, time, timedelta, date


from lidarpy.preprocessing.lidar_preprocessing import preprocess
from lidarpy.utils.utils import (
    is_within_datetime_slice,
    licel_to_datetime,
    to_licel_date_str,
)
from lidarpy.general_utils.io import unzip_file
from lidarpy.utils.utils import LIDAR_INFO
from lidarpy.utils.utils import filter_wildcard
from lidarpy.nc_convert.utils import search_config_file
from lidarpy.utils.file_manager import filename2info, info2general_path, info2path
from lidarpy.utils.types import LidarName, MeasurementType, Telescope
from lidarpy.general_utils.utils import parse_datetime

RAW_FIRST_LETTER = LIDAR_INFO["metadata"]["licel_file_wildcard"]


class Measurement:
    path: Path
    type: MeasurementType
    lidar_name: LidarName
    telescope: Telescope
    _unzipped_dir: tempfile.TemporaryDirectory | None

    def __init__(
        self,
        path: Path,
        type: MeasurementType,
        lidar_name: str,
        telescope: str = "xf",
        **kwargs,
    ):
        self.path = path
        self.type = type
        self._is_zip = None
        self._filenames = None
        self.lidar_name = LidarName(lidar_name)
        self.telescope = Telescope(telescope)
        self._session_datetime = None
        self._datetimes = None
        self._unique_dates = None
        self._sub_dirs = None
        self._has_linked_dc = None
        self._dc = None
        self._filepaths = None
        self.config = None
        self._unzipped_dir = None
        self._is_linked = None

    @property
    def is_zip(self) -> bool:
        if self._is_zip is None:
            self._is_zip = self.path.suffix.endswith("zip")
        return self._is_zip

    @property
    def session_datetime(self) -> dt:
        if self._session_datetime is None:
            # Get datetime from filename RS_yyyymmdd_hhmm or RS_yyyymmdd_hhmm.zip
            self._session_datetime = dt.strptime(
                self.path.name.split(".")[0], f"{self.type.value}_%Y%m%d_%H%M"
            )
        return self._session_datetime

    @property
    def filenames(self) -> list[str]:
        """Get the filenames from the measurement directory (or zip file) withiout extracting them.

        Returns:
            list[str]: List of filenames.
        """

        if self.is_zip:
            with zipfile.ZipFile(self.path, "r") as zip_ref:
                self._filenames = sorted(
                    [
                        Path(file).name
                        for file in zip_ref.namelist()
                        if not (file.endswith("/") or file.endswith("dat"))
                    ]
                )
        else:
            self._filenames = sorted(
                [
                    file.name
                    for file in self.path.rglob("*.*")
                    if not (
                        file.name.endswith("/")
                        or file.name.endswith("dat")
                        or file.name.endswith("txt")
                    )
                ]
            )
        return self._filenames

    @property
    def sub_dirs(self) -> list[str]:
        """Extract sub-directories from the measurement path

        Returns:

            - list[str]: list of sub-directories.
        """
        if self._sub_dirs is None:
            folders = []
            path = self.path  # Store path in a local variable for use in the method
            if self.is_zip:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    file_list = zip_ref.namelist()
                    folders = [
                        file.split("/")[-2] for file in file_list if file.endswith("/")
                    ]
            elif path.is_dir():
                folders = [f.name for f in path.iterdir() if f.is_dir()]
            self._sub_dirs = folders
        return self._sub_dirs

    @property
    def datetimes(self) -> list[dt]:
        """Get the dates from the measurement

        Returns:

            - list[dt]: List of dates
        """
        if self._datetimes is None:
            self._datetimes = sorted(
                [licel_to_datetime(filename_) for filename_ in self.filenames]
            )
        return self._datetimes

    @property
    def unique_dates(self) -> list[date]:
        """Get the unique dates from the measurement (not including the linked measurements)

        Returns:

            - list[dt]: List of unique dates
        """
        if self._unique_dates is None:
            self._unique_dates = np.unique(
                np.array([datetime_.date() for datetime_ in self.datetimes])
            ).tolist()
        return self._unique_dates

    @property
    def has_linked_dc(self) -> bool:
        """Check if the DC type measurement exists

        Returns:

            - bool: True if the DC type measurement exists. False otherwise.
        """
        if self._has_linked_dc is None:
            if self.type == MeasurementType.DC:
                self._has_linked_dc = False
            else:
                dc_path = self.path.parent / self.path.name.replace(
                    self.type.value, MeasurementType.DC.value
                )
                self._has_linked_dc = dc_path.exists()
        return self._has_linked_dc

    @property
    def dc(self) -> "Measurement | None":
        """Get the DC type measurement

        Returns:

            - Measurement | None: DC type measurement
        """
        if self._dc is None:
            if self.type == MeasurementType.DC:
                self._dc = None
            else:
                if self.has_linked_dc:
                    dc_path = self.path.parent / self.path.name.replace(
                        self.type.value, MeasurementType.DC.value
                    )
                    self._dc = Measurement(
                        path=dc_path,
                        type=MeasurementType.DC,
                        lidar_name=self.lidar_name,
                    )
        return self._dc

    def get_filenames_within_datetime_slice(self, datetime_slice: slice) -> list[str]:
        return [
            filename
            for filename in self.filenames
            if is_within_datetime_slice(filename, datetime_slice)
        ]

    def unzip(
        self,
        pattern_or_list: str | list[str] = r"\.\d+$",
        destination: Path | None = None,
    ) -> Path | None:
        """Extract the zip file

        Args:

            - pattern_or_list (str, optional): pattern or list of patterns. Defaults to r'\\.\\d+$'.
            - destination (Path | None, optional): Directory to extract files. Defaults to None (extract to the same directory as the zip file).
        """
        if self.is_zip:
            if isinstance(pattern_or_list, str):
                _unzipped_dir = unzip_file(
                    self.path, pattern_or_list=pattern_or_list, destination=destination
                )
            else:
                _unzipped_dir = unzip_file(
                    self.path, pattern_or_list=pattern_or_list, destination=destination
                )                
        else:
            _unzipped_dir = None
        if isinstance(_unzipped_dir, tempfile.TemporaryDirectory):
            self._unzipped_dir = _unzipped_dir
            unzipped_dir = Path(_unzipped_dir.name)
        else:
            unzipped_dir = None
        return unzipped_dir

    def get_config(self, config_dir) -> Config:
        # Get lidar configuration file
        config_filepath = search_config_file(
            self.lidar_name, self.session_datetime, config_dir
        )
        return get_config(config_filepath)

    def has_target_date(self, target_date: dt | date | str) -> bool:
        """Check if the measurement has the target date

        Args:

            - date (dt): Target date

        Returns:

            - bool: True if the measurement has the target date. False otherwise.
        """
        if isinstance(target_date, str):
            date_ = parse_datetime(target_date).date()
        elif isinstance(target_date, dt):
            date_ = target_date.date()
        return date_ in self.unique_dates

    def generate_nc_output_path(
        self,
        output_dir: Path,
        target_date: dt | date | str | None = None,
        signal_type: MeasurementType | str = "rs",
        subdir: str = "",
        add_hour: bool = True,
    ) -> Path:
        """Generate the output dir for the netCDF file

        Args:

            - output_dir (Path): Directory to save the netCDF file.
            - signal_type (str): Signal type (rs :regular signal or sd: standard deviation).

        Returns:

            - Path: Output path for the netCDF file.
        """

        if isinstance(signal_type, str):
            signal_type = MeasurementType(signal_type.upper())
        
        if target_date is None:
            target_date = self.session_datetime
        elif isinstance(target_date, str):
            target_date = parse_datetime(target_date)        

        return info2path(
            lidar_name=self.lidar_name,
            date=target_date,
            measurement_type=self.type.value,
            signal_type=signal_type.value,
            telescope=self.telescope.value,
            dir=output_dir,
            subdir=subdir,
            add_hour=add_hour,
        )

    def _find_previous_paths(
        self, number_of_previous_days: int, lidar_name: LidarName, raw_dir: Path
    ) -> list[Path]:
        """Find the paths for the previous days

        Args:

             number_of_previous_days (int): Number of previous days to look for.
             lidar_name (LidarName): Lidar name (see lidarpy.nc_convert.types.LidarName).
             raw_dir (Path): Directory for the raw data.

        Returns:

             list[Path]: List of paths for the previous days.
        """
        current_date = self.session_datetime.date()
        prev_paths = [
            info2general_path(
                lidar_name.value,
                date=current_date - timedelta(days=n_day),
                data_dir=raw_dir,
            )
            for n_day in range(1, number_of_previous_days + 1)
        ]
        # Remove paths that do not exist
        previous_paths = [prev_path for prev_path in prev_paths if prev_path.exists()]
        return previous_paths

    def get_filepaths(
        self,
        pattern_or_list: str | list[str] = r"\.\d+$",
        destination: Path | None = None,
    ) -> set[Path] | None:
        """Get the filepaths from the measurement directory (or zip file) extracting them.

        Args:
            pattern_or_list (str, optional): Wildcard pattern or list of. Defaults to r"\\.\\d+$" (all). Example hour period (15:30-16:30) -> (r'RM\\d{6}(15[3-5]\\d|16[0-2]\\d)\\.\\d{7}')
            within_period (tuple[dt, dt] | None, optional): Initial (included) and final (excluded) hour of the period. Defaults to None (all files).
            include_linked_measurements (bool, optional): Include linked measurements. Defaults to True.

        Raises:
            Exception: If there are no files found in the measurement directory to meet the wildcard.

        Returns:
            set[Path]: Set of filepaths.
        """
        if self.is_zip:
            dir_ = self.unzip(pattern_or_list=pattern_or_list, destination=destination)
        else:
            dir_ = self.path

        if dir_ is None:
            raise Exception(
                f"No files found in {self.path} to meet the wildcard {pattern_or_list}"
            )

        if self.is_zip:
            found_files = set([*dir_.rglob("*.*")])
        else:
            found_files = set(filter_wildcard(dir_, pattern_or_list))
        return found_files

    def to_nc(
        self,
        pattern_or_list: str | list[str] = r"\.\d+$",
        output_dir: str | Path | None = None,
        config_dir: str | Path | None = None,
        filename: str | None = None,
        by_dates: bool = False,
        licel_pattern: str = "[A-Za-z]{2}\\d{2}[A-Za-z0-9]{1}\\d{4}\\.\\d{6}",
        overwrite: bool = False,
    ) -> list[Path] | None:
        """Convert the measurement to a netCDF file.

        Args:
            pattern_or_list (str | list[str], optional): Wildcard of licel files (only for RS and HF measurements).
            output_dir (str | Path | None, optional): Output directory. Defaults to current working directory.
            config_dir (str | Path | None, optional): Configuration directory. Defaults to automatic search.
            filename (str | None, optional): Output filename. Defaults to automatic generation.
            by_dates (bool, optional): Split conversion by dates (only apply for RS and HF measurement types).
            licel_pattern (str, optional): Pattern for licel filenames used in multi-subdir handling.
            overwrite (bool, optional): If True, overwrite existing output files. If False and a file exists, raises FileExistsError.

        Returns:
            list[Path] | None: List of output paths or None if no files were converted.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        elif output_dir is None:
            output_dir = Path.cwd()

        # Get lidar configuration file
        config = self.get_config(config_dir=config_dir)

        def _prepare_and_write(result_path: Path, files2convert, config):
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if filename is not None:
                result_path = result_path.parent / filename
            # handle existing file
            if result_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Output file {result_path!s} already exists. Pass overwrite=True to replace it."
                    )
                try:
                    result_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove existing file {result_path}: {e}")
            logger.info(f"Writing {result_path.name}")
            try:
                write_nc_legacy(files2convert, result_path, config=config)
            except Exception as e:
                raise Exception(f"Error writing {result_path}: {e}")
            return result_path

        if self.type == MeasurementType.RS or self.type == MeasurementType.HF:
            if by_dates:
                result_paths = []
                for date_ in self.unique_dates:
                    dstr = to_licel_date_str(dt.combine(date_, time(0, 0)))[:5]
                    pattern_: str = r"[A-Za-z]{2}" + f"{dstr}" + r"\d{2}\.\d{6}"
                    files2convert = self.get_filepaths(pattern_or_list=pattern_)
                    if files2convert:
                        files2convert = sorted(list(files2convert))
                        date_ = licel_to_datetime(files2convert[0].name)
                        result_path = self.generate_nc_output_path(
                            target_date=date_, output_dir=output_dir, add_hour=True
                        )
                        written = _prepare_and_write(result_path, files2convert, config)
                        result_paths.append(written)
            else:
                files2convert = self.get_filepaths(pattern_or_list=pattern_or_list)
                if files2convert:
                    files2convert = sorted(list(files2convert))
                    date_ = licel_to_datetime(files2convert[0].name)
                    result_path = self.generate_nc_output_path(
                        target_date=date_, output_dir=output_dir
                    )
                    written = _prepare_and_write(result_path, files2convert, config)
                    result_paths = [written]
        elif self.type == MeasurementType.DC:
            files2convert = self.get_filepaths()
            if files2convert:
                result_path = self.generate_nc_output_path(output_dir=output_dir)
                written = _prepare_and_write(result_path, files2convert, config)
                result_paths = [written]
        else:
            result_paths = []
            for subdir in self.sub_dirs:
                if subdir == self.unique_dates[0].strftime("%Y%m%d"):
                    continue
                try:
                    pattern = r".*" + re.escape(f"{subdir}") + r".*" + licel_pattern
                    files2convert = self.get_filepaths(pattern_or_list=pattern)
                except Exception:
                    raise RuntimeError(f"Error while converting: {self.path}")
                if files2convert:
                    result_path = self.generate_nc_output_path(
                        output_dir=output_dir, subdir=subdir
                    )
                    written = _prepare_and_write(result_path, files2convert, config)
                    result_paths.append(written)
        return result_paths
    
    def remove_tmp_unzipped_dir(self):
        """Delete temporary folders created during the process.

        Raises:

            - OSError: If there is an error deleting the temporary folder.
        """
        if self._unzipped_dir is not None:
            try:
                to_be_deleted = Path(self._unzipped_dir.name)
                self._unzipped_dir.cleanup()
                self._unzipped_dir = None
                logger.info(f"Temporary folder deleted: {to_be_deleted}")
            except:
                logger.error(f"Error deleting temporary folder: {to_be_deleted.name}")

    # def __str__(self):
    #     return f"\nUnzipped Path: {self.unzipped_path}\n\n\n"
    def __str__(self):
        return f"Measurement Object\nPath: {self.path}\nType: {self.type}\nDates: {self.unique_dates}\nIs ZIP: {self.is_zip}\nHas DC: {self.has_linked_dc}\n"


def merge_measurements_by_date(
    lidar_name: str,
    target_date: dt | date | str,
    product_dir: str | Path,
    measurement_type: MeasurementType | None = None,
    output_dir: str | Path = Path.cwd(),
    channels: list[str] | None = None,
    **kwargs,
) -> xr.Dataset | xr.DataArray:

    if isinstance(product_dir, str):
        product_dir = Path(product_dir)
        logger.info(f"Searching measurements in {product_dir}")

    if isinstance(target_date, str):
        target_date = parse_datetime(target_date).date()
    elif isinstance(target_date, dt):
        target_date = target_date.date()
    logger.info(f"Target date: {target_date}")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        logger.info(f"Output directory: {output_dir}")

    if isinstance(measurement_type, str):
        measurement_type = MeasurementType(measurement_type)
    if measurement_type is None:
        raise ValueError("Measurement type is required.")

    # Create target date path
    path_with_measurements = []
    target_path = info2general_path(lidar_name, target_date, product_dir, data_level="1a")
    if not target_path.exists():
        logger.warning(
            f"No measurements found in {product_dir} for {lidar_name} on {target_date}"
        )
    else:
        path_with_measurements.append(target_path)

    # Create previous date
    prev_day = target_date - timedelta(days=1)
    prev_path = info2general_path(lidar_name, prev_day, product_dir, data_level="1a")

    if prev_path.exists():
        path_with_measurements.append(prev_path)

    if path_with_measurements == []:
        raise FileNotFoundError(
            f"No measurements found in {product_dir} for {lidar_name} on {target_date}"
        )

    logger.info(f"Searching measurements in previous target date path: {prev_path}")
    result_paths = []
    for path_ in path_with_measurements:
        for m_path in path_.glob("*"):
            result_paths.append(m_path)

    measures = []
    for m_path in result_paths:
        if m_path is None:
            continue
        _, _, type_, _, _, _ = filename2info(m_path.name)
        if measurement_type == MeasurementType(type_.upper()):
            measures.append(m_path)

    lidars = []
    crop_ranges = kwargs.get("crop_ranges", (0, 14000.0))
    for file in measures:
        lidar_ = preprocess(
            file, channels=channels, crop_ranges=crop_ranges, gluing_products=False
        )
        lidars.append(lidar_)

    merge_dataset = xr.combine_by_coords(
        [lidar_ for lidar_ in lidars], combine_attrs="drop_conflicts"
    )
    return merge_dataset


def info2measurements(
    lidar_name: str,
    target_date: dt | date | str,
    raw_dir: str | Path,
    measurement_type: MeasurementType | None = None,
) -> list[Measurement] | None:
    """Converts a list of paths to a list of Measurement objects.

    Args:
        lidar_name (str): _description_
        target_date (dt | date | str): _description_
        raw_dir (str | Path): _description_
        measurement_type (MeasurementType | None, optional): _description_

    Returns:
        list[Measurement] | None: _description_
    """

    if isinstance(raw_dir, str):
        raw_dir = Path(raw_dir)
        logger.info(f"Searching measurements in {raw_dir}")

    if isinstance(target_date, str):
        target_date = parse_datetime(target_date).date()
    elif isinstance(target_date, dt):
        target_date = target_date.date()
    logger.info(f"Target date: {target_date}")

    if isinstance(measurement_type, str):
        measurement_type = MeasurementType(measurement_type)

    list_paths = set()

    # Create target date path
    target_path = info2general_path(lidar_name, target_date, raw_dir)
    if target_path.exists():
        logger.info(f"Searching measurements in target path: {target_path}")
        list_paths.update([*target_path.glob("*")])

    # Looking for previous days
    prev_day = target_date - timedelta(days=1)
    prev_path = info2general_path(lidar_name, prev_day, raw_dir)

    if prev_path.exists():
        logger.info(f"Searching measurements in previous target date path: {prev_path}")
        list_paths.update([*prev_path.glob("*")])

    if list_paths == []:
        logger.warning(
            f"No measurements found in {raw_dir} for {lidar_name} on {target_date}"
        )
        return None

    logger.info(f"Number of measurement candidates: {len(list_paths)}")
    # Filter by type
    if measurement_type is not None:
        list_paths = [
            path_ for path_ in list_paths if measurement_type.value in path_.name
        ]

    logger.info(
        f"Number of measurement candidates after filtering by type: {len(list_paths)}"
    )

    measurements = []
    for path_ in list_paths:
        if len(path_.name.split(".")[0]) == 16:
            logger.info(f"Checking candidate measurement: {path_}")
            m_ = Measurement(
                path=path_,
                type=MeasurementType(path_.name[:2]),
                lidar_name=lidar_name,
            )
            if target_date in m_.unique_dates:
                logger.info(f"Candidate measurement accepted: {m_.path.name}")
                measurements.append(m_)
            else:
                logger.warning(f"Candidate measurement rejected: {path_.name}")
        else:
            logger.warning(f"Invalid measurement name (skipped): {path_.name}")
    if measurements == []:
        return None
    return measurements


def filter_by_type(
    measurements: list[Measurement], mtype: MeasurementType
) -> list[Measurement]:
    """Filter a list of measurements by type.

    Args:

        - measurements (list[Measurement]): List of measurements to filter.
        - mtype (MeasurementType): Type to filter by.

    Returns:

        - list[Measurement]: Filtered list of measurements.
    """
    return list(
        filter(
            lambda m: m.type == mtype,
            measurements,
        )
    )
