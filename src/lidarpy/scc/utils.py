import os
from pathlib import Path
from pdb import set_trace
import sys
import glob
import json
import importlib
import re
import numpy as np
import xarray as xr
import datetime as dt
from loguru import logger
from multiprocessing import Pool

from lidarpy.scc.licel2scc import licel2scc
from lidarpy.scc.licel2scc.licel import LicelFile
from lidarpy.utils.utils import date_from_filename, getTP

from lidarpy.general_utils.io import read_yaml

"""
Functions for deriving SCC info directly from measurement folder/files
(Currently Not Used)
"""


SCC_CONFIG_DIRECTORY = Path(__file__).parent / "scc_configFiles"
ACTRIS_CONFIG_FILE = SCC_CONFIG_DIRECTORY / "actris_config.yml"


def get_scc_config_id_from_binary(
    binary_file: str | Path,
    lidar_prefix: str,
    *,
    scc_config_directory: str | Path | None = None,
    target_datetime: dt.datetime | None = None,
    actris_config_file: str | Path | None = None,
) -> int | None:
    """Identify the SCC configuration ID represented by a Licel file.

    If the ACTRIS configuration describes a rule table for the lidar, the
    binary channels are first converted to feature flags and matched against
    that table. Otherwise, the selection falls back to the available
    ``*_parameters_scc_*.py`` files.
    """
    licel_file = LicelFile(str(binary_file), use_id_as_name=True, import_now=False)
    measurement_datetime = target_datetime or licel_file.start_time.replace(tzinfo=None)
    channel_ids = {channel_info["ID"] for channel_info in licel_file.channel_info}
    configs = find_scc_parameter_files(
        lidar_prefix,
        scc_config_directory=scc_config_directory,
        target_datetime=measurement_datetime,
    )
    actris_config = load_actris_config(actris_config_file)
    system_config = get_actris_system_config(actris_config, lidar_prefix)
    if system_config is not None:
        channel_string_ids = channel_string_ids_from_transient_ids(channel_ids, configs)
        return select_scc_config_id_from_actris_rules(
            channel_string_ids, system_config
        )

    return select_scc_config_id_from_channel_ids(
        channel_ids,
        configs,
    )


def read_licel_channel_ids(binary_file: str | Path) -> set[str]:
    """Read transient channel IDs from a Licel binary header."""
    licel_file = LicelFile(str(binary_file), use_id_as_name=True, import_now=False)
    return {channel_info["ID"] for channel_info in licel_file.channel_info}


def load_actris_config(actris_config_file: str | Path | None = None) -> dict:
    """Load ACTRIS SCC selection config."""
    config_file = Path(actris_config_file) if actris_config_file else ACTRIS_CONFIG_FILE
    if not config_file.is_file():
        return {}
    config = read_yaml(config_file)
    return config or {}


def get_actris_system_config(actris_config: dict, lidar_prefix: str) -> dict | None:
    """Return the ACTRIS system config matching a parameter-file prefix."""
    systems = actris_config.get("systems", {})
    for system_config in systems.values():
        if system_config.get("parameter_prefix") == lidar_prefix:
            return system_config
    return None


def channel_string_ids_from_transient_ids(
    channel_ids: set[str],
    scc_config_files: dict[int, Path],
) -> set[str]:
    """Map Licel transient IDs such as ``BT0`` to SCC channel string IDs."""
    channel_string_ids = set()
    for config_file in scc_config_files.values():
        config = get_scc_config(str(config_file))
        if config is None:
            continue
        for transient_id in channel_ids:
            parameters = config["channel_parameters"].get(transient_id)
            if parameters is not None and "channel_string_ID" in parameters:
                channel_string_ids.add(parameters["channel_string_ID"])
    return channel_string_ids


def select_scc_config_id_from_actris_rules(
    channel_string_ids: set[str],
    system_config: dict,
) -> int | None:
    """Select SCC config ID from ACTRIS feature rules."""
    features = get_channel_features_from_actris_config(
        channel_string_ids, system_config
    )
    for rule in system_config.get("scc_config_rules", []):
        if all(
            rule[key] == features[key]
            for key in ("has_far", "has_near", "has_raman")
        ):
            return int(rule["scc_config_id"])
    return None


def get_channel_features_from_actris_config(
    channel_string_ids: set[str],
    system_config: dict,
) -> dict[str, bool]:
    """Calculate feature flags used by ACTRIS SCC config rules."""
    pattern = re.compile(system_config["channel_string_id_pattern"])
    elastic_wavelengths = {
        int(wavelength) for wavelength in system_config["elastic_wavelengths"]
    }
    far_tokens = set(system_config.get("far_telescope_tokens", []))
    near_tokens = set(system_config.get("near_telescope_tokens", []))
    features = {"has_far": False, "has_near": False, "has_raman": False}

    for channel_string_id in channel_string_ids:
        match = pattern.match(channel_string_id)
        if match is None:
            continue
        telescope = match.groupdict().get("telescope")
        wavelength = int(match.group("wavelength"))
        features["has_far"] = features["has_far"] or telescope in far_tokens
        features["has_near"] = features["has_near"] or telescope in near_tokens
        features["has_raman"] = features["has_raman"] or wavelength not in elastic_wavelengths

    return features


def find_scc_parameter_files(
    lidar_prefix: str,
    *,
    scc_config_directory: str | Path | None = None,
    target_datetime: dt.datetime | None = None,
) -> dict[int, Path]:
    """Return the best dated SCC parameter file per config ID."""
    config_directory = (
        Path(scc_config_directory) if scc_config_directory else SCC_CONFIG_DIRECTORY
    )
    candidates = sorted(
        config_directory.glob(f"{lidar_prefix}_parameters_scc_*.py")
    )
    selected: dict[int, tuple[dt.datetime | None, Path]] = {}
    for candidate in candidates:
        parsed = _parse_scc_parameter_filename(candidate.name, lidar_prefix)
        if parsed is None:
            continue
        config_id, config_date = parsed
        if (
            target_datetime is not None
            and config_date is not None
            and config_date > target_datetime
        ):
            continue

        previous = selected.get(config_id)
        if previous is None or _scc_config_date_key(config_date) > _scc_config_date_key(previous[0]):
            selected[config_id] = (config_date, candidate)
    return {config_id: path for config_id, (_, path) in selected.items()}


def select_scc_config_id_from_channel_ids(
    channel_ids: set[str],
    scc_config_files: dict[int, Path],
) -> int | None:
    """Select the most specific SCC config compatible with channel IDs."""
    matches = []
    for config_id, config_file in scc_config_files.items():
        config = get_scc_config(str(config_file))
        if config is None:
            continue
        config_channels = set(config["channel_parameters"])
        if config_channels and config_channels.issubset(channel_ids):
            matches.append((len(config_channels), config_id))

    if not matches:
        return None
    return max(matches)[1]


def _parse_scc_parameter_filename(
    filename: str, lidar_prefix: str
) -> tuple[int, dt.datetime | None] | None:
    pattern = rf"^{re.escape(lidar_prefix)}_parameters_scc_(?P<code>\d+)(?:_(?P<date>\d{{8}}))?\.py$"
    match = re.match(pattern, filename)
    if match is None:
        return None
    config_date = None
    if match.group("date") is not None:
        config_date = dt.datetime.strptime(match.group("date"), "%Y%m%d")
    return int(match.group("code")), config_date


def _scc_config_date_key(config_date: dt.datetime | None) -> dt.datetime:
    return config_date or dt.datetime.min


def get_scc_code_from_measurement_folder(meas_folder, campaign):
    """
    get scc_code from measurement folder.
    Para todas las scc posibles (2) de la campaña, se busca si los canales que las definen existen en la medida.
    En caso de que existan las 2, la scc correcta es la que tiene más canales.
    Input:
    - measurement folder
    - campaign name
    """

    assert isinstance(meas_folder, str), "meas_folder must be String Type"
    assert isinstance(campaign, str), "campaign must be String Type"

    if campaign is not None:
        campaign_info, campaign_scc_fn = get_campaign_info(campaign)
        scc_cfg = campaign_info.scc_cfg
        scc_codes = campaign_info.scc_codes
        CustomLidarMeasurement = licel2scc.create_custom_class(
            campaign_scc_fn, use_id_as_name=True
        )
        rm_files = glob.glob(os.path.join(meas_folder, "R*"))
        if len(rm_files) > 0:
            rm_file = [rm_files[0]]
            measurement = CustomLidarMeasurement(rm_file)
            channels_in_rm = list(measurement.channels.keys())
            sccs = len(scc_codes)
            exist_scc = [False] * sccs
            channels_in_scc = [0] * sccs
            for i, i_scc in enumerate(scc_cfg):
                exist_scc[i] = all(
                    j in channels_in_rm for j in scc_cfg[i_scc]["channels"]
                )
                channels_in_scc[i] = len(scc_cfg[i_scc]["channels"])
            scc_code = [b for a, b in zip(exist_scc, scc_codes) if a]
            if len(scc_code) == 0:  # no estan los scc posibles en la medida
                scc_code = None
            elif len(scc_code) == 1:  # hay 1.
                scc_code = int(scc_code[0])
            elif (
                len(scc_code) == 2
            ):  # los dos son posibles. elegimos el que tiene mas canales
                max_chan = channels_in_scc == np.max(channels_in_scc)
                scc_code = [b for a, b in zip(max_chan, scc_codes) if a]
                scc_code = int(scc_code[0])
        else:
            scc_code = None
    else:
        scc_code = None
    return scc_code


def get_campaign_info(campaign, scc_config_directory=None):
    """ """
    # TODO: make scc_config_directory optional ¿?. Enlazar con crear archivo de configuracion
    try:
        if scc_config_directory is None:
            scc_config_directory = os.path.join(
                os.path.abspath(__file__), "scc_configFiles"
            )
        if campaign == "covid":  # there is only one campaign so far.
            campaign_scc_fn = "scc_channels_covid19.py"
            campaign_scc_fn = os.path.join(scc_config_directory, campaign_scc_fn)
            campaign_info = import_campaign_scc(campaign_scc_fn)
            campaign_info.scc_codes = [*campaign_info.scc_cfg]
        else:
            campaign_info = None
            campaign_scc_fn = None
    except:
        campaign_info = None
        campaign_scc_fn = None

    return campaign_info, campaign_scc_fn


def import_campaign_scc(campaign_scc_fn):
    """ """
    # TODO: darle una vuelta
    try:
        sys.path.append(os.path.dirname(campaign_scc_fn))
        campaign_scc = importlib.import_module(
            os.path.splitext(os.path.basename(campaign_scc_fn))[0]
        )
    except:
        raise (f"ERROR. importing scc-channels info from {campaign_scc_fn}")
    return campaign_scc

""" Handling Exceedance of Execution Time.
    Inspired in: https://stackoverflow.com/questions/51712256/how-to-skip-to-the-next-input-if-time-is-out-in-python """
# Maximum Allowed Execution Time (seconds)
# Class for timeout exception
class TimeoutException(Exception):
    pass


# Handler function to be called when SIGALRM is received
def sigalrm_handler(signum, frame):
    # We get signal!
    raise TimeoutException()


def apply_pc_peak_correction(filelist, scc_pc_channels):
    """
    Correction of the PC peaks in the PC channels caused by PMT degradation.

    Parameters
    ----------
    filelist: list(str)
        File list (e.g., /c/*.nc') (list)

    Returns
    -------
    outputlist: list(str)
        NetCDF file [file]
    """
    outputlist = list()
    threshold = 1000

    # scc_pc_channels = [1047, 1048, 1090, 1093, 1094]  # TODO: esto aqui a fuego ...
    if np.logical_and(len(filelist) > 0, len(scc_pc_channels) > 0):
        for file_ in filelist:
            try:
                lxarray = xr.open_dataset(file_)
                output_directory = os.path.dirname(file_)
                filename = os.path.basename(file_)
                for channel_ in scc_pc_channels:
                    idx_channel = np.where(lxarray.channel_ID == channel_)[0]
                    if idx_channel.size > 0:
                        # Call pc_peak_correction from utils_gfat
                        profile_raw = lxarray.Raw_Lidar_Data[:, idx_channel, :].values
                        shape_raw = profile_raw.shape
                        profile = np.squeeze(profile_raw)
                        new_profile = mulhacen_pc_peak_correction(profile)
                        lxarray.Raw_Lidar_Data[:, idx_channel, :] = np.reshape(
                            new_profile.astype("int"), shape_raw
                        )

                        profile_raw = lxarray.Background_Profile[
                            :, idx_channel, :
                        ].values
                        shape_raw = profile_raw.shape
                        profile = np.squeeze(profile_raw)
                        new_profile = mulhacen_pc_peak_correction(profile)
                        lxarray.Background_Profile[:, idx_channel, :] = np.reshape(
                            new_profile.astype("int"), shape_raw
                        )

                # save corrected data in the same file
                auxfilepath = os.path.join(output_directory, "aux")
                lxarray.to_netcdf(path=auxfilepath, mode="w")
                os.remove(file_)
                os.rename(auxfilepath, file_)
            except Exception as e:
                logger.warning(str(e))
                logger.warning("PC peak correction not performed")
            outputlist.append(file_)
    else:
        print("Files not found.")
    return outputlist


def get_info_from_measurement_file(mea_fn, scc_config_fn):
    """
    From a R File, extract information

    Parameters
    ----------
    mea_fn: str
        full path of measurement file
    scc_config_fn: str
        py file of scc configuration

    Returns
    -------
    time_ini_i: datetime.datetime
        initial time
    time_end_i: datetime.datetime
        end time
    channels_in_rm: collections.OrderedDict
        channels info

    """
    CustomLidarMeasurement = licel2scc.create_custom_class(
        scc_config_fn, use_id_as_name=True
    )
    mea = CustomLidarMeasurement([mea_fn])  # MUY LENTO
    time_ini_i = mea.info["start_time"].replace(tzinfo=None)
    time_end_i = mea.info["stop_time"].replace(tzinfo=None)
    # channels: Object LicelChannel (atmospheric_lidar/licel.py, L516)
    channels_in_rm = mea.channels
    del CustomLidarMeasurement, mea
    return time_ini_i, time_end_i, channels_in_rm


def get_info_from_measurement_files(meas_files, scc_config_fn):
    """
    given a list of measurement files, information about:
        - scc configuration
        - time lapse of measurement is taken
    a scc_config.py file is needed to read file contents. [mhc_parameters_scc_xxx.py]

    Parameters
    ----------
    meas_files : [type]
        [description]
    scc_config_fn : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    try:
        # Parallelization: reduces near 1 minute when normal takes 2 minutes.
        with Pool(os.cpu_count()) as pool:
            x = [
                (
                    mea_fn,
                    scc_config_fn,
                )
                for mea_fn in meas_files
            ]
            res = np.array(pool.starmap(get_info_from_measurement_file, x))
        times_ini = [x[0] for x in res]
        times_end = [x[1] for x in res]
        channels_in_rm = [x[2] for x in res][0]
        time_start = times_ini[0].replace(second=0)
        time_end = times_end[-1] + dt.timedelta(minutes=1)
        time_end = time_end.replace(second=0)

        """ 
        # Non-parallel
        t1 = time.time()
        times_ini_s = []
        times_end_s = []
        for i, m_file in enumerate(meas_files):
            # times ini, end
            time_ini_i, time_end_i, channels_in_rm = get_info_from_measurement_file(scc_config_fn, m_file)
            if i == 0:
                # time of first measurement starts
                time_start = time_ini_i.replace(second=0)
            if i == len(meas_files) - 1:
                # time of last measurement ends
                time_end = time_end_i + dt.timedelta(minutes=1)
                time_end = time_end.replace(second=0)
            times_ini_s.append(time_ini_i)
            times_end_s.append(time_end_i)
        print(time.time() - t1)
        """
    except Exception as e:
        logger.error(str(e))
        logger.error("Measurement files not read")
        return

    return times_ini, times_end, time_start, time_end, channels_in_rm


def get_scc_config(scc_config_fn):
    """

    Parameters
    ----------
    scc_config_fn: str
        scc configuration file

    Returns
    -------
    scc_config_dict: dict
        Dictionary with scc configuration info from scc config file


    """
    try:
        sys.path.append(os.path.dirname(scc_config_fn))
        module = importlib.import_module(
            os.path.splitext(os.path.basename(scc_config_fn))[0]
        )
        scc_config_dict = dict()
        scc_config_dict["general_parameters"] = module.general_parameters
        scc_config_dict["channel_parameters"] = module.channel_parameters
        scc_config_dict["channels"] = [*scc_config_dict["channel_parameters"]]
    except:
        logger.warning("ERROR. importing scc parameteres from %s" % scc_config_fn)
        scc_config_dict = None

    return scc_config_dict


def get_campaign_config(
    campaign_cfg_fn=None,
    scc_config_id=None,
    hour_ini=None,
    hour_end=None,
    hour_resolution=None,
    timestamp=0,
    slot_name_type=0,
):
    """
    Get Campaign Info from file into a dictionary
    If not campaign_cfg_fn is given, a campaign_cfg is built, using,
    if scc_config_id is given

    Campaign config file is a json with different configurations as keys

    Parameters
    ----------
    campaign_cfg_fn : str
        campaign config file. Default: GFATserver
    scc_config_id: int
        scc lidar configuration number
    hour_ini: float
    hour_end: float
    hour_resolution: float
    timestamp: int
        0: timestamp at beginning of interval
        1: timestamp at center of interval
    slot_name_type: int
        0: earlinet: YYYYMMDD+station+slot_number.
        1: scc campaigns: YYYYMMDD+station+HHMM.
        2: alternative: YYYYMMDD+station+HHMM+_scc

    Returns
    -------
    scc_campaign_cfg: dict

    """

    if campaign_cfg_fn is None:
        campaign_cfg_fn = "GFATserver"

    if campaign_cfg_fn == "GFATserver":  # Default Campaign. Dictionary as template
        scc_campaign_cfg = {
            "name": "operational",
            "lidar_config": {
                "operational": {
                    "scc": scc_config_id,
                    "hour_ini": hour_ini,
                    "hour_end": hour_end,
                    "hour_res": hour_resolution,
                    "timestamp": timestamp,
                    "slot_name_type": slot_name_type,
                }
            },
        }
    else:  # If Campaign File is given
        if os.path.isfile(campaign_cfg_fn):
            with open(campaign_cfg_fn) as f:
                scc_campaign_cfg = json.load(f)
        else:
            logger.error(
                "Campaign File %s Does Not Exist. Exit program" % campaign_cfg_fn
            )
            sys.exit()
    return scc_campaign_cfg


def check_scc_output_inlocal(scc_output_slot_dn):
    """
    Check if Products have been downloaded from SCC server for a given slot output directory

    Parameters
    ----------
    scc_output_slot_dn: str
        full path local directory of scc slot
    Returns
    -------
    scc_output_inlocal: bool
        False if something has not been downloaded from SCC server
    """

    expected_dns = [
        "hirelpp",
        "cloudmask",
        "scc_preprocessed",
        "scc_optical",
        "scc_plots",
    ]
    exist_dns = [
        os.path.isdir(os.path.join(scc_output_slot_dn, i)) for i in expected_dns
    ]
    if not all(exist_dns):
        scc_output_inlocal = False
    else:
        if len(glob.glob(os.path.join(scc_output_slot_dn, "scc_optical"))) == 0:
            scc_output_inlocal = False
        if len(glob.glob(os.path.join(scc_output_slot_dn, "scc_plots"))) == 0:
            scc_output_inlocal = False
        else:
            scc_output_inlocal = True

    return scc_output_inlocal
