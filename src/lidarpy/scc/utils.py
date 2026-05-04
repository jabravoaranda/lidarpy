import os
from pathlib import Path
from pdb import set_trace
import sys
import glob
import json
import importlib
import numpy as np
import xarray as xr
import datetime as dt
from loguru import logger
from multiprocessing import Pool

from lidarpy.scc.licel2scc import licel2scc

from lidarpy.general_utils.io import read_yaml

"""
Functions for deriving SCC info directly from measurement folder/files
(Currently Not Used)
"""

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


""" get date from raw lidar files name """
def date_from_filename(filelist):
    """
    It takes the date from the file name of licel files.
    Parameters
    ----------
    filelist: list, str
        list of licel-formatted files

    Returns
    -------
    datelist: list, datetime
        list of datetimes for each file in input list
    """

    datelist = []
    if filelist:
        for _file in filelist:
            body = _file.split(".")[0]
            tail = _file.split(".")[1]
            year = int(body[-7:-5]) + 2000
            month = body[-5:-4]
            try:
                month = int(month)
            except Exception as e:
                if month == "A":
                    month = 10
                elif month == "B":
                    month = 11
                elif month == "C":
                    month = 12
            day = int(body[-4:-2])
            hour = int(body[-2:])
            minute = int(tail[0:2])
            # print('from body %s: the date %s-%s-%s' % (body, year, month, day))
            cdate = dt.datetime(year, month, day, hour, minute)
            datelist.append(cdate)
    else:
        print("Filelist is empty.")
        datelist = None
    return datelist


def getTP(filepath):
    """
    Get temperature and pressure from header of a licel-formatted binary file.
    Inputs:
    - filepath: path of a licel-formatted binary file (str)
    Output:
    - temperature: temperature in celsius (float).
    - pressure: pressure in hPa (float).
    """
    # This code should evolve to read the whole header, not only temperature and pressure.
    if os.path.isfile(filepath):
        with open(filepath, mode="rb") as f:  #
            filename = f.readline().decode("utf-8").rstrip()[1:]
            second_line = f.readline().decode("utf-8")
            f.close()
        second_line_list = second_line.split(" ")
        if len(second_line_list) == 14:
            temperature = float(second_line_list[12].rstrip())
            pressure = float(second_line_list[13].rstrip())
        else:
            logger.warning("Cannot find temperature, pressure values. set to None")
            temperature = None
            pressure = None
    else:
        logger.warning("File not found.")
        temperature = None
        pressure = None
    return temperature, pressure


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
