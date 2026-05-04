#!/usr/bin/env python
"""
SCC GFAT

Functionality: Chain Process of Lidar Measurements from Raw format to SCC server processing
    1. Convert Lidar Raw Data (0a) into SCC format
    2. Make some plots of the SCC format data
    3. Upload SCC format data to SCC server where data is processed
    4. Download Data processed in SCC server
    5. Make some plots of the processed data downloaded from SCC server

Usage:
run scc.py
    -i YYYYMMDD
    -d DATA_PATH [DATA_PATH/LIDAR/...]
    -c CAMPAIGN CONFIG FILE JSON

"""

import os
from pathlib import Path
import sys
import glob
import importlib
import shutil
import re
import argparse
from distutils.dir_util import mkpath
import itertools
import zipfile
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import json
import logging
from multiprocessing import Pool
import pickle
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from pdb import set_trace
from lidarpy.scc.io import move2odir
from lidarpy.scc.plot import plot_scc_input, plot_scc_output
from lidarpy.scc.utils import (
    SCC_INFO,    
    TimeoutException,
    apply_pc_peak_correction,
    check_scc_output_inlocal,
    get_campaign_config,
    get_info_from_measurement_files,
    get_scc_code_from_measurement_folder,
    get_campaign_info,
    get_scc_config,
    getTP,
)

""" 3rd party code dependencies """
from lidarpy.utils.utils import LIDAR_INFO

MODULE_DIR = os.path.dirname(sys.modules[__name__].__file__)
sys.path.insert(0, MODULE_DIR)
from lidarpy.scc.licel2scc import licel2scc  # copied to utils_gfat
import scc_access as sa  # copied to utils_gfat

# import gfat_config


""" """
__author__ = "Bravo-Aranda, Juan Antonio"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Bravo-Aranda, Juan Antonio"
__email__ = "jabravo@ugr.es"
__status__ = "Production"


""" directories of interest  """
# Root Directory (in NASGFAT)  according to operative system
# DATA_DN = gfat_config.DATA_DN

# Source Code Dir
MODULE_DN = Path(__file__).parent

# SCC Config Directory
SCC_CONFIG_DN = MODULE_DN / "scc_configFiles"

# Run Dir
RUN_DN = os.getcwd()


""" Common Variables along the module """
STATION_ID = "gra"

LIDAR_SYSTEMS = {
    "ALHAMBRA": {"nick": "alh", "code": ""},
    "MULHACEN": {"nick": "mhc", "code": ""},
}
MEASUREMENT_TYPES = ["RS", "HF"]


""" SCC Server Connection Settings """
SCC_SERVER_SETTINGS = SCC_INFO["server_settings"]

""" logging  """
log_formatter = logging.Formatter(
    "%(levelname)s: %(funcName)s(). L%(lineno)s: %(message)s"
)
debug = True
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
if debug:
    handler = logging.StreamHandler(sys.stdout)
else:
    log_fn = os.path.join(
        RUN_DN, "scc_%s.log" % dt.datetime.utcnow().strftime("%Y%m%dT%H%M")
    )
    handler = logging.FileHandler(log_fn)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


""" Email 
    From a gmail account. Permission must be granted previously.
    At manage account: less secure apps, enable
"""
# TODO: cambiar a INFO.yml y crear reader
email_sender = {
    "server": "smtp.gmail.com",  # "smtp.ugr.es",
    "port": 587,
    "sender_email": "controlador.mulhacen@gmail.com",
    "password": "lidarceama12345",
}
# email_sender = {
#                "server": "smtp.ugr.es", #"smtp.gmail.com", #"smtp.ugr.es",
#                "port": 587,
#                "sender_email": "icenet@ugr.es", #"controlador.mulhacen@gmail.com",
#                "password": "ib3r1c0", # "lidarceama12345"
#                }
email_receiver = ["dbp@ugr.es", "mjgranados@ugr.es", "rascado@ugr.es", "jabravo@ugr.es"]
# email_receiver = ["dbp@ugr.es"]


def send_email(email_sender, email_receiver, email_content):
    """ """
    # Check Input
    if isinstance(email_receiver, str):
        email_receiver = [email_receiver]

    logger.info("Send Email")
    date = dt.datetime.utcnow()
    return_code = 1
    try:
        # Prepare Message
        message = MIMEMultipart()
        message["From"] = email_sender["sender_email"]
        message["To"] = COMMASPACE.join(email_receiver)
        message["Date"] = formatdate(localtime=True)
        message["Subject"] = "SCC ERROR [%s UTC]]" % date.strftime("%Y%m%d_%H%M")
        message.attach(MIMEText(email_content))

        # Connect to email server and send email
        # ssl_context = ssl.create_default_context()
        # conn = smtplib.SMTP_SSL(email_sender['server'], email_sender['port'], context=ssl_context)
        conn = smtplib.SMTP(email_sender["server"], email_sender["port"])
        conn.ehlo()
        conn.starttls()
        conn.ehlo()
        conn.login(email_sender["sender_email"], email_sender["password"])
        conn.sendmail(email_sender["sender_email"], email_receiver, message.as_string())
        logger.info("Email Sent")
    except:
        logger.error("Email not sent")
        return_code = 0
    finally:
        conn.quit()
    return return_code


"""
Processor Functions
"""


def process_day(
    lidar_name: str,
    date_str: str,
    meas_type: str,
    scc_campaign_config: int,
    process: int,
    mode: int,
    raw_dir: Path,
    scc_dir: Path,
    check_in_server: bool = False,
):
    """[summary]

    Parameters
    ----------
    date_str: str, YYYYMMDD

    Returns
    -------
    status: bool
        True: all ok; False: something went totally or partially (slot) wrong
    msg: str
        output message. Especially useful if errors

    """
    logger.info("Start Process day: %s" % date_str)

    msg = ""

    # TODO: check input variables

    # date info
    date_dt = dt.datetime.strptime(date_str, "%Y%m%d")
    i_year, i_month, i_day = [date_dt.strftime(x) for x in ["%Y", "%m", "%d"]]
    r_year = date_dt.strftime("%y")
    if date_dt.month < 10:
        r_month = "%i" % date_dt.month
    elif date_dt.month == 10:
        r_month = "A"
    elif date_dt.month == 11:
        r_month = "B"
    elif date_dt.month == 12:
        r_month = "C"
    else:
        msg = "Error Naming Month For RS filename"
        logger.warning(msg)
        return False, msg
    r_day = date_dt.strftime("%d")
    r_date_str = f"{r_year}{r_month}{r_day}"

    """ Measurement Directories """
    day_0a_dir = raw_dir / i_year / i_month / i_day

    # Measurements Directories: RS
    rs_dns = day_0a_dir.glob(f"{meas_type}_{date_str}_*")

    # Fin RS files in previous day
    date_ytd_dt = date_dt - dt.timedelta(days=1)
    date_ytd_str = date_ytd_dt.strftime("%Y%m%d")
    y_year, y_month, y_day = [date_ytd_dt.strftime(x) for x in ["%Y", "%m", "%d"]]
    previous_dir = raw_dir / y_year / y_month / y_day
    add_rs_dns = previous_dir.glob(f"{meas_type}_{date_ytd_str}_*")

    rs_dns = itertools.chain(rs_dns, add_rs_dns)

    """ Run process for every configuration in the scc campaign config """
    campaign_name = scc_campaign_config["name"]
    lidar_configs = scc_campaign_config["lidar_config"]
    # Loop over lidar configurations within the campaign
    for lidar_config in lidar_configs:
        # Get Lidar Configuration Info
        ldr_cfg = lidar_configs[lidar_config]
        scc_id = ldr_cfg["scc"]
        hour_ini = ldr_cfg["hour_ini"]
        hour_end = ldr_cfg["hour_end"]
        hour_res = ldr_cfg["hour_res"]
        timestamp = ldr_cfg["timestamp"]
        slot_name_type = ldr_cfg["slot_name_type"]

        # SCC Config File
        if scc_id is None:
            # TODO: integrar deduccion de SCC a partir de la medida
            msg = "SCC cannot be None"
            logger.error(msg)
            return False, msg
        else:
            lidar_nick = LIDAR_INFO["metadata"]["name2nick"][lidar_name]
            scc_config_fn = SCC_CONFIG_DN / f"{lidar_nick}_parameters_scc_{scc_id}.py"

            if not scc_config_fn.exists():
                raise FileNotFoundError(f"SCC config file {scc_config_fn} not found")

        scc_config_dict = get_scc_config(scc_config_fn)

        logger.info("SCC = %d" % scc_id)

        # Measurement Files within the day
        regex_ = f"R*{r_date_str}*.*"
        rs_files = []
        for rs_dn in rs_dns:
            rs_files.extend(rs_dn.rglob(regex_))
        rs_files.sort()

        if len(rs_files) > 0:
            # Time Period of Measurements within the day (Bottleneck)
            (
                times_ini_rs,
                times_end_rs,
                time_start_rs,
                time_end_rs,
                meas_channels,
            ) = get_info_from_measurement_files(rs_files, scc_config_fn)
            # This is only for testing purposes
            # with open(os.path.join(MODULE_DN, "times.pickle"), "wb") as f:
            #   pickle.dump([times_ini_rs, times_end_rs, time_start_rs, time_end_rs, meas_channels], f)
            # with open(os.path.join(MODULE_DN, "times.pickle"), "rb") as f:
            #    times_ini_rs, times_end_rs, time_start_rs, time_end_rs, meas_channels = pickle.load(f)
            # More than 10 minutes of measurements
            if (time_end_rs - time_start_rs).total_seconds() / 60 < 10:
                msg = "Less than 10 minutes of measurements. Exit Program"
                logger.error(msg)
                return False, msg

            # Build Slots:
            logger.info("Building Slots...")
            if hour_ini is None:
                hour_ini = float(
                    time_start_rs.hour
                    + time_start_rs.minute / 60.0
                    + time_start_rs.second / 3600.0
                )
            else:
                hour_ini = float(hour_ini)
            if hour_end is None:
                if time_end_rs.day == time_start_rs.day:
                    hour_end = float(
                        time_end_rs.hour
                        + time_end_rs.minute / 60.0
                        + time_end_rs.second / 3600.0
                    )
                else:
                    hour_end = 23.99
            else:
                hour_end = float(hour_end)
            if hour_res is None:
                hour_res = float(1)
            else:
                hour_res = float(hour_res)
            if timestamp is None:
                timestamp = 1
            if slot_name_type is None:
                slot_name_type = 0
            if hour_end == hour_ini:
                hour_end = hour_ini + hour_res
            slots_ini = [
                date_dt + dt.timedelta(hours=x)
                for x in np.arange(hour_ini, hour_end, hour_res)
            ]
            slots_end = [s + dt.timedelta(hours=hour_res) for s in slots_ini]
            slots_stp = [
                s + dt.timedelta(hours=(timestamp / 2) * hour_res) for s in slots_ini
            ]
            logger.info("Done")

            # SCC input directory for the day:
            if isinstance(hour_res, int) or hour_res.is_integer():
                input_pttn = "input_%02dhoras" % hour_res
            else:
                input_pttn = "input_%dminutes" % int(hour_res * 60)
            scc_input_dn = (
                scc_dir / f"scc{scc_id}" / input_pttn / i_year / i_month / i_day
            )
            if not scc_input_dn.exists():
                scc_input_dn.mkdir(parents=True)
                logger.info("Local Directory %s CREATED" % scc_input_dn)

            # SCC Output Directory for the day
            scc_output_dn = Path(scc_input_dn.as_posix().replace("input", "output"))

            # SCC Object for Interaction withh SCC Server
            logger.info("Creating SCC object...")
            SCC_SERVER_SETTINGS[
                "output_dir"
            ] = scc_output_dn  # se tiene que llamar asi por scc_access
            scc_obj = sa.SCC(
                tuple(SCC_SERVER_SETTINGS["basic_credentials"]),
                SCC_SERVER_SETTINGS["output_dir"],
                SCC_SERVER_SETTINGS["base_url"],
            )
            logger.info("Done")

            # Loop over Slots
            # TODO: paralelizar?
            for s in range(len(slots_stp)):
                """Name Of Slot"""
                slot_number = s
                # Slot ID. For the Name of the Netcdf
                if slot_name_type == 0:  # EARLINET. operational slot_id
                    slot_id = f"{date_str}{STATION_ID}{slot_number}"
                elif slot_name_type == 1:  # SCC. timestamp
                    slot_id = f"{date_str}{STATION_ID}{slots_stp[s].hour:02d}{slots_stp[s].minute:02d}"
                elif slot_name_type == 2:  # ALTERNATIVE. timestamp + station + scc
                    slot_id = f"{date_str}{STATION_ID}{slots_stp[s].hour:02d}{slots_stp[s].minute:02d}_{scc_id}"
                else:
                    logger.warning("slot name type not defined. Set to operational")
                    slot_id = f"{date_str}{STATION_ID}{slot_number}"

                """ In Operational Mode, Check if Slot is completed so it can be processed """
                do_process_slot = True
                if mode == 0:
                    now_dt = dt.datetime.utcnow()
                    if now_dt < slots_end[s]:
                        do_process_slot = False

                """ Process Slot if Possible """
                if do_process_slot:
                    """Check Slot Exists in Local and in SCC server"""
                    # Local File for Slot
                    slot_fn = "%s.nc" % slot_id
                    scc_slot_fn = os.path.join(scc_input_dn, slot_fn)
                    if os.path.isfile(scc_slot_fn):
                        slot_in_local = True
                        logger.info("Slot %s Already exists in Local" % slot_id)
                    else:
                        slot_in_local = False
                        logger.info("Slot %s does not exist in Local" % slot_id)
                        # Slot in Server
                    if check_in_server:
                        scc_obj.login(SCC_SERVER_SETTINGS["website_credentials"])
                        meas_obj, _ = scc_obj.get_measurement(slot_id)
                        scc_obj.logout()
                        if meas_obj is not None:
                            slot_in_scc = True
                            logger.info("Slot %s Already exists in SCC" % slot_id)
                        else:
                            slot_in_scc = False
                            logger.info("Slot %s Does Not Exist in SCC" % slot_id)

                    # Local Output SCC Slot
                    scc_output_slot_dn = os.path.join(scc_output_dn, slot_id)

                    """ 0a to SCC """       
                    if np.logical_or.reduce((process == 0, process == 10)):
                        # not slot_in_local,
                        # Subset RS Files for Slot
                        ids = []
                        for (i, t_i), (j, t_e) in zip(
                            enumerate(times_ini_rs), enumerate(times_end_rs)
                        ):                            
                            slot_condition = np.logical_and(np.array(times_ini_rs) > slots_ini[s], np.array(times_ini_rs) < slots_end[s])
                            if slot_condition.any() > 0:  # IF there are files within the slot
                                logger.info("Creating SCC format for slot %s" % slot_id)
                                slot_times = times_ini_rs[slot_condition]
                                # RS Files and DC Pattern
                                rs_files_slot = [rs_files[i] for i in ids]
                                dc_files_patt = os.path.join(
                                    os.path.dirname(rs_files_slot[0]).replace(
                                        meas_type, "DC"
                                    ),
                                    "R*",
                                )
                                # if there is no operational DC, it is searched
                                if len(glob.glob(dc_files_patt)) == 0:
                                    try:
                                        logger.warning("No DC files for day %s" % date_str)
                                        logger.warning(
                                            "Search DC files within the previous 30 days"
                                        )
                                        for b_day in range(1, 30):
                                            b_date_dt = date_dt + dt.timedelta(days=-b_day)
                                            b_year, b_month, b_day = [
                                                b_date_dt.strftime(x)
                                                for x in ["%Y", "%m", "%d"]
                                            ]
                                            b_day_0a_dir = os.path.join(
                                                raw_dir, b_year, b_month, b_day
                                            )
                                            dc_files_patt = os.path.join(
                                                b_day_0a_dir, "DC*", "R*"
                                            )
                                            if len(glob.glob(dc_files_patt)) > 0:
                                                break
                                        if len(glob.glob(dc_files_patt)) > 0:
                                            dc_dns = np.unique(
                                                [
                                                    os.path.basename(os.path.dirname(x))
                                                    for x in glob.glob(dc_files_patt)
                                                ]
                                            ).tolist()
                                            dc_hours = np.asarray(
                                                [
                                                    dt.datetime.strptime(
                                                        i, "DC_%Y%m%d_%H%M"
                                                    ).hour
                                                    for i in dc_dns
                                                ]
                                            )
                                            # Daytime or Nighttime slot
                                            limit_hour = 17
                                            if (
                                                time_start_rs.hour <= limit_hour
                                            ):  # Daytimeslot
                                                ix = np.argwhere(dc_hours <= limit_hour)[0]
                                            else:  # Nighttime Slot
                                                ix = np.argwhere(dc_hours > limit_hour)[0]
                                            if len(ix) > 0:
                                                dc_dn = dc_dns[ix[0]]
                                                dc_files_patt = os.path.join(
                                                    b_day_0a_dir, dc_dn, "R*"
                                                )
                                    except Exception as e:
                                        logger.error(
                                            "No DC files found for slot %s" % slot_id
                                        )                    
                                # Take temperature and pressure from the first file:
                                temperature, pressure = getTP(rs_files_slot[0])
                                # Prepare Licel2SCC input:
                                CustomLidarMeasurement = licel2scc.create_custom_class(
                                    scc_config_fn,
                                    use_id_as_name=True,
                                    temperature=temperature,
                                    pressure=pressure,
                                )
                                set_trace()
                                # Convert from Raw to SCC Format:
                                try:
                                    licel2scc.convert_to_scc(
                                        CustomLidarMeasurement,
                                        rs_files_slot,
                                        dc_files_patt,
                                        slot_id,
                                        slot_number,
                                    )
                                except Exception as e:
                                    logger.error(
                                        "%s. Conversion from 0a to SCC not possible for slot %s"
                                        % (str(e), slot_id)
                                    )
                                    msg += "Slot %s not created. " % slot_id

                                """ Name of Netcdf File """
                                # With reverse engineering, we know that the name of the
                                # netcdf file is:
                                scc_slot_ini_fn = os.path.join(RUN_DN, slot_fn)

                                """ PC peak correction if MULHACEN """
                                if lidar_name == "MULHACEN":
                                    if os.path.isfile(scc_slot_ini_fn):
                                        logger.info("PC peak correction")
                                        pc_channels = []
                                        for c in scc_config_dict["channels"]:
                                            if c in meas_channels.keys():
                                                if (
                                                    meas_channels[c].analog_photon_string
                                                    == "ph"
                                                ):
                                                    pc_channels.append(
                                                        scc_config_dict[
                                                            "channel_parameters"
                                                        ][c]["channel_ID"]
                                                    )
                                        if len(pc_channels) > 0:
                                            scc_slot_ini_fn = apply_pc_peak_correction(
                                                [scc_slot_ini_fn], pc_channels
                                            )  # se traga una lista de archivos
                                            # Devuelve una lista de archivos. TODO: Arreglar esto
                                            scc_slot_ini_fn = scc_slot_ini_fn[0]

                                """ Move File to NAS """
                                if os.path.isfile(scc_slot_ini_fn):
                                    scc_saved = move2odir(scc_slot_ini_fn, scc_input_dn)
                                    if scc_saved:
                                        logger.info(
                                            "Created SCC Format for slot %s" % slot_id
                                        )
                                    else:
                                        logger.warning(
                                            "Error: SCC Format for slot %s not created"
                                            % slot_id
                                        )
                                else:
                                    logger.error("Slot %s not created" % slot_id)

                            else:
                                logger.warning(
                                    "No RS Files within the slot: [%s, %s]"
                                    % (
                                        slots_ini[s].strftime("%H%M"),
                                        slots_end[s].strftime("%H%M"),
                                    )
                                )

                    """ PLOT SCC INPUT """
                    if np.logical_or.reduce((process == 0, process == 20)):
                        # not slot_in_local,
                        logger.info("Plot Input SCC")
                        plot_scc_input(
                            [scc_slot_fn]
                        )  # Se traga una lista de archivos en vez de uno. TODO: arreglar esto
                    else:
                        logger.info("Plot Input SCC already done")

                    """ SCC UPLOAD AND PROCESS """
                    if np.logical_or.reduce(
                        (process == 0, process == 1, process == 30)
                    ):
                        # not slot_in_scc,
                        """UPLOAD SLOT TO SCC SERVER"""
                        if (
                            slot_in_scc
                        ):  # Delete Slot in SCC if exists: DOES NOT WORK YET
                            try:
                                logger.info(
                                    "Start Delete Slot %s from SCC server" % slot_id
                                )
                                scc_obj.login(
                                    SCC_SERVER_SETTINGS["website_credentials"]
                                )
                                deleted = scc_obj.delete_measurement(slot_id)
                                scc_obj.logout()
                                if deleted:
                                    logger.info(
                                        "Slot %s deleted from SCC server" % slot_id
                                    )
                                else:
                                    logger.error(
                                        "Slot %s NOT deleted from SCC server" % slot_id
                                    )
                                    logger.error(
                                        "Slot %s must be deleted manually from SCC server"
                                        % slot_id
                                    )
                            except Exception as e:
                                logger.error(str(e))
                                logger.error("Slot %s not deleted from SCC server")
                        try:
                            # Wrap for skip in case of upload takes longer than MAX_EXEC_TIME
                            logger.info("Start Upload Slot %s" % slot_id)
                            # old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                            # signal.alarm(MAX_EXEC_TIME)
                            if os.path.isfile(scc_slot_fn):
                                scc_obj.login(
                                    SCC_SERVER_SETTINGS["website_credentials"]
                                )
                                scc_obj.upload_file(scc_slot_fn, scc_id)
                                meas_obj, _ = scc_obj.get_measurement(slot_id)
                                scc_obj.logout()
                                if meas_obj is not None:
                                    logger.info("End Upload Slot %s" % slot_id)
                                else:
                                    logger.info("Slot %s not uploaded" % slot_id)
                            else:
                                logger.error(
                                    "Slot %s cannot be uploaded because it does not exist"
                                    % slot_id
                                )
                        except TimeoutException:
                            logger.error(
                                "Upload file took longer than %d s" % SCC_INFO["MAX_EXEC_TIME"]
                            )  # TODO: include MAX_EXEC_TIME in info.yml
                        except Exception as e:
                            logger.error(str(e))
                            logger.error("Slot %s not uploaded" % slot_id)
                        # finally:
                        # Turn off timer, Restore handler to previous value
                        # signal.alarm(0)
                        # signal.signal(signal.SIGALRM, old_handler)

                        """ Check, afterall, if Slot is at SCC """
                        scc_obj.login(SCC_SERVER_SETTINGS["website_credentials"])
                        meas_obj, _ = scc_obj.get_measurement(slot_id)
                        scc_obj.logout()
                        if meas_obj is None:
                            logger.error(
                                "Slot %s has not been uploaded to SCC server" % slot_id
                            )
                    else:
                        logger.info("Slot %s already in SCC" % slot_id)

                    scc_output_inlocal = check_scc_output_inlocal(scc_output_slot_dn)

                    """ PROCESS AND DOWNLOAD SCC-PROCESSED """
                    if np.logical_or.reduce(
                        (
                            process == 0,
                            process == 1,
                            process == 2,
                            process == 40,
                            process == 50,
                        )
                    ):
                        # not scc_output_inlocal,
                        # not slot_in_local, not slot_in_scc,
                        try:
                            # Wrap for skip in case of upload takes longer than MAX_EXEC_TIME
                            # old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                            # signal.alarm(MAX_EXEC_TIME)

                            # connect to scc
                            scc_obj.login(SCC_SERVER_SETTINGS["website_credentials"])

                            # get measurement object
                            meas_obj, status = scc_obj.get_measurement(slot_id)

                            if meas_obj:
                                """PROCESSING"""
                                if np.logical_or(process == 2, process == 40):
                                    logger.info("Start Processing Slot %s" % slot_id)
                                    request = scc_obj.session.get(
                                        meas_obj.rerun_all_url, stream=True
                                    )
                                    if request.status_code != 200:
                                        logger.error(
                                            "Could not rerun pre processing for %s. Status code: %s"
                                            % (slot_id, request.status_code)
                                        )
                                """ DOWNLOADING """
                                logger.info(
                                    "Start Download Processed Slot %s" % slot_id
                                )
                                meas_obj = scc_obj.monitor_processing(slot_id)
                                logger.info("End Download Processed Slot %s" % slot_id)
                            else:
                                logger.error("Measurement Slot %s not found." % slot_id)
                            scc_obj.logout()
                        except TimeoutException:
                            logger.error(
                                "Upload file took longer than %d s" % SCC_INFO["MAX_EXEC_TIME"]
                            )
                        except Exception as e:
                            logger.error(str(e))
                            logger.error("Slot %s not downloaded" % slot_id)
                        # finally:
                        # Turn off timer, Restore handler to previous value
                        # signal.alarm(0)
                    else:
                        logger.info(
                            "Slot %s already downloaded from SCC server" % slot_id
                        )

                    """ PLOT OUTPUT SCC SERVER """
                    if np.logical_or.reduce(
                        (process == 0, process == 1, process == 2, process == 60)
                    ):
                        # not slot_in_local, not slot_in_scc,
                        #                    not scc_output_inlocal,
                        logger.info("Plot Output SCC")
                        plot_scc_output(scc_output_slot_dn, scc_code=scc_id)
                    else:
                        logger.info("Plot Output SCC already done")
                else:
                    logger.warning(
                        "Too Early to process Slot %s: %s"
                        % (slot_id, now_dt.strftime("%Y%m%d_%H%M"))
                    )
                    logger.warning(
                        "Slot %s ends at %s"
                        % (slot_id, slots_end[s].strftime("%Y%m%d_%H%M"))
                    )
        else:
            msg = "No Measurement Files found."
            logger.warning(msg)
            return False, msg

    logger.info("End Process day: %s" % date_str)
    if msg == "":
        status = True
    else:
        status = False
    return status, msg


def process_scc(
    lidar_name: str = "alhambra",
    measuremente_type: str = "RS",
    scc_config_id: int = 729,
    raw_dir: Path | str | None = None,
    scc_dir: Path | str | None = None,
    campaign_cfg_fn: str | Path | None = None,
    **kwargs,
):
    """[summary]

    Parameters
    ----------

    Returns
    -------

    """

    """ Start SCC Processing """
    logger.info("Start SCC")

    """Get Input Arguments"""
    if lidar_name not in LIDAR_INFO["lidars"]:
        raise ValueError(f"Lidar {lidar_name} does not registered in lidarpy.")

    # Measurement Type
    if measuremente_type not in MEASUREMENT_TYPES:
        raise ValueError(
            f"Measurement Type {measuremente_type} does not registered in lidarpy. Exit program."
        )

    if raw_dir is None:
        raw_dir = Path.cwd()
    elif isinstance(raw_dir, str):
        raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory {raw_dir} does not exist")

    if scc_dir is None:
        scc_dir = Path.cwd()
    elif isinstance(scc_dir, str):
        scc_dir = Path(scc_dir)

    if not scc_dir.exists():
        raise FileNotFoundError(f"SCC data directory {scc_dir} does not exist")

    # Campaign Config
    if campaign_cfg_fn is None:
        # SCC lidar config Necessary if No Campaign Config File is given
        if scc_config_id is not None:
            raise ValueError("Campaign config or SCC config ID must be given")
    else:
        if isinstance(campaign_cfg_fn, str):
            campaign_cfg_fn = Path(campaign_cfg_fn)
        if not campaign_cfg_fn.exists():
            raise FileNotFoundError(f"Campaign config {campaign_cfg_fn} does not exist")

    # Dates
    date_ini_str = kwargs.get("date_ini_str", None)
    if date_ini_str is None:
        date_ini_dt = dt.datetime.utcnow()
    else:
        date_ini_dt = dt.datetime.strptime(date_ini_str, "%Y%m%d")
    date_end_str = kwargs.get("date_end_str", None)
    if date_end_str is None:
        date_end_dt = date_ini_dt
    else:
        date_end_dt = dt.datetime.strptime(date_end_str, "%Y%m%d")

    # Start Hour:
    hour_ini = kwargs.get("hour_ini", None)

    # End Hour:
    hour_end = kwargs.get("hour_end", None)

    # Hour Resolution: If no campaign config file is given and different from default
    hour_resolution = kwargs.get("hour_resolution", None)
    if hour_resolution is None:
        hour_resolution = 1.0

    # Timestamp:
    timestamp = kwargs.get("timestamp", None)
    if timestamp is None:
        timestamp = 0

    # Slot Name Type:
    slot_name_type = kwargs.get("slot_name_type", None)
    if slot_name_type is None:
        slot_name_type = 1

    # Type of Processing
    process = kwargs.get("process", None)
    if process is None:
        process = 0

    # Mode of Processing: Operational/Offline
    mode = kwargs.get("mode", None)
    if mode is None:
        mode = 0
    if mode == 1:
        do_send_email = False
    else:
        do_send_email = True

    # SCC data format directory: [data_dn/LIDAR/SCCxxx] / user-defined
    logger.info("LIDAR: %s" % lidar_name)

    try:
        """Set SCC Campaign Configuration"""
        scc_campaign_config = get_campaign_config(
            campaign_cfg_fn=campaign_cfg_fn,
            scc_config_id=scc_config_id,
            hour_ini=hour_ini,
            hour_end=hour_end,
            hour_resolution=hour_resolution,
            timestamp=timestamp,
            slot_name_type=slot_name_type,
        )

        """ Process SCC workflow: loop along days """
        if date_end_dt.date() > date_ini_dt.date():
            logger.info("process period: %s - %s" % (date_ini_str, date_end_str))
            date_range = pd.date_range(date_ini_dt, date_end_dt)
            for i_date in date_range:
                i_date_str = i_date.strftime("%Y%m%d")
                result, msg_day = process_day(
                    lidar_name,
                    i_date_str,
                    meas_type=measuremente_type,
                    scc_campaign_config=scc_campaign_config,
                    process=process,
                    mode=mode,
                    raw_dir=raw_dir,
                    scc_dir=scc_dir,
                )
                if not result:
                    msg = "ERROR: Lidar %s. Day %s. %s " % (
                        lidar_name,
                        i_date_str,
                        msg_day,
                    )
                    if do_send_email:
                        send_email(email_sender, email_receiver, msg)
        elif date_end_dt.date() == date_ini_dt.date():
            logger.info("process single day")
            date_str = date_ini_dt.strftime("%Y%m%d")
            result, msg_day = process_day(
                lidar_name,
                date_str,
                meas_type=measuremente_type,
                scc_campaign_config=scc_campaign_config,
                process=process,
                mode=mode,
                raw_dir=raw_dir,
                scc_dir=scc_dir,
            )
            if not result:
                msg = "ERROR: Lidar %s. Day %s. %s " % (lidar_name, date_str, msg_day)
                if do_send_email:
                    send_email(email_sender, email_receiver, msg)
        else:
            msg = "Error in input dates. Exit program"
            logger.error(msg)
            if do_send_email:
                send_email(email_sender, email_receiver, msg)
            return
    except Exception as e:
        logger.error("%s. Exit program" % str(e))
        msg = "Exception Error in process_scc"
        logger.error(msg)
        if do_send_email:
            send_email(email_sender, email_receiver, msg)
        return


def parse_args():
    """Parse Input Arguments
    python -u scc.py -arg1 v1 -arg2 v2 ...

    Parameters
    ----------
    ()

    Returns
    -------
    args: dict
        Dictionary 'arg':value for input

    """
    # TODO: incluir input sobre resolucion de slot y timestamp

    logger.info("Parse Input")

    parser = argparse.ArgumentParser(description="usage %prog [arguments]")

    parser.add_argument(
        "-i",
        "--initial_date",
        action="store",
        dest="date_ini_str",
        help="Initial date [example: '20190131'].",
    )
    parser.add_argument(
        "-e",
        "--final_date",
        action="store",
        dest="date_end_str",  # required=True,
        help="Final date [example: '20190131'].",
    )
    parser.add_argument(
        "-l",
        "--lidar_name",
        action="store",
        dest="lidar_name",
        default="MULHACEN",
        help="Name of lidar system ['MULHACEN', 'VELETA']. Default: 'MULHACEN'.",
    )
    parser.add_argument(
        "-t",
        "--measurement_type",
        action="store",
        dest="meas_type",
        default="RS",
        help="Type of measurement [example: 'RS', 'HF'].",
    )
    parser.add_argument(
        "-c",
        "--campaign_cfg_fn",
        action="store",
        dest="campaign_cfg_fn",
        default="GFATserver",
        help="campaign config file in JSON format (full path), including name and scc lidar configurations.\
              Default: GFATserver means ACTRIS standardized format. \
              File must include the following fields: \
              name: str",
    )
    parser.add_argument(
        "-s",
        "--scc_lidar_cfg",
        type=int,
        action="store",
        dest="scc_lidar_cfg",
        help="SCC lidar configuration [example: 436, 403]\
              IF NO CAMPAIGN CONFIG FILE IS GIVEN.",
    )
    parser.add_argument(
        "-hi",
        "--hour_ini",
        type=float,
        action="store",
        dest="hour_ini",
        help="Start Hour (HH.H) for creation of slots.\
              DEFAULT: First Time of Measurement within the day",
    )
    parser.add_argument(
        "-he",
        "--hour_end",
        type=float,
        action="store",
        dest="hour_end",
        help="End Hour (HH.H) for creation of slots.\
              DEFAULT: Last Time of Measurement within the day",
    )
    parser.add_argument(
        "-r",
        "--hour_resolution",
        action="store",
        type=float,
        dest="hour_resolution",
        default=1,
        help="time resolution measurement slot. Lidar measurements will be \
              splitted in time slots in number of hours (default: 1 hour) \
              IF NO CAMPAIGN CONFIG FILE IS GIVEN.",
    )
    parser.add_argument(
        "-ts",
        "--timestamp",
        type=int,
        action="store",
        dest="timestamp",
        help="Timestamp for slot: 0, beginning of interval; 1: center of interval.\
              DEFAULT: 0",
    )
    parser.add_argument(
        "-snt",
        "--slot_name_type",
        type=int,
        action="store",
        dest="slot_name_type",
        help="Type of slot naming:\
                0: Earlinet:  YYYYMMDD+station+slot_number.\
                1: SCC Campaigns: YYYYMMDD+station+HHMM(Timestamp).\
                2: Alternative: YYYYMMDDHHMM(Timestamp)+station+scc.\
                DEFAULT: 1",
    )
    parser.add_argument(
        "-p",
        "--process",
        action="store",
        type=int,
        dest="process",
        default=0,
        help="what steps to process [-1: whole chain skipping steps if possible, \
                                0 (DEFAULT): whole chain forcing to process all steps, \
                                1: skip 0a->scc format, plot input. do: upload, process, download, plot output\
                                2: skip 0a->scc, plot input, upload to scc. do: process, download, plot_output\
                               10: only 0a->scc format, \
                               20: only plot input scc format, \
                               30: only upload&process to scc, \
                               40: only process&download from scc, \
                               50: only download from scc, \
                               60: only plot downloaded from scc",
    )
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        type=int,
        dest="mode",
        default=0,
        help="[0: operational (real-time. checks if the slot period has ended before processing it), 1: off-line",
    )
    parser.add_argument(
        "-d",
        "--data_dn",
        action="store",
        dest="data_dn",
        default="GFATserver",
        help="disk directory for all data",
    )
    parser.add_argument(
        "-o",
        "--scc_dn",
        action="store",
        dest="scc_dn",
        default="GFATserver",
        help="directory where scc files transformed from 0a are saved. \
              Default: 'GFATserver' means copy files to the GFAT NAS.",
    )

    args = parser.parse_args()

    return args.__dict__


def main():
    """
    To be called from terminal / shell:
    python -u scc.py -arg1 v1 -arg2 v2 ...

    """

    logger.info("Start SCC: %s" % dt.datetime.utcnow().strftime("%Y%m%d_%H%M"))

    """ parse args """
    args = parse_args()

    """ process scc workflow """
    process_scc(**args)

    logger.info("End SCC: %s" % dt.datetime.utcnow().strftime("%Y%m%d_%H%M"))


if __name__ == "__main__":
    """main"""
    main()
