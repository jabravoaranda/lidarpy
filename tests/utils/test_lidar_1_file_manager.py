#!/usr/bin/env python

import datetime

import numpy as np

from lidarpy.utils.file_manager import (
    channel2info,
    info2filename,
    info2path,
    search_dc,
    filename2info,
)


def test_channel_str2info():
    wave, telescope, polarization, detection = channel2info("532xpa")
    assert wave == 532
    assert telescope == "xf"
    assert polarization == "p"
    assert detection == "a"


def test_filename2info():
    fn_rs = "mhc_1a_Prs_rs_xf_20220808_1700.nc"
    (
        lidar_nick,
        data_level,
        measurement_type,
        signal_type,
        telescope,
        date,
    ) = filename2info(fn_rs)

    assert lidar_nick
    assert data_level
    assert measurement_type
    assert signal_type
    assert telescope
    assert date


def test_info2filename():
    fn_rs = info2filename(        
        datetime.datetime(2022, 8, 8, 17, 00, 00),
        lidar_nick="mhc",
        measurement_type="RS",                
        add_hour=True,
    )

    fn_dc = info2filename(        
        datetime.datetime(2022, 8, 8, 20, 1, 00),
        lidar_nick="mhc",
        measurement_type="DC",                
        add_hour=True,
    )
    assert fn_rs == "mhc_1a_Prs_rs_xf_20220808_1700.nc"    
    assert fn_dc == "mhc_1a_Pdc_rs_xf_20220808_2001.nc"


def test_info2path(tmp_path):
    fp_rs = info2path(
        lidar_name="mulhacen",        
        date=datetime.datetime(2022, 8, 8, 17, 00, 00),
        dir=tmp_path,
        measurement_type="RS",
        signal_type="rs",
        add_hour=True,
    )        
    assert fp_rs == tmp_path / "mulhacen/1a/2022/08/08/mhc_1a_Prs_rs_xf_20220808_1700.nc"

def test_search_dc(alhambra_rs_nc, alhambra_dc_nc):
    dc_path = search_dc(
        alhambra_rs_nc,
        np.array(
            ["2023-08-30T03:00:00.0", "2023-08-30T04:00:00.0"],
            dtype="M8",
        ),
    )

    assert dc_path == alhambra_dc_nc
