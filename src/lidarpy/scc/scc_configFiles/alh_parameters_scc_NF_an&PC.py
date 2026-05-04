# INSERT HERE THE SYSTEM PARAMETERS
general_parameters = {
    "System": "ALHAMBRA",
    "Laser_Pointing_Angle": 3,
    "Molecular_Calc": 0,  # Use US standard atmosphere
    "Latitude_degrees_north": 37.16,
    "Longitude_degrees_east": -3.60,
    "Altitude_meter_asl": 660.0,
    "Call sign": "gr",
}

# LINK YOUR LICEL CHANNELS TO SCC PARAMETERS. USE BT0, BC0 ETC AS NAMES (AS IN LICEL FILES).
channel_parameters = {
    "BT10": {
        "channel_ID": 2123,  # 1064fta
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT11": {
        "channel_ID": 2124,  # 532npa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BC11": {
        "channel_ID": 2125,  # 532npp
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT12": {
        "channel_ID": 2126,  # 532nsa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BC12": {
        "channel_ID": 2127,  # 532nsp
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT13": {
        "channel_ID": 2128,  # 355npa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BC13": {
        "channel_ID": 2129,  # 355npp
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT14": {
        "channel_ID": 2130,  # 355nsa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BC14": {
        "channel_ID": 2131,  # 355nsp
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
}
