# INSERT HERE THE SYSTEM PARAMETERS
general_parameters = {
    "System": "ALHAMBRA",
    "Laser_Pointing_Angle": 3,
    "Molecular_Calc": 0,  # Use ECMWF or US standard atmosphere
    "Latitude_degrees_north": 37.16,
    "Longitude_degrees_east": -3.60,
    "Altitude_meter_asl": 660.0,
    "Call sign": "gr",
}

# LINK YOUR LICEL CHANNELS TO SCC PARAMETERS. USE BT0, BC0 ETC AS NAMES (AS IN LICEL FILES).
channel_parameters = {
    "BT0": {
        "channel_ID": 2203,  # 1064fta
        "channel_string_ID": "1064fta",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,        
        "DAQ_Range": 500.0,
    },
    "BT2": {
        "channel_ID": 2204,  # 532fta
         "channel_string_ID": "532fta",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC2": {
        "channel_ID": 2232,  # 532ftp
        "channel_string_ID": "532ftp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT3": {
        "channel_ID": 2215,  # 531fta
        "channel_string_ID": "531fta",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC3": {
        "channel_ID": 2272,  # 531ftp
        "channel_string_ID": "531ftp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    
    "BT4": {
        "channel_ID": 2213,  # 355fpa
        "channel_string_ID": "355fpa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC4": {
        "channel_ID": 2233,  # 355fpp
        "channel_string_ID": "355fpp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT5": {
        "channel_ID": 2214,  # 355fsa
        "channel_string_ID": "355fsa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC5": {
        "channel_ID": 2234,  # 355fsp
        "channel_string_ID": "355fsp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT6": {
        "channel_ID": 2216,  # 354fta
        "channel_string_ID": "354fta",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC6": {
        "channel_ID": 2217,  # 354ftp
        "channel_string_ID": "354ftp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT10": {
        "channel_ID": 2123,  # 1064nta
        "channel_string_ID": "1064nta",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BT11": {
        "channel_ID": 2124,  # 532npa
        "channel_string_ID": "532npa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC11": {
        "channel_ID": 2268,  # 532npp
        "channel_string_ID": "532npp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT12": {
        "channel_ID": 2126,  # 532nsa
        "channel_string_ID": "532nsa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC12": {
        "channel_ID": 2269,  # 532nsp
        "channel_string_ID": "532nsp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT13": {
        "channel_ID": 2128,  # 355npa
        "channel_string_ID": "355npa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC13": {
        "channel_ID": 2270,  # 355npp
        "channel_string_ID": "355npp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },
    "BT14": {
        "channel_ID": 2130,  # 355nsa
        "channel_string_ID": "355nsa",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "DAQ_Range": 100.0,
    },
    "BC14": {
        "channel_ID": 2271,  # 355nsp
        "channel_string_ID": "355nsp",
        "Background_Low": 100.,
        "Background_High": 800.0,
        "Background_Mode": 0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "DAQ_Range": 100.0,
    },  
}