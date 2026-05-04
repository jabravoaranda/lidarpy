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
    "BT0": {
        "channel_ID": 2203,  # 1064fta
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 500.0,
    },
    "BT2": {
        "channel_ID": 2204,  # 532fta
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT3": {
        "channel_ID": 2215,  # 531fta
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },        
    # "BT4": {
    #     "channel_ID": 2213,  # 355fpa
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 1200,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },
    # "BT5": {
    #     "channel_ID": 2214,  # 355fsa
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 1200,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },        
    # "BT6": {
    #     "channel_ID": 2216,  # 354fta
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 1200,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },            
    "BT11": {
        "channel_ID": 2124,  # 532npa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 600,
        "Raw_Data_Range_Resolution": 3.75,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    "BT12": {
        "channel_ID": 2126,  # 532nsa
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 600,
        "Raw_Data_Range_Resolution": 3.75,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },
    # "BT13": {
    #     "channel_ID": 2128,  # 355npa
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 600,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },
    # "BT14": {
    #     "channel_ID": 2130,  # 355nsa
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 600,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },
    # "BT15": {
    #     "channel_ID": 2218,  # 387nta
    #     "Background_Low": 40000.0,
    #     "Background_High": 50000.0,
    #     "Laser_Shots": 600,
    #     "LR_Input": 1,
    #     "DAQ_Range": 100.0,
    # },    
    "BT16": {
        "channel_ID": 2219,  # 607nta
        "Background_Low": 40000.0,
        "Background_High": 50000.0,
        "Laser_Shots": 600,
        "Raw_Data_Range_Resolution": 3.75,
        "LR_Input": 1,
        "DAQ_Range": 100.0,
    },        
}
