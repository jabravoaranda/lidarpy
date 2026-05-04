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
    "BT4_p45": {
        "channel_ID": 2205,  # 355fpa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,        
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,
    },
    "BT4_m45": {
        "channel_ID": 2209,  # 355fpa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },

    "BT5_p45": {
        "channel_ID": 2207,  # 355fsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,
    },
    "BT5_m45": {
        "channel_ID": 2211,  # 355fsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },

    "BT11_p45": {
        "channel_ID": 2132,  # 532npa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },        
    
    "BT11_m45": {
        "channel_ID": 2136,  # 532npa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },

    "BT12_p45": {
        "channel_ID": 2134,  # 532nsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },        
    
    "BT12_m45": {
        "channel_ID": 2138,  # 532nsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },


    "BT13_p45": {
        "channel_ID": 2140,  # 355npa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,
    },
    "BT13_m45": {
        "channel_ID": 2144,  # 355npa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,        
    },
    "BT14_p45": {
        "channel_ID": 2142,  # 355nsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,
    },
    "BT14_m45": {
        "channel_ID": 2146,  # 355nsa
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 807,
        "Pol_Calib_Range_Min": 1000,
        "Pol_Calib_Range_Max": 2500,
    },


    "BC4_p45": {
        "channel_ID": 2206,  # 355fpp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,         
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,
    },
    "BC4_m45": {
        "channel_ID": 2210,  # 355fpp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },    

    "BC5_p45": {
        "channel_ID": 2208,  # 355fsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,         
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,
    },
    "BC5_m45": {
        "channel_ID": 2212,  # 355fsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 600,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },

    "BC11_p45": {
        "channel_ID": 2133,  # 532npp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,         
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,
    },
    "BC11_m45": {
        "channel_ID": 2137,  # 532npp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },
    "BC12_p45": {
        "channel_ID": 2135,  # 532nsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },
    "BC12_m45": {
        "channel_ID": 2139,  # 532nsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },       
    
    "BC13_p45": {
        "channel_ID": 2141,  # 355npp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },
    "BC13_m45": {
        "channel_ID": 2145,  # 355npp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,      
    },
    "BC14_p45": {
        "channel_ID": 2143,  # 355nsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },
    "BC14_m45": {
        "channel_ID": 2147,  # 355nsp
        "Background_Low": 100.0,
        "Background_High": 800.0,
        "Background_Mode": 1,
        "Laser_Shots": 1200,
        "LR_Input": 1,
        "First_Signal_Rangebin": 805,
        "Pol_Calib_Range_Min": 1500,
        "Pol_Calib_Range_Max": 3500,        
    },
}
