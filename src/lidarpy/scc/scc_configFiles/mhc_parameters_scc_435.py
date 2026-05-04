"""
Config File for lidarpy.scc.licel2scc.licel2scc.py
licel2scc.create_custom_class(this_file) -> CustomLidarMeasurement
"""

""" general parameters """
general_parameters = {
    'System': '\'MULHACEN\'',
    'Laser_Pointing_Angle': 0,
    'Molecular_Calc': 0,
    'Call sign': "gr",
    }

"""  create dictionary channel_parameters """
# licel channels to be used in scc config linked to scc channels id
licel_channel_id = {
    'BT0': 1284,
    'BT1': 1283,
    'BT2': 1282,
    'BT3': 1045,
    'BC0': 1094,
    'BC3': 1048,
    'BC4': 1047,
}
channel_parameters = dict()
for k, v in licel_channel_id.items():
    if v == 1047 or v == 1048:
        channel_parameters[k] = {
            'channel_ID': v,
            'Background_Low': 75000,
            'Background_High': 105000,
            'Laser_Shots': 600,
            'LR_Input': 1,
            }
    else:
        channel_parameters[k] = {
            'channel_ID': v,
            'Background_Low': 75000,
            'Background_High': 105000,
            'LR_Input': 1,
            }

