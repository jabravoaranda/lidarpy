"""
Config File for lidarpy.scc.licel2scc.licel2scc.py
licel2scc.create_custom_class(this_file) -> CustomLidarMeasurement
"""

""" general parameters """
general_parameters = {
    'System': '\'MULHACEN\'',
    'Laser_Pointing_Angle': 0,
    'Molecular_Calc': 0,
#     'Latitude_degrees_north': 37.164,
#     'Longitude_degrees_east': -3.605,
#     'Altitude_meter_asl': 680.0,
    'Call sign': "gr"
    }

"""  create dictionary channel_parameters """
# licel channels to be used in scc config linked to scc channels id
licel_channel_id = {
    'BT0': 1091,
    'BT1': 1092,
    'BT2': 1049,
    'BT3': 1045,
    'BC0': 1094,
    'BC1': 1093,
    'BC2': 1090,
    'BC4': 1047,
}
channel_parameters = dict()
for k, v in licel_channel_id.items():
    channel_parameters[k] = {
        'channel_ID': v,
        'Background_Low': 75000,
        'Background_High': 105000,
        'LR_Input': 1,
        }

# Old fashioned
#channel_parameters = {
#    'BT0': {'channel_ID': 1091,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },
#    'BT1': {'channel_ID': 1092,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },
#    'BT2': {'channel_ID': 1049,
#             'Background_Low': 75000,
#             'Background_High': 105000,
#             'LR_Input': 1, },
#    'BT3': {'channel_ID': 1045,
#             'Background_Low': 75000,
#             'Background_High': 105000,
#             'LR_Input': 1, },               
#    'BC0': {'channel_ID': 1094,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },               
#    'BC1': {'channel_ID': 1093,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },               
#    'BC2': {'channel_ID': 1090,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },               
#    'BC4': {'channel_ID': 1047,
#            'Background_Low': 75000,
#            'Background_High': 105000,
#            'LR_Input': 1, },               
#    }
