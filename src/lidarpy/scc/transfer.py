

from typing import Tuple
from lidarpy.scc import scc_access

def check_scc_connection(scc_server_settings) -> bool: 
    """ Check if SCC server is available. """

    scc_obj = scc_access.SCC(
        tuple(scc_server_settings["basic_credentials"]),
        None,
        scc_server_settings["base_url"],
    )

    connection = scc_obj.login(scc_server_settings["website_credentials"])
    check_connection = connection.status_code == 200
    if not check_connection:
        raise Exception("Connection to SCC server failed.")
    else:
        scc_obj.logout()

    return check_connection

def check_measurement_id_in_scc(scc_server_settings, measurement: str) -> Tuple[bool, scc_access.Measurement | None]:
    """ Check if a measurement is already in SCC. """

    if len(measurement) != 15:
        raise ValueError("Measurement ID must have 14 characters.")

    scc_obj = scc_access.SCC( tuple(scc_server_settings["basic_credentials"]), None, scc_server_settings["base_url"], )

    scc_obj.login(scc_server_settings["website_credentials"])
    
    try:
        meas_obj, _ = scc_obj.get_measurement(measurement)
        measurement_id_in_scc = meas_obj is not None        
    except Exception as e:
        meas_obj = None
        number = int(str(e).split("Status code ")[-1][:-1])
        scc_obj.logout()
        raise ValueError(f"Error code: {number}")
    
    scc_obj.logout()

    return measurement_id_in_scc, meas_obj
