""" Command line tool to convert Licel binary files to SCC NetCDF format.
"""
import argparse
import glob
import importlib
import logging
import os
from pdb import set_trace
import sys

from .licel_depol import LicelCalibrationMeasurement, DarkLicelCalibrationMeasurement
from .__init__ import __version__

logger = logging.getLogger(__name__)


def create_custom_class(custom_netcdf_parameter_path, use_id_as_name=False, temperature=25., pressure=1020.,
                        licel_timezone='UTC'):
    """ This funtion creates a custom LicelLidarMeasurement subclass,
    based on the input provided by the users.

    Parameters
    ----------
    custom_netcdf_parameter_path : str
       The path to the custom channels parameters.
    use_id_as_name : bool
       Defines if channels names are descriptive or transient digitizer IDs.
    temperature : float
       The ground temperature in degrees C (default 25.0).
    pressure : float
       The ground pressure in hPa (default: 1020.0).
    licel_timezone : str
       String describing the timezone according to the tz database.

    Returns
    -------
    CustomLidarMeasurement:
       A custom sub-class of LicelLidarMeasurement
    """
    # TODO: Remove the custom netcdf parameter artifact: pass parameters as optional input argument,
    # TODO: change setting format to YAML.
    custom_netcdf_parameters = read_settings_file(custom_netcdf_parameter_path)
    class CustomLidarMeasurement(LicelCalibrationMeasurement):
        extra_netcdf_parameters = custom_netcdf_parameters

        def __init__(self, plus45_files=None, minus45_files=None):
            super(CustomLidarMeasurement, self).__init__(plus45_files, minus45_files, use_id_as_name=use_id_as_name,
                                                         licel_timezone=licel_timezone)

        def set_PT(self):
            ''' Sets the pressure and temperature at station level. This is used if molecular_calc parameter is
            set to 0 (i.e. use US Standard atmosphere).

            The results are stored in the info dictionary.
            '''

            self.info['Temperature'] = temperature
            self.info['Pressure'] = pressure

    return CustomLidarMeasurement


def create_custom_dark_class(custom_netcdf_parameter_path, use_id_as_name=False, temperature=25., pressure=1020.,
                        licel_timezone='UTC'):
    """ This funtion creates a custom LicelLidarMeasurement subclass,
    based on the input provided by the users.

    Parameters
    ----------
    custom_netcdf_parameter_path : str
       The path to the custom channels parameters.
    use_id_as_name : bool
       Defines if channels names are descriptive or transient digitizer IDs.
    temperature : float
       The ground temperature in degrees C (default 25.0).
    pressure : float
       The ground pressure in hPa (default: 1020.0).
    licel_timezone : str
       String describing the timezone according to the tz database.

    Returns
    -------
    CustomLidarMeasurement:
       A custom sub-class of LicelLidarMeasurement
    """
    # TODO: Remove the custom netcdf parameter artifact: pass parameters as optional input argument,
    # TODO: change setting format to YAML.
    custom_netcdf_parameters = read_settings_file(custom_netcdf_parameter_path)

    class CustomLidarMeasurement(DarkLicelCalibrationMeasurement):
        extra_netcdf_parameters = custom_netcdf_parameters

        def __init__(self, dark_files=None):
            super(CustomLidarMeasurement, self).__init__(dark_files, use_id_as_name=use_id_as_name,
                                                         licel_timezone=licel_timezone)

        def set_PT(self):
            ''' Sets the pressure and temperature at station level. This is used if molecular_calc parameter is
            set to 0 (i.e. use US Standard atmosphere).

            The results are stored in the info dictionary.
            '''

            self.info['Temperature'] = temperature
            self.info['Pressure'] = pressure

    return CustomLidarMeasurement


def read_settings_file(settings_path):
    """ Read the settings file.

    The file should contain python code."""
    if not os.path.isfile(settings_path):
        logging.error("The provided settings path does not correspond to a file.")
        sys.exit(1)

    dirname, basename = os.path.split(settings_path)
    sys.path.append(dirname)

    module_name, _ = os.path.splitext(basename)
    settings = importlib.import_module(module_name)
    return settings


def main():
    # Define the command line argument
    parser = argparse.ArgumentParser(description="A program to convert Licel binary files from depolarization calibration measurements to the SCC NetCDF format.")
    parser.add_argument("parameter_file", help="The path to a parameter file linking licel and SCC channels.")
    parser.add_argument("plus45_string", help="Search string for plus 45 degree files")
    parser.add_argument("minus45_string", help="Search string for minus 45 degree files")
    parser.add_argument("-i", '--id_as_name',
                        help="Use transient digitizer ids as channel names, instead of descriptive names",
                        action="store_true")
    parser.add_argument("-m", "--measurement_id", help="The new measurement id", default=None)
    parser.add_argument("-n", "--measurement_number",
                        help="The measurement number for the date from 00 to 99. Used if no id is provided",
                        default="00")
    parser.add_argument("-t", "--temperature", type=float,
                        help="The temperature (in C) at lidar level, required if using US Standard atmosphere",
                        default="25")
    parser.add_argument("-p", "--pressure", type=float,
                        help="The pressure (in hPa) at lidar level, required if using US Standard atmosphere",
                        default="1020")
    parser.add_argument('--licel_timezone', help="String describing the timezone according to the tz database.",
                        default="UTC", dest="licel_timezone",
                       )
    # Verbosity settings from http://stackoverflow.com/a/20663028
    parser.add_argument('-d', '--debug', help="Print dubuging information.", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO,
                        )
    parser.add_argument('-s', '--silent', help="Show only warning and error messages.", action="store_const",
                        dest="loglevel", const=logging.WARNING
                        )
    parser.add_argument('-D', '--dark_measurements', help="Location of files containing dark measurements. Use relative path and filename wildcars, see 'files' parameter for example.",
                        default="", dest="dark_files"
                        )
    parser.add_argument('--version', help="Show current version.", action='store_true')

    # TODO: Add check that either measurement id or measurement number are provided
    args = parser.parse_args()

    # Get the logger with the appropriate level
    logging.basicConfig(format='%(levelname)s: %(message)s', level=args.loglevel)
    logger = logging.getLogger(__name__)

    #coloredlogs.install(fmt='%(levelname)s: %(message)s', level=args.loglevel)

    if args.version:
        print("Version: %s" % __version__)
        sys.exit(0)

    # Get a list of files to convert
    plus45_files = glob.glob(args.plus45_string)
    minus45_files = glob.glob(args.minus45_string)

    if len(plus45_files)==0 or len(minus45_files)==0:
        logger.error("No files found when searching for %s and %s." % (plus45_files, minus45_files))
        sys.exit(1)

    # Read the files
    logger.info("Reading {0} files from {1}".format(len(plus45_files), args.plus45_string))
    logger.info("Reading {0} files from {1}".format(len(minus45_files), args.minus45_string))

    CustomLidarMeasurement = create_custom_class(args.parameter_file, args.id_as_name, args.temperature,
                                                 args.pressure, args.licel_timezone)

    CustomDarkMeasurement = create_custom_dark_class(args.parameter_file, args.id_as_name, args.temperature,
                                                     args.pressure, args.licel_timezone)

    measurement = CustomLidarMeasurement(plus45_files, minus45_files)
    
    # Get a list of files containing dark measurements
    if args.dark_files != "":
        dark_files = glob.glob(args.dark_files)

        if dark_files:
            logger.debug("Using %s as dark measurements files!" % ', '.join(dark_files))
            measurement.dark_measurement = CustomDarkMeasurement(dark_files)
        else:
            logger.warning('No dark measurement files found when searching for %s. Will not use any dark measurements.' % args.dark_files)

    try:
        measurement = measurement.subset_by_scc_channels()
    except ValueError as err:
        logging.error(err)
        sys.exit(1)

    # Save the netcdf
    logger.info("Saving netcdf")
    measurement.set_measurement_id(args.measurement_id, args.measurement_number)
    measurement.save_as_SCC_netcdf()
    logger.info("Created file %s" % measurement.scc_filename)
