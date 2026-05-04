""" Command line tool to convert Licel binary files to SCC NetCDF format.
"""
import argparse
import glob
import importlib
import logging
import os
from pdb import set_trace
import sys

import yaml

from .licel import LicelLidarMeasurement
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
    logger.debug('Reading parameter files: %s' % custom_netcdf_parameter_path)
    custom_netcdf_parameters = read_settings_file(custom_netcdf_parameter_path)

    class CustomLidarMeasurement(LicelLidarMeasurement):
        extra_netcdf_parameters = custom_netcdf_parameters

        def __init__(self, file_list=None):
            super(CustomLidarMeasurement, self).__init__(file_list, use_id_as_name=use_id_as_name,
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


def read_cloudmask_settings_file(settings_file_path):
    """ Read the configuration file.

    The file should be in YAML syntax."""

    if not os.path.isfile(settings_file_path):
        logging.error("Wrong path for cloudmask settings file (%s)" % settings_file_path)
        sys.exit(1)

    with open(settings_file_path) as yaml_file:
        try:
            settings = yaml.load(yaml_file)
            logging.debug("Read cloudmask settings file(%s)" % settings_file_path)
        except:
            logging.error("Could not parse YAML file (%s)" % settings_file_path)
            sys.exit(1)

    return settings


def get_corrected_measurement_id(args, n):
    """ Correct the provided measurement id, in case of multiple cloud-free periods. """
    if args.measurement_id is not None:
        order = int(args.measurement_id[-2:])
        new_no = order + n
        measurement_id = args.measurement_id[:-2] + str(new_no)
        measurement_no = args.measurement_number  # The same
    else:
        measurement_no = str(int(args.measurement_number) + n).zfill(2)
        measurement_id = None

    return measurement_id, measurement_no


def convert_to_scc(CustomLidarMeasurement, files, dark_pattern, measurement_id, measurement_number=None, output_dir=None):
    """ Convert files to SCC. """
    measurement = CustomLidarMeasurement(files)
    # Get a list of files containing dark measurements
    if dark_pattern != "":
        dark_files = glob.glob(dark_pattern)


        if dark_files:
            logger.debug("Using %s as dark measurements files!" % ', '.join(dark_files))
            measurement.dark_measurement = CustomLidarMeasurement(dark_files)
        else:
            logger.warning(
                'No dark measurement files found when searching for %s. Will not use any dark measurements.' % dark_pattern)
    try:
        measurement = measurement.subset_by_scc_channels()
    except ValueError as err:
        logger.error(err)
        sys.exit(1)

    # Save the netcdf
    logger.info("Saving netcdf")
    measurement.set_measurement_id(measurement_id)
    measurement.save_as_SCC_netcdf(output_dir=output_dir)
    logger.info("Created file %s" % measurement.scc_filename)


def main():
    # Define the command line argument
    parser = argparse.ArgumentParser(description="A program to convert Licel binary files to the SCC NetCDF format.")
    parser.add_argument("parameter_file", help="The path to a parameter file linking licel and SCC channels.")
    parser.add_argument("files",
                        help="Search pattern of Licel files. Use filename wildcards. (default './*.*')",
                        default="./*.*")
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
    parser.add_argument('-D', '--dark_measurements',
                        help="Search pattern of files containing dark measurements. Use filename wildcars, see 'files' parameter for example.",
                        default="", dest="dark_files"
                        )
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
    parser.add_argument('--version', help="Show current version.", action='store_true')

    args = parser.parse_args()

    # Get the logger with the appropriate level
    logging.basicConfig(format='%(levelname)s: %(message)s', level=args.loglevel)
    logger = logging.getLogger(__name__)

    # coloredlogs.install(fmt='%(levelname)s: %(message)s', level=args.loglevel)

    # Check for version
    if args.version:
        print("Version: %s" % __version__)
        sys.exit(0)

    # Get a list of files to convert
    files = glob.glob(args.files)

    # If not files found, exit
    if len(files) == 0:
        logger.error("No files found when searching for %s." % args.files)
        sys.exit(1)

    # If everything OK, proceed
    logger.info("Found {0} files matching {1}".format(len(files), os.path.abspath(args.files)))
    CustomLidarMeasurement = create_custom_class(args.parameter_file, args.id_as_name, args.temperature,
                                                 args.pressure, args.licel_timezone)

    convert_to_scc(CustomLidarMeasurement, files, args.dark_files, args.measurement_id, args.measurement_number)
