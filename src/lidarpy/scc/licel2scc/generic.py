import datetime
import logging
from operator import itemgetter
import itertools
import collections
from pathlib import Path
from pdb import set_trace

import matplotlib as mpl
import netCDF4 as netcdf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

NETCDF_FORMAT = 'NETCDF4'  # choose one of 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC' and 'NETCDF4'

logger = logging.getLogger(__name__)


class BaseLidarMeasurement(object):
    """ 
    This class represents a general measurement object, independent of the input files.
    
    Each subclass should implement the following:
    * the _import_file method;
    * set the "extra_netcdf_parameters" variable to a dictionary that includes the appropriate parameters;
    
    You can override the set_PT method to define a custom procedure to get ground temperature and pressure.   
    
    The class assumes that the input files are consecutive, i.e. there are no measurements gaps.
    """
    extra_netcdf_parameters = None

    def __init__(self, file_list=None):
        """
        This is run when creating a new object.
        
        Parameters
        ----------
        file_list : list or str
           A list of the full paths to the input file(s).
        """
        self.info = {}
        self.dimensions = {}
        self.variables = {}
        self.channels = collections.OrderedDict()
        self.attributes = {}
        self.files = []
        self.dark_measurement = None

        if file_list:
            self._import_files(file_list)

    def _import_files(self, file_list):
        """
        Imports a list of files, and updates the object parameters.
        
        Parameters
        ----------
        file_list : list
           A list of the full paths to the input file. 
        """
        for f in file_list:
            self._import_file(f)
        self.update()

    def _import_file(self, filename):
        """
        Reads a single lidar file.
        
        This method should be overwritten at all subclasses.
        
        Parameters
        ----------
        filename : str
           Path to the lidar file.
        """
        raise NotImplementedError('Importing files should be defined in the instrument-specific subclass.')

    def update(self):
        """
        Update the info dictionary, variables, and dimensions of the measurement object
        based on the information found in the channels objects.
        
        Notes
        -----
        Reading of the scan_angles parameter is not implemented.
        """
        # Initialize
        start_time = []
        stop_time = []
        points = []
        all_time_scales = []
        channel_names = []

        # Get the information from all the channels
        for channel_name, channel in self.channels.items():
            channel.update()
            start_time.append(channel.start_time)
            stop_time.append(channel.stop_time)
            points.append(channel.points)
            all_time_scales.append(channel.time)
            channel_names.append(channel_name)

        # Find the unique time scales used in the channels
        time_scales = set(all_time_scales)

        # Update the info dictionary
        self.info['start_time'] = min(start_time)
        self.info['stop_time'] = max(stop_time)
        self.info['duration'] = self.info['stop_time'] - self.info['start_time']

        # Update the dimensions dictionary
        self.dimensions['points'] = max(points)
        self.dimensions['channels'] = len(self.channels)
        # self.dimensions['scan angles'] = 1
        self.dimensions['nb_of_time_scales'] = len(time_scales)

        # Make a dictionaries to match time scales and channels
        channel_timescales = {}
        timescale_channels = dict((ts, []) for ts in time_scales)
        for (channel_name, current_time_scale) in zip(channel_names, all_time_scales):
            for (ts, n) in zip(time_scales, list(range(len(time_scales)))):
                if current_time_scale == ts:
                    channel_timescales[channel_name] = n
                    timescale_channels[ts].append(channel_name)

        self.variables['id_timescale'] = channel_timescales

        # Update the variables dictionary
        # Write time scales in seconds
        raw_data_start_time = []
        raw_data_stop_time = []

        for current_time_scale in list(time_scales):
            raw_start_time = np.array(current_time_scale) - min(start_time)  # Time since start_time
            raw_start_in_seconds = np.array([t.seconds for t in raw_start_time])  # Convert in seconds
            raw_data_start_time.append(raw_start_in_seconds)  # And add to the list

            channel_name = timescale_channels[current_time_scale][0]
            channel = self.channels[channel_name]

            # TODO: Define stop time for each measurement based on real data
            raw_stop_in_seconds = raw_start_in_seconds + channel.get_duration()
            raw_data_stop_time.append(raw_stop_in_seconds)

        self.variables['Raw_Data_Start_Time'] = raw_data_start_time
        self.variables['Raw_Data_Stop_Time'] = raw_data_stop_time

    def subset_by_channels(self, channel_subset):
        """ 
        Create a measurement object containing only a subset of  channels. 
        
        Parameters
        ----------
        channel_subset : list
           A list of channel names (str) to be included in the new measurement object.
        
        Returns
        -------
        m : BaseLidarMeasurements object
           A new measurements object
        """

        m = self.__class__()  # Create an object of the same type as this one.
        m.channels = dict([(channel, self.channels[channel]) for channel
                           in channel_subset])
        m.files = self.files

        m.update()

        # Dark measurements should also be subsetted.
        if self.dark_measurement is not None:
            dark_subset = self.dark_measurement.subset_by_channels(channel_subset)
            m.dark_measurement = dark_subset

        return m

    def subset_by_scc_channels(self):
        """
        Subset the measurement based on the channels provided in the 
        extra_netecdf_parameter file.
        
        Returns
        -------
        m : BaseLidarMeasurements object
           A new measurements object
        """
        if self.extra_netcdf_parameters is None:
            raise RuntimeError("Extra netCDF parameters not defined, cannot subset measurement.")

        scc_channels = list(self.extra_netcdf_parameters.channel_parameters.keys())
        common_channels = list(set(scc_channels).intersection(list(self.channels.keys())))

        if not common_channels:
            logger.debug("Config channels: %s." % ','.join(scc_channels))
            logger.debug("Measurement channels: %s." % ','.join(list(self.channels.keys())))
            raise ValueError('No common channels between measurement and configuration files.')

        return self.subset_by_channels(common_channels)

    def subset_by_time(self, start_time, stop_time):
        """
        Subset the measurement for a specific time interval
        
        Parameters
        ----------
        start_time : datetime 
           The starting datetime to subset.
        stop_time : datetime
           The stopping datetime to subset.

        Returns
        -------
        m : BaseLidarMeasurements object
           A new measurements object
        """

        if start_time > stop_time:
            raise ValueError('Stop time should be after start time')

        if (start_time < self.info['start_time']) or (stop_time > self.info['stop_time']):
            raise ValueError('The time interval specified is not part of the measurement')

        m = self.__class__()  # Create an object of the same type as this one.
        for channel_name, channel in self.channels.items():
            m.channels[channel_name] = channel.subset_by_time(start_time, stop_time)

        m.update()

        # Transfer dark measurement to the new object. They don't need time subsetting.
        m.dark_measurement = self.dark_measurement
        return m

    def subset_by_bins(self, b_min=0, b_max=None):
        """
        Remove some height bins from the measurement data. 
        
        This could be needed to remove acquisition artifacts at 
        the first or last bins of the profiles.
        
        Parameters
        ----------
        b_min : int
          The fist valid data bin
        b_max : int or None
          The last valid data bin. If equal to None, all bins are used.
          
        Returns
        -------
        m : BaseLidarMeasurements object
           A new measurements object
        """
        m = self.__class__()  # Create an object of the same type as this one.

        for channel_name, channel in self.channels.items():
            m.channels[channel_name] = channel.subset_by_bins(b_min, b_max)

        m.update()

        # Dark measurements should also be subseted.
        if self.dark_measurement is not None:
            dark_subset = self.dark_measurement.subset_by_bins(b_min, b_max)
            m.dark_measurement = dark_subset

        return m

    def rename_channels(self, prefix="", suffix=""):
        """ Add a prefix and a suffix to all channel name.
        
        This is uses when processing Delta90 depolarization calibration measurements.
        
        Parameters
        ----------
        prefix : str
          The prefix to add to channel names. 
        suffix : str
          The suffix to add to channel names.
        """
        channel_names = list(self.channels.keys())

        for channel_name in channel_names:
            new_name = prefix + channel_name + suffix
            self.channels[new_name] = self.channels.pop(channel_name)

    def set_PT(self):
        """ 
        Sets the pressure and temperature at station level at the info dictionary .
        
        In this method, default values are used. It can be overwritten by subclasses
        to define more appropriate values for each system.
        """
        self.info['Temperature'] = 15.0  # Temperature in degC
        self.info['Pressure'] = 1013.15  # Pressure in hPa

    def subtract_dark(self):
        """
        Subtract dark measurements from the raw lidar signals. 

        This method is here just for testing.
        
        Notes
        -----
        This method should not be called if processing the data with the SCC. The SCC
        performs this operations anyway. 
        """
        if not self.dark_measurement:
            raise IOError('No dark measurements have been imported yet.')

        for (channel_name, dark_channel) in self.dark_measurement.channels.items():
            dark_profile = dark_channel.average_profile()
            channel = self.channels[channel_name]

            for measurement_time, data in channel.data.items():
                channel.data[measurement_time] = data - dark_profile

            channel.update()

    def set_measurement_id(self, measurement_id):
        """
        Sets the measurement id for the SCC file.

        Parameters
        ----------
        measurement_id: str           
        """

        self.info['Measurement_ID'] = measurement_id

    def save_as_SCC_netcdf(self, filename=None, output_dir=None):
        """Saves the measurement in the netCDF format as required by the SCC.
        
        If no filename is provided <measurement_id>.nc will be used. 
        
        Parameters
        ----------
        filename : str
           Output file name. If None, <measurement_id>.nc will be used. 
        """
        parameters = self.extra_netcdf_parameters

        #Management output path
        if output_dir is None:
            output_dir = Path.cwd()

        # Guess measurement ID if none is provided
        if 'Measurement_ID' not in self.info:
            self.set_measurement_id()

        # Check if temperature and pressure are defined
        for attribute in ['Temperature', 'Pressure']:
            stored_value = self.info.get(attribute, None)
            if stored_value is None:
                try:
                    self.set_PT()
                except:
                    raise ValueError('No value specified for %s' % attribute)

        if not filename:
            filename = "%s.nc" % self.info['Measurement_ID']

        filepath = output_dir / filename

        self.scc_filename = filename

        dimensions = {'points': 1,
                      'channels': 1,
                      'time': None,
                      'nb_of_time_scales': 1,
                      'scan_angles': 1, }  # Mandatory dimensions. Time bck not implemented

        global_attributes = {'Measurement_ID': None,
                             'RawData_Start_Date': None,
                             'RawData_Start_Time_UT': None,
                             'RawData_Stop_Time_UT': None,
                             'RawBck_Start_Date': None,
                             'RawBck_Start_Time_UT': None,
                             'RawBck_Stop_Time_UT': None,
                             'Sounding_File_Name': None,
                             'LR_File_Name': None,
                             'Overlap_File_Name': None,
                             'Location': None,
                             'System': None,
                             'Latitude_degrees_north': None,
                             'Longitude_degrees_east': None,
                             'Altitude_meter_asl': None,}

        channel_names = list(self.channels.keys())

        input_values = dict(self.dimensions, **self.variables)

        # Add some mandatory global attributes
        input_values['Measurement_ID'] = self.info['Measurement_ID']
        input_values['RawData_Start_Date'] = self.info['start_time'].strftime('%Y%m%d')
        input_values['RawData_Start_Time_UT'] = self.info['start_time'].strftime('%H%M%S')
        input_values['RawData_Stop_Time_UT'] = self.info['stop_time'].strftime('%H%M%S')

        # Add any global attributes provided by the subclass
        for attribute in self._get_custom_global_attributes():
            input_values[attribute["name"]] = attribute["value"]

        # Override global attributes, if provided in the settings file.
        for attribute_name in ['System', 'Latitude_degrees_north', 'Longitude_degrees_east', 'Altitude_meter_asl',
                               'Overlap_File_Name', 'LR_File_Name', 'Sounding_File_Name']:
            if attribute_name in parameters.general_parameters.keys():
                if attribute_name in input_values:
                    logger.info("Overriding {0} attribute, using the value provided in the parameter file.".format(
                        attribute_name))
                input_values[attribute_name] = parameters.general_parameters[attribute_name]
        # Open a netCDF file. The format is specified in the beginning of this module.
        with netcdf.Dataset(filepath, 'w', format=NETCDF_FORMAT) as f:

            # Create the dimensions in the file
            for (d, v) in dimensions.items():
                v = input_values.pop(d, v)
                f.createDimension(d, v)

            # Create global attributes
            for (attrib, value) in global_attributes.items():
                val = input_values.pop(attrib, value)
                if val:
                    setattr(f, attrib, val)
            
            # Variables
            # Write either channel_id or string_channel_id in the file
            first_channel_keys = list(parameters.channel_parameters.items())[0][1].keys()
            if "channel_ID" in first_channel_keys:
                channel_var = 'channel_ID'
                variable_type = 'i'
            elif "channel_string_ID" in first_channel_keys:
                channel_var = 'channel_string_ID'
                variable_type = str
            else:
                raise ValueError('Channel parameters should define either "chanel_id" or "channel_string_ID".')

            temp_v = f.createVariable(channel_var, variable_type, ('channels',))
            for n, channel in enumerate(channel_names):
                temp_v[n] = parameters.channel_parameters[channel][channel_var]

            # Write the custom subclass variables:
            for variable in self._get_custom_variables(channel_names):
                logger.debug("Saving variable {0}".format(variable['name']))
                temp_v = f.createVariable(variable["name"], variable["type"], variable["dimensions"])
                temp_v[:] = variable["values"]

            # Write the values of fixed channel parameters:            
            channel_variable_specs = self._get_scc_channel_variables()
            if 'First_Signal_Rangebin' not in channel_variable_specs.keys():
                channel_variable_specs['First_Signal_Rangebin'] = (('channels',), 'i')
            if 'Background_Mode' not in channel_variable_specs.keys():
                channel_variable_specs['Background_Mode'] = (('channels',), 'i')                
            provided_variable_names = self._get_provided_variable_names()

            for variable_name in provided_variable_names:
                # Check if the variable already defined (e.g. from values in Licel files).
                if variable_name in f.variables.keys():
                    logger.warning(
                        "Provided values of \"{0}\" were ignored because they were read from other source.".format(
                            variable_name))
                    continue

                if variable_name not in channel_variable_specs.keys():
                    logger.warning("Provided variable {0} is not parts of the specs: {1}".format(variable_name, list(channel_variable_specs.keys())))
                    continue

                temp_v = f.createVariable(variable_name,
                                          channel_variable_specs[variable_name][1],
                                          channel_variable_specs[variable_name][0])

                for n, channel_name in enumerate(channel_names):
                    try:
                        temp_v[n] = parameters.channel_parameters[channel_name][variable_name]
                    except KeyError:  # The parameter was not provided for this channel so we mask the value.
                        pass  # Default value should be a missing value  # TODO: Check this.

            # Write the id_timescale values
            temp_id_timescale = f.createVariable('id_timescale', 'i', ('channels',))
            for (channel, n) in zip(channel_names, list(range(len(channel_names)))):
                temp_id_timescale[n] = self.variables['id_timescale'][channel]

            # Laser pointing angle
            temp_v = f.createVariable('Laser_Pointing_Angle', 'd', ('scan_angles',))
            temp_v[:] = parameters.general_parameters['Laser_Pointing_Angle']

            # Molecular calculation
            temp_v = f.createVariable('Molecular_Calc', 'i')
            temp_v[:] = parameters.general_parameters['Molecular_Calc']

            # Laser pointing angles of profiles
            temp_v = f.createVariable('Laser_Pointing_Angle_of_Profiles', 'i', ('time', 'nb_of_time_scales'))
            for (time_scale, n) in zip(self.variables['Raw_Data_Start_Time'],
                                       list(range(len(self.variables['Raw_Data_Start_Time'])))):
                temp_v[:len(time_scale), n] = 0  # The lidar has only one laser pointing angle

            # Raw data start/stop time
            temp_raw_start = f.createVariable('Raw_Data_Start_Time', 'i', ('time', 'nb_of_time_scales'))
            temp_raw_stop = f.createVariable('Raw_Data_Stop_Time', 'i', ('time', 'nb_of_time_scales'))
            for (start_time, stop_time, n) in zip(self.variables['Raw_Data_Start_Time'],
                                                  self.variables['Raw_Data_Stop_Time'],
                                                  list(range(len(self.variables['Raw_Data_Start_Time'])))):
                temp_raw_start[:len(start_time), n] = start_time
                temp_raw_stop[:len(stop_time), n] = stop_time

            # Laser shots
            if "Laser_Shots" in provided_variable_names:
                if "Laser_Shots" in f.variables.keys():
                    logger.warning("Provided values of \"Laser_Shots\" were ignored because they were read from other source.")
                else:
                    temp_v = f.createVariable('Laser_Shots', 'i', ('time', 'channels'))
                    for (channel, n) in zip(channel_names, list(range(len(channel_names)))):
                        time_length = len(self.variables['Raw_Data_Start_Time'][self.variables['id_timescale'][channel]])
                        # Array slicing stopped working as usual ex. temp_v[:10] = 100 does not work. ??? np.ones was added.
                        temp_v[:time_length, n] = np.ones(time_length) * parameters.channel_parameters[channel][
                            'Laser_Shots']
            else:
                if "Laser_Shots" not in f.variables.keys():
                    logger.error("Mandatory variable \"Laser_Shots\" was not found in the settings or input files.")
                else:
                    pass  # Laser shots already in variables, so all good.

            # Raw lidar data
            temp_v = f.createVariable('Raw_Lidar_Data', 'd', ('time', 'channels', 'points'), zlib=True)
            for (channel, n) in zip(channel_names, list(range(len(channel_names)))):
                c = self.channels[channel]
                temp_v[:len(c.time), n, :c.points] = c.matrix

            self.add_dark_measurements_to_netcdf(f, channel_names)

            # Pressure at lidar station
            temp_v = f.createVariable('Pressure_at_Lidar_Station', 'd')
            temp_v[:] = self.info['Pressure']

            # Temperature at lidar station
            temp_v = f.createVariable('Temperature_at_Lidar_Station', 'd')
            temp_v[:] = self.info['Temperature']

            self.save_netcdf_extra(f)

    def _get_scc_channel_variables(self):        
        channel_variables = \
            {'Laser_Repetition_Rate': (('channels',), 'i'),
             'Scattering_Mechanism': (('channels',), 'i'),
             'Signal_Type': (('channels',), 'i'),
             'Emitted_Wavelength': (('channels',), 'd'),
             'Detected_Wavelength': (('channels',), 'd'),
             'Raw_Data_Range_Resolution': (('channels',), 'd'),
             'Background_Mode': (('channels',), 'i'),
             'Background_Low': (('channels',), 'd'),
             'Background_High': (('channels',), 'd'),
             'Dead_Time_Corr_Type': (('channels',), 'i'),
             'Dead_Time': (('channels',), 'd'),
             'Acquisition_Mode': (('channels',), 'i'),
             'Trigger_Delay': (('channels',), 'd'),
             'LR_Input': (('channels',), 'i'),
             'DAQ_Range': (('channels',), 'd'),
             'First_Signal_Rangebin': (('channels',), 'i'),
             }        
        return channel_variables

    def _get_provided_variable_names(self):
        """ Return a list of """
        # When looking for channel parameters, ignore the following parameter names:
        ignore = {'channel_ID', 'channel_string_ID', 'Depolarization_Factor', 'Laser_Shots'}  # Set

        channels = list(self.channels.keys())
        channel_parameters = self.extra_netcdf_parameters.channel_parameters

        # Get all the provided parameters (both mandatory and optional):
        all_provided_variables = [list(channel_parameters[channel].keys()) for channel in channels]
        provided_variables = set(itertools.chain.from_iterable(all_provided_variables))

        # Discard certain parameter names:
        provided_variables -= ignore

        return provided_variables

    def add_dark_measurements_to_netcdf(self, f, channels):
        """
        Adds dark measurement variables and properties to an open netCDF file.
        
        Parameters
        ----------
        f : netcdf Dataset
           A netCDF Dataset, open for writing.
        channels : list
           A list of channels names to consider when adding dark measurements.
        """
        # Get dark measurements. If it is not given in self.dark_measurement
        # try to get it using the get_dark_measurements method. If none is found
        # return without adding something.
        if self.dark_measurement is None:
            self.dark_measurement = self.get_dark_measurements()

        if self.dark_measurement is None:
            return

        dark_measurement = self.dark_measurement

        # Calculate the length of the time_bck dimensions
        number_of_profiles = [len(c.time) for c in dark_measurement.channels.values()]
        max_number_of_profiles = np.max(number_of_profiles)

        # Create the dimension
        f.createDimension('time_bck', max_number_of_profiles)

        # Save the dark measurement data
        temp_v = f.createVariable('Background_Profile', 'd', ('time_bck', 'channels', 'points'), zlib=True)
        for (channel, n) in zip(channels, list(range(len(channels)))):
            c = dark_measurement.channels[channel]
            temp_v[:len(c.time), n, :c.points] = c.matrix

        # Dark profile start/stop time
        temp_raw_start = f.createVariable('Raw_Bck_Start_Time', 'i', ('time_bck', 'nb_of_time_scales'))
        temp_raw_stop = f.createVariable('Raw_Bck_Stop_Time', 'i', ('time_bck', 'nb_of_time_scales'))
        for (start_time, stop_time, n) in zip(dark_measurement.variables['Raw_Data_Start_Time'],
                                              dark_measurement.variables['Raw_Data_Stop_Time'],
                                              list(range(len(dark_measurement.variables['Raw_Data_Start_Time'])))):
            temp_raw_start[:len(start_time), n] = start_time
            temp_raw_stop[:len(stop_time), n] = stop_time

        # Dark measurement start/stop time
        f.RawBck_Start_Date = dark_measurement.info['start_time'].strftime('%Y%m%d')
        f.RawBck_Start_Time_UT = dark_measurement.info['start_time'].strftime('%H%M%S')
        f.RawBck_Stop_Time_UT = dark_measurement.info['stop_time'].strftime('%H%M%S')

    def save_netcdf_extra(self, f):
        """ Save extra netCDF parameters to an open netCDF file. 
        
        If required, this method should be overwritten by subclasses of BaseLidarMeasurement.
        """
        pass

    def plot(self):
        for channel in self.channels:
            self.channels[channel].plot(show_plot=False)
        plt.show()

    def get_dark_measurements(self):
        return None

    def _get_custom_global_attributes(self):
        """
        Abstract method to provide NetCDF global attributes that should be included
        in the final NetCDF file.
        
        If required, this method should be implemented by subclasses of BaseLidarMeasurement.
        """
        return []

    def _get_custom_variables(self, channel_names=None):
        """
        Abstract method to provide custom NetCDF variables that should be included in the final NetCDF file.
        
        If required, this method should be implemented by subclasses of BaseLidarMeasurement.
        """
        return []


class LidarChannel(object):
    """ 
    This class represents a general measurement channel, independent of the input files.

    The class assumes that the input files are consecutive, i.e. there are no measurements gaps.
    """

    def __init__(self, channel_parameters):
        """
        This is run when creating a new object.
        
        The input dictionary should contain 'name', 'binwidth', and 'data' keys. 
        
        Parameters
        ----------
        channel_parameters : dict
           A dict containing channel parameters.
        """
        # TODO: Change channel_parameters to explicit keyword arguments?

        c = 299792458.  # Speed of light
        self.wavelength = channel_parameters['name']
        self.name = str(self.wavelength)
        self.binwidth = float(channel_parameters['binwidth'])  # in microseconds
        self.data = {}
        self.resolution = self.binwidth * c / 2
        self.z = np.arange(
            len(channel_parameters['data'])) * self.resolution + self.resolution / 2.0  # Change: add half bin in the z
        self.points = len(channel_parameters['data'])
        self.rc = []

    def get_duration(self):
        """ Get an array with the duration of each profile in seconds.

        If the duration property already exists, returns this.
        If not, it tries to estimate it based on the time difference of the profiles.

        Returns
        -------
        duration : ndarray
           1D array containing profile durations.
        """

        current_value = getattr(self, 'duration', None)

        if current_value:
            return np.array(current_value)

        time_diff = np.diff(self.time)
        durations = [dt.seconds for dt in time_diff]
        # Assume the last two profiles have the same duration
        duration = np.array(durations)
        return duration

    def calculate_rc(self, idx_min=-2000, idx_max=-500, first_signal_bin=0):
        """ Calculate range corrected signal.
        
        The background signal is estimated as the mean signal between idx_min and idx_max. 
        
        The calculations is stored in the self.rc parameters. 
        
        Parameters
        ----------
        idx_min : int
           Minimum index to calculate background signal.
        idx_max : int
           Maximum index to calculate background signal.
        first_signal_bin : int
           First bin with useful signal. Positive integer.
        """
        background = np.mean(self.matrix[:, idx_min:idx_max], axis=1)
        background_corrected = (self.matrix.transpose() - background).transpose()
        background_corrected = np.roll(background_corrected, -first_signal_bin, axis=1)
        background_corrected[:, -first_signal_bin:] = 0
        self.rc = background_corrected * (self.z ** 2)

    def noise_mask(self, idx_min=-2000, idx_max=-500, threshold=1.):
        """ Calculate points that are probably noise.

        To calculate this, we find the max value of the background region. Then we reject all points that
        are less than `threshold` times this value.
        Parameters
        ----------
        idx_min : int
           Minimum index to calculate background signal.
        idx_max : int
           Maximum index to calculate background signal.
        threshold : float
           Threshold value.
        """
        background_max = np.max(self.matrix[:, idx_min:idx_max], axis=1)
        background_mean = np.mean(self.matrix[:, idx_min:idx_max], axis=1)
        mean_to_max = background_max - background_mean

        noise_limit = background_mean + mean_to_max * threshold

        mask = self.matrix <= noise_limit[:, np.newaxis]

        return mask

    def update(self):
        """
        Update the time parameters and data according to the raw input data. 
        """
        self.start_time = min(self.data.keys())
        self.stop_time = max(self.data.keys()) + datetime.timedelta(seconds=self.duration[-1])
        self.time = tuple(sorted(self.data.keys()))
        sorted_data = sorted(iter(self.data.items()), key=itemgetter(0))
        self.matrix = np.array(list(map(itemgetter(1), sorted_data)))

    def _nearest_datetime(self, input_time):
        """
        Find the profile datetime and index that is nearest to the given datetime.
        
        Parameters
        ----------
        input_time : datetime.datetime
           Input datetime object.

        Returns
        -------
        profile_datetime : datetime
           The datetime of the selected profile.
        profile_idx : int
           The index of the selected profile.
        """
        max_duration = np.max(self.duration)

        margin = datetime.timedelta(seconds=max_duration * 5)

        if ((input_time + margin) < self.start_time) | ((input_time - margin) > self.stop_time):
            logger.error("Requested date not covered in this file")
            raise ValueError("Requested date not covered in this file")
        dt = np.abs(np.array(self.time) - input_time)

        profile_idx = np.argmin(dt)
        profile_datetime = self.time[profile_idx]

        dt_min = dt[profile_idx]
        if dt_min > datetime.timedelta(seconds=max_duration):
            logger.warning("Nearest profile more than %s seconds away. dt = %s." % (max_duration, dt_min))

        return profile_datetime, profile_idx

    def subset_by_time(self, start_time, stop_time):
        """
        Subset the channel for a specific time interval.

        Parameters
        ----------
        start_time : datetime 
           The starting datetime to subset.
        stop_time : datetime
           The stopping datetime to subset.

        Returns
        -------
        c : LidarChannel object
           A new channel object
        """
        time_array = np.array(self.time)
        condition = (time_array >= start_time) & (time_array <= stop_time)

        subset_time = time_array[condition]
        subset_data = dict([(c_time, self.data[c_time]) for c_time in subset_time])

        # Create a list with the values needed by channel's __init__()
        parameter_values = {'name': self.wavelength,
                            'binwidth': self.binwidth,
                            'data': subset_data[subset_time[0]], }

        # We should use __class__ instead of class name, so that this works properly
        # with subclasses
        # Eg: c = self.__class__(parameters_values)
        # This does not currently work with Licel files though
        # TODO: Fix this!
        c = LidarChannel(parameter_values)
        c.data = subset_data
        c.update()
        return c

    def subset_by_bins(self, b_min=0, b_max=None):
        """
        Remove some height bins from the channel data. 
        
        This could be needed to remove acquisition artifacts at 
        the first or last bins of the profiles.
        
        Parameters
        ----------
        b_min : int
          The fist valid data bin
        b_max : int or None
          The last valid data bin. If equal to None, all bins are used.
          
        Returns
        -------
        m : LidarChannel object
           A new channel object
        """

        subset_data = {}

        for ctime, cdata in self.data.items():
            subset_data[ctime] = cdata[b_min:b_max]

        # Create a list with the values needed by channel's __init__()
        parameters_values = {'name': self.wavelength,
                             'binwidth': self.binwidth,
                             'data': subset_data[
                                 list(subset_data.keys())[0]], }  # We just need any array with the correct length

        c = LidarChannel(parameters_values)
        c.data = subset_data
        c.update()
        return c

    def get_profile(self, input_datetime, range_corrected=True):
        """ Returns a single profile, that is nearest to the input datetime.
        
        Parameters
        ----------
        input_datetime : datetime
           The required datetime of the profile
        range_corrected : bool
           If True, the range corrected profile is returned. Else, normal signal is returned.
        
        Returns
        -------
        profile : ndarray
           The profile nearest to input_datetime.
        t : datetime
           The actual datetime corresponding to the profile.
        """
        t, idx = self._nearest_datetime(input_datetime)
        if range_corrected:
            data = self.rc
        else:
            data = self.matrix

        profile = data[idx, :][0]

        return profile, t

    def get_slice(self, start_time, stop_time, range_corrected=True):
        """ Returns a slice of data, between the provided start and stop time.

        Parameters
        ----------
        start_time : datetime
           The starting time of the slice
        stop_time : datetime
           The stoping time of the slice
        range_corrected : bool
           If True, the range corrected profile is returned. Else, normal signal is returned.

        Returns
        -------
        slice : ndarray
           The slice of profiles.
        slice_datetimes : ndarray of datetime
           The actual datetimes corresponding to the slice.
        """
        if range_corrected:
            data = self.rc
        else:
            data = self.matrix

        time_array = np.array(self.time)
        start_time = self._nearest_datetime(start_time)[0]
        stop_time = self._nearest_datetime(stop_time)[0]

        condition = (time_array >= start_time) & (time_array <= stop_time)

        slice = data[condition, :]
        slice_datetimes = time_array[condition]

        return slice, slice_datetimes

    def average_profile(self):
        """ Return the average profile (NOT range corrected) for all the duration of 
        the measurement. 
        
        Returns
        -------
        profile : ndarray
           The average profile for all data.
        """
        profile = np.mean(self.matrix, axis=0)
        return profile

    def plot(self, figsize=(8, 4), signal_type='rc', zoom=[0, 12000, 0, None], show_plot=True,
             cmap=plt.cm.jet, z0=None, title=None, vmin=0, vmax=1.3 * 10 ** 7):
        """
        Plot of the channel data.
        
        Parameters
        ----------
        figsize : tuple
           (width, height) of the output figure (inches)
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        zoom : list
           A four elemet list defined as [xmin, xmax, ymin, ymax]. Use ymin=0, ymax=-1 to plot full range.
        show_plot : bool
           If True, the show_plot command is run. 
        cmap : cmap
           An instance of a matplotlib colormap to use.
        z0 : float
           The ground-level altitude. If provided the plot shows altitude above sea level.
        title : str
           Optional title for the plot.
        vmin : float
           Minimum value for the color scale.
        vmax : float
           Maximum value for the color scale.
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        self.draw_plot_new(ax1, cmap=cmap, signal_type=signal_type, zoom=zoom, z0=z0, vmin=vmin, vmax=vmax)

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("%s signal - %s" % (signal_type.upper(), self.name))

        if show_plot:
            plt.show()

    def draw_plot(self, ax1, cmap=plt.cm.jet, signal_type='rc',
                  zoom=[0, 12000, 0, None], z0=None,
                  add_colorbar=True, cmap_label='a.u.', cb_format=None,
                  vmin=0, vmax=1.3 * 10 ** 7):
        """
        Draw channel data on the given axis.
        
        Parameters
        ----------
        ax1 : axis object
           The axis object to draw.
        cmap : cmap
           An instance of a matplotlib colormap to use.
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        zoom : list
           A four elemet list defined as [xmin, xmax, ymin, ymax]. Use ymin=0, ymax=-1 to plot full range.
        z0 : float
           The ground-level altitude. If provided the plot shows altitude above sea level.
        add_colorbar : bool
           If True, a colorbar will be added to the plot.
        cmap_label : str
           Label for the colorbar. Ignored if add_colorbar is False.
        cb_format : str
           Colorbar tick format string.
        vmin : float
           Minimum value for the color scale.
        vmax : float
           Maximum value for the color scale.
        """
        if signal_type == 'rc':
            if len(self.rc) == 0:
                self.calculate_rc()
            data = self.rc
        else:
            data = self.matrix

        hmax_idx = self._index_at_height(zoom[1])

        # If z0 is given, then the plot is a.s.l.
        if z0:
            ax1.set_ylabel('Altitude a.s.l. [km]')
        else:
            ax1.set_ylabel('Altitude a.g.l. [km]')
            z0 = 0

        ax1.set_xlabel('Time UTC [hh:mm]')
        # y axis in km, xaxis /2 to make 30s measurements in minutes. Only for 1064
        # dateFormatter = mpl.dates.DateFormatter('%H.%M')
        # hourlocator = mpl.dates.HourLocator()

        # dayFormatter = mpl.dates.DateFormatter('\n\n%d/%m')
        # daylocator = mpl.dates.DayLocator()
        hourFormatter = mpl.dates.DateFormatter('%H:%M')
        hourlocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)

        # ax1.axes.xaxis.set_major_formatter(dayFormatter)
        # ax1.axes.xaxis.set_major_locator(daylocator)
        ax1.axes.xaxis.set_major_formatter(hourFormatter)
        ax1.axes.xaxis.set_major_locator(hourlocator)

        ts1 = mpl.dates.date2num(self.start_time)
        ts2 = mpl.dates.date2num(self.stop_time)

        im1 = ax1.imshow(data.transpose()[zoom[0]:hmax_idx, zoom[2]:zoom[3]],
                         aspect='auto',
                         origin='lower',
                         cmap=cmap,
                         vmin=vmin,
                         # vmin = data[:,10:400].max() * 0.1,
                         vmax=vmax,
                         # vmax = data[:,10:400].max() * 0.9,
                         extent=[ts1, ts2, self.z[zoom[0]] / 1000.0 + z0 / 1000.,
                                 self.z[hmax_idx] / 1000.0 + z0 / 1000.],
                         )

        if add_colorbar:
            if cb_format:
                cb1 = plt.colorbar(im1, format=cb_format)
            else:
                cb1 = plt.colorbar(im1)
            cb1.ax.set_ylabel(cmap_label)

            # Make the ticks of the colorbar smaller, two points smaller than the default font size
            cb_font_size = mpl.rcParams['font.size'] - 2
            for ticklabels in cb1.ax.get_yticklabels():
                ticklabels.set_fontsize(cb_font_size)
            cb1.ax.yaxis.get_offset_text().set_fontsize(cb_font_size)

    def draw_plot_new(self, ax1, cmap=plt.cm.jet, signal_type='rc',
                      zoom=[0, 12000, 0, None], z0=None,
                      add_colorbar=True, cmap_label='a.u.',
                      cb_format=None, power_limits=(-2, 2),
                      date_labels=False,
                      vmin=0, vmax=1.3 * 10 ** 7):

        """ 
        Updated drawing routine, using pcolormesh. It is slower but provides more flexibility / precision
        in time plotting. The 2 drawing routines should be merged.
        
        Check draw_plot method for the meaning of the input arguments.
        """
        # TODO: Merge the two drawing routines.

        if signal_type == 'rc':
            if len(self.rc) == 0:
                self.calculate_rc()
            data = self.rc
        else:
            data = self.matrix

        self.draw_data(ax1, data, add_colorbar, cmap, cb_format, cmap_label, date_labels, power_limits, vmax, vmin,
                       z0, zoom)

    def draw_data(self, ax1, data, add_colorbar, cmap, cb_format, cmap_label, date_labels, power_limits, vmax, vmin,
                  z0, zoom):

        """
        This method plots generic lidar data in 2d Plot.

        Updated drawing routine, using pcolormesh. It is slower but provides more flexibility / precision
        in time plotting. The 2 drawing routines should be merged.


        """
        hmax_idx = self._index_at_height(zoom[1])
        hmin_idx = self._index_at_height(zoom[0])

        # If z0 is given, then the plot is a.s.l.
        if z0:
            ax1.set_ylabel('Altitude a.s.l. [km]')
        else:
            ax1.set_ylabel('Altitude a.g.l. [km]')
            z0 = 0
        ax1.set_xlabel('Time UTC [hh:mm]')

        # y axis in km, xaxis /2 to make 30s measurements in minutes. Only for 1064
        # dateFormatter = mpl.dates.DateFormatter('%H.%M')
        # hourlocator = mpl.dates.HourLocator()
        if date_labels:
            dayFormatter = mpl.dates.DateFormatter('%H:%M\n%d/%m/%Y')
            daylocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
            ax1.axes.xaxis.set_major_formatter(dayFormatter)
            ax1.axes.xaxis.set_major_locator(daylocator)
        else:
            hourFormatter = mpl.dates.DateFormatter('%H:%M')
            hourlocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=8, interval_multiples=True)
            ax1.axes.xaxis.set_major_formatter(hourFormatter)
            ax1.axes.xaxis.set_major_locator(hourlocator)

        # Get the values of the time axis
        dt = datetime.timedelta(seconds=self.duration[-1])

        time_cut = self.time[zoom[2]:zoom[3]]

        time_last = time_cut[-1] + dt  # The last element needed for pcolormesh
        time_all = time_cut + (time_last,)
        t_axis = mpl.dates.date2num(time_all)

        # Get the values of the z axis
        z_cut = self.z[hmin_idx:hmax_idx] - self.resolution / 2.
        z_last = z_cut[-1] + self.resolution
        z_axis = np.append(z_cut, z_last) / 1000. + z0 / 1000.  # Convert to km
        # Plot
        im1 = ax1.pcolormesh(t_axis, z_axis, data.T[hmin_idx:hmax_idx, zoom[2]:zoom[3]],
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax,
                             )
        if add_colorbar:
            if cb_format:
                cb1 = plt.colorbar(im1, format=cb_format, extend='both')
            else:
                cb_formatter = ScalarFormatter()
                cb_formatter.set_powerlimits(power_limits)
                cb1 = plt.colorbar(im1, format=cb_formatter, extend='both')
            cb1.ax.set_ylabel(cmap_label)

            # Make the ticks of the colorbar smaller, two points smaller than the default font size
            cb_font_size = mpl.rcParams['font.size'] - 2
            for ticklabels in cb1.ax.get_yticklabels():
                ticklabels.set_fontsize(cb_font_size)
            cb1.ax.yaxis.get_offset_text().set_fontsize(cb_font_size)

    def _index_at_height(self, height):
        """
        Get the altitude index nearest to the specified height.
        
        Parameters
        ----------
        height : float
           Height (m)

        Returns
        -------
        idx : int
           Index corresponding to the provided height.
        """
        idx = np.array(np.abs(self.z - height).argmin())
        if idx.size > 1:
            idx = idx[0]
        return idx
