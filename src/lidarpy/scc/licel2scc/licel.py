import datetime
import logging
import copy
import os
import collections

import numpy as np
import pytz

from .generic import BaseLidarMeasurement, LidarChannel
from .diva import DivaConverterMixin

logger = logging.getLogger(__name__)

c = 299792458.0  # Speed of light


class LicelChannelData:
    """ A class representing a single channel found in a single Licel file."""

    def __init__(self, raw_info, raw_data, duration, use_id_as_name=False, channel_name=None):
        """
        This is run when creating a new object.

        Parameters
        ----------
        raw_info : dict
           A dictionary containing raw channel information.
        raw_data : dict
           An array with raw channel data.
        duration : float
           Duration of the file, in seconds
        use_id_as_name : bool
           If True, the transient digitizer name (e.g. BT0) is used as a channel
           name. If False, a more descriptive name is used (e.g. '01064.o_an').
        channel_name : str or None
           If provided, it will override the automatic generated channel name. It can be used if names are not unique.
        """
        self.raw_info = raw_info
        self.raw_data = raw_data
        self.duration = duration
        self.use_id_as_name = use_id_as_name
        self.channel_name_input = channel_name
        self._assign_properties()

    def _assign_properties(self):
        """ Assign properties """
        self.adcbits = int(self.raw_info['ADCbits'])
        self.active = int(self.raw_info['active'])
        self.analog_photon = self.raw_info['analog_photon']
        self.bin_width = float(self.raw_info['bin_width'])
        self.data_points = int(self.raw_info['number_of_datapoints'])
        self.hv = float(self.raw_info['HV'])
        self.id = self.raw_info['ID']
        self.laser_used = int(self.raw_info['laser_used'])
        self.number_of_shots = int(self.raw_info['number_of_shots'])
        self.wavelength_str = self.raw_info['wavelength']

        self.address = int(self.id[-1:], base=16)

        if self.is_analog:
            self.discriminator = float(self.raw_info['discriminator']) * 1000  # Analog range in mV
        else:
            self.discriminator = float(self.raw_info['discriminator'])

    @property
    def is_photodiode(self):
        return self.id[0:2] == 'PD'

    @property
    def wavelength(self):
        """ Property describing the nominal wavelength of the channel.

        Returns
        -------
        : int or None
           The integer value describing the wavelength. If no raw_info have been provided,
           returns None.
        """
        wavelength = self.wavelength_str.split('.')[0]
        return int(wavelength)

    @property
    def channel_name(self):
        """
        Construct the channel name adding analog photon info to avoid duplicates

        If use_id_as_name is True, the channel name will be the transient digitizer ID (e.g. BT01).
        This could be useful if the lidar system has multiple telescopes, so the descriptive name is
        not unique.

        Returns
        -------
        channel_name : str
           The channel name
        """
        if self.channel_name_input is not None:
            return self.channel_name_input

        if self.use_id_as_name:
            channel_name = self.id
        else:
            acquisition_type = self.analog_photon_string
            channel_name = "%s_%s" % (self.wavelength_str, acquisition_type)
        return channel_name

    @property
    def analog_photon_string(self):
        """ Convert the analog/photon flag found in the Licel file to a proper sting.

        Returns
        -------
        string : str
           'an' or 'ph' string, for analog or photon-counting channel respectively.
        """
        if self.analog_photon == '0':
            string = 'an'
        elif self.analog_photon == '1':
            string = 'ph'
        elif self.analog_photon == '2':
            string = 'std_an'
        elif self.analog_photon == '3':
            string = 'std_ph'
        else:
            string = str(self.analog_photon)
        return string

    def calculate_physical(self):
        """ Calculate physically-meaningful data from raw channel data:

        * In case of analog signals, the data are converted to mV.
        * In case of photon counting signals, data are stored as number of photons.

        In addition, some ancillary variables are also calculated (z, dz, number_of_bins).
        """

        norm = self.raw_data / float(self.number_of_shots)
        dz = self.bin_width

        if self.is_analog:
            # If the channel is in analog mode
            ADCrange = self.discriminator  # Discriminator value already in mV

            if self.is_photodiode and (self.adcbits == 0):
                logger.debug(
                    "Assuming adcbits equal 1. This is a bug in current licel format when storing photodiode data.")
                channel_data = norm * ADCrange / (2 ** self.adcbits)
            else:
                channel_data = norm * ADCrange / (
                            (2 ** self.adcbits) - 1)  # Licel LabView code has a bug (does not account -1).

        else:
            channel_data = norm * float(self.number_of_shots)
        # Calculate Z
        self.z = np.array([dz * bin_number + dz / 2.0 for bin_number in range(self.data_points)])
        self.dz = dz
        self.data = channel_data

    @property
    def is_analog(self):
        return self.analog_photon == '0'

    @property
    def laser_shots(self):
        """ Alias for number_of_shots """
        return self.number_of_shots


class LicelFile(object):
    """ A class representing a single binary Licel file. """

    licel_file_header_format = ['filename',
                                'start_date start_time end_date end_time altitude longitude latitude zenith_angle',
                                # Appart from Site that is read manually
                                'LS1 rate_1 LS2 rate_2 number_of_datasets', ]
    licel_file_channel_format = 'active analog_photon laser_used number_of_datapoints 1 HV bin_width wavelength d1 d2 d3 d4 ADCbits number_of_shots discriminator ID'

    channel_data_class = LicelChannelData

    # If True, it corrects the old Raymetrics convention of zenith angle definition (zenith = -90 degrees)
    fix_zenith_angle = False

    def __init__(self, file_path, use_id_as_name=False, get_name_by_order=False, licel_timezone="UTC", import_now=True):
        """
        This is run when creating a new object.

        Parameters
        ----------
        file_path : str
           The path to the Licel file.
        use_id_as_name : bool
           If True, the transient digitizer name (e.g. BT0) is used as a channel
           name. If False, a more descriptive name is used (e.g. '01064.o_an').
        get_name_by_order : bool
           If True, the channel name is given by the order of the channel in the file. In this case the
           `use_id_as_name` variable is ignored.
        licel_timezone : str
           The timezone of dates found in the Licel files. Should match the available
           timezones in the TZ database.
        import_now : bool
           If True, the header and data are read immediately. If not, the user has to call the
           corresponding methods directly. This is used to speed up reading files when only
           header information are required.
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        self.use_id_as_name = use_id_as_name
        self.get_name_by_order = get_name_by_order
        self.start_time = None
        self.stop_time = None
        self.licel_timezone = licel_timezone

        self.header_lines = []  # Store raw header lines, to be used in save_as_txt

        if import_now:
            self.import_file()
        else:
            self.import_header_only()

    def import_file(self):
        """ Read the header info and data of the Licel file.
        """
        channels = collections.OrderedDict()
        photodiodes = collections.OrderedDict()

        with open(self.file_path, 'rb') as f:

            self.read_header(f)

            # Check the complete header is read
            f.readline()

            # Import the data
            for channel_no, current_channel_info in enumerate(self.channel_info):
                raw_data = np.fromfile(f, 'i4', int(current_channel_info['number_of_datapoints']))
                a = np.fromfile(f, 'b', 1)
                b = np.fromfile(f, 'b', 1)

                if (a[0] != 13) | (b[0] != 10):
                    logger.warning("No end of line found after record. File could be corrupt: %s" % self.file_path)
                    logger.warning('a: {0}, b: {1}.'.format(a, b))

                if self.get_name_by_order:
                    channel_name = channel_no
                else:
                    channel_name = None

                channel = self.channel_data_class(current_channel_info, raw_data, self.duration(),
                                                  use_id_as_name=self.use_id_as_name, channel_name=channel_name)

                # Assign the channel either as normal channel or photodiode
                if channel.is_photodiode:
                    if channel.channel_name in photodiodes.keys():
                        # Check if current naming convention produces unique files
                        raise IOError('Trying to import two photodiodes with the same name')
                    photodiodes[channel.channel_name] = channel
                else:
                    if channel.channel_name in channels.keys():
                        # Check if current naming convention does not produce unique files
                        raise IOError('Trying to import two channels with the same name')
                    channels[channel.channel_name] = channel

        self.channels = channels
        self.photodiodes = photodiodes

        self._calculate_physical()

    def read_header(self, f):
        """ Read the header of an open Licel file.

        Parameters
        ----------
        f : file-like object
           An open file object.
        """
        # Read the first 3 lines of the header
        raw_info = {}
        channel_info = []

        # Read first line
        first_line = f.readline().decode().strip()
        raw_info['Filename'] = first_line
        self.header_lines.append(first_line)

        raw_info.update(self._read_second_header_line(f))

        raw_info.update(self._read_rest_of_header(f))

        # Update the object properties based on the raw info
        start_string = '%s %s' % (raw_info['start_date'], raw_info['start_time'])
        stop_string = '%s %s' % (raw_info['end_date'], raw_info['end_time'])
        date_format = '%d/%m/%Y %H:%M:%S'

        try:
            logger.debug('Creating timezone object %s' % self.licel_timezone)
            timezone = pytz.timezone(self.licel_timezone)
        except:
            raise ValueError("Cloud not create time zone object %s" % self.licel_timezone)

        # According to pytz docs, timezones do not work with default datetime constructor.
        local_start_time = timezone.localize(datetime.datetime.strptime(start_string, date_format))
        local_stop_time = timezone.localize(datetime.datetime.strptime(stop_string, date_format))

        # Only save UTC time.
        self.start_time = local_start_time.astimezone(pytz.utc)
        self.stop_time = local_stop_time.astimezone(pytz.utc)

        # Read the rest of the header.
        for c1 in range(int(raw_info['number_of_datasets'])):
            channel_line = f.readline().decode()
            channel_info.append(self.match_lines(channel_line, self.licel_file_channel_format))
            self.header_lines.append(channel_line.strip())

        self.raw_info = raw_info
        self.channel_info = channel_info

        self._assign_properties()

    def _assign_properties(self):
        """ Assign properties from the raw_info dictionary. """
        self.number_of_datasets = int(self.raw_info['number_of_datasets'])
        self.altitude = float(self.raw_info['altitude'])
        self.longitude = float(self.raw_info['longitude'])
        self.latitude = float(self.raw_info['latitude'])

        self.zenith_angle_raw = float(self.raw_info['zenith_angle'])

        logger.debug('Fix zenith angle? %s' % self.fix_zenith_angle)

        if self.fix_zenith_angle:
            logger.debug('Fixing zenith angle.')
            self.zenith_angle = self._correct_zenith_angle(self.zenith_angle_raw)
        else:
            self.zenith_angle = self.zenith_angle_raw

    @staticmethod
    def _correct_zenith_angle(zenith_angle):
        """ Correct zenith angle from Raymetrics convention (zenith = -90 degrees).

        Parameters
        ----------
        zenith_angle : float
           Zenith angle in Raymetrics convention.

        Returns
        -------
        corrected_angle : float
           Corrected zenith angle.
        """
        corrected_angle = 90 + zenith_angle
        return corrected_angle

    def _read_second_header_line(self, f):
        """ Read the second line of a licel file. """
        raw_info = {}

        second_line = f.readline().decode()
        self.header_lines.append(second_line.strip())
        # Many Licel files don't follow the licel standard. Specifically, the
        # measurement site is not always 8 characters, and can include white
        # spaces. For this, the site name is detect everything before the first
        # date. For efficiency, the first date is found by the first '/'.
        # e.g. assuming a string like 'Site name 01/01/2010 ...'

        site_name = second_line.split('/')[0][:-2]
        clean_site_name = site_name.strip()
        raw_info['site'] = clean_site_name
        self.site = clean_site_name

        raw_info.update(self.match_lines(second_line[len(clean_site_name) + 1:], self.licel_file_header_format[1]))
        return raw_info

    def _read_rest_of_header(self, f):
        """ Read the rest of the header lines, after line 2. """
        # Read third line
        third_line = f.readline().decode()
        self.header_lines.append(third_line.strip())

        raw_dict = self.match_lines(third_line, self.licel_file_header_format[2])
        return raw_dict

    def _calculate_physical(self):
        """ Calculate physical quantities from raw data for all channels in the file. """
        for channel in self.channels.values():
            channel.calculate_physical()

        for photodiode in self.photodiodes.values():
            photodiode.calculate_physical()

    def duration(self):
        """ Return the duration of the file.

        Returns
        -------
        : float
           The duration of the file in seconds.
        """
        dt = self.stop_time - self.start_time
        return dt.seconds

    def import_header_only(self):
        """ Import only the header lines, without reading the actual data."""
        with open(self.file_path, 'rb') as f:
            self.read_header(f)

    @property
    def has_photodiode(self):
        return len(self.photodiodes) != 0

    @staticmethod
    def match_lines(f1, f2):
        # TODO: Change this to regex?
        list1 = f1.split()
        list2 = f2.split()

        if len(list1) != len(list2):
            logging.debug("Channel parameter list has different length from LICEL specifications.")
            logging.debug("List 1: %s" % list1)
            logging.debug("List 2: %s" % list2)

        combined = list(zip(list2, list1))
        combined = dict(combined)
        return combined

    def save_as_csv(self, file_path=None, fill_value=-999):
        """ Save the Licel file in txt format.

        The format roughly follows the txt files created by Licel software. There are two main differences:
        a) Channel names are used as headers.
        b) Photon-counting data are given in shots, not in MHz.

        Parameters
        ----------
        file_path : str or None
           The output file path. If nan, the input file path is used with a .txt suffix.
        fill_value : float
           A fill value to be used in case of different length columns, e.g. when saving photodiode data.

        Returns
        -------
        file_path : str
           Returns the used file paths. This is useful when input file_path is None.
        """
        if file_path is None:
            file_path = self.file_path + ".csv"

        # Collect channel names and data
        column_names = []
        column_data = []

        for name, channel in self.channels.items():
            if channel.is_analog:
                column_name = "{0} (mV)".format(name)
            else:
                column_name = "{0} (counts)".format(name)

            column_names.append(column_name)
            column_data.append(channel.data)

        for name, photodiode in self.photodiodes.items():
            if 'PD' not in name:
                name = 'PD_' + name

            column_names.append(name)
            column_data.append(photodiode.data)

        column_data = self._common_length_array(column_data, fill_value)

        header_text = '\n'.join(self.header_lines) + '\n'
        column_header = ', '.join(column_names)

        np.savetxt(file_path, column_data.T,  fmt='%.4f', delimiter=',', header=header_text + column_header, comments='')

        return file_path

    @staticmethod
    def _common_length_array(array_list, fill_value):
        """ Make a signle array out of serveral 1D arrays with, possibly, different length"""

        lengths = [len(a) for a in array_list]

        if len(set(lengths)) == 1:
            output_array = np.array(array_list)
        else:
            dimensions = (len(lengths), max(lengths))
            output_array = np.ma.masked_all(dimensions)

            for n, array in enumerate(array_list):
                output_array[n, :len(array)] = array

            output_array.filled(fill_value)

        return output_array


class LicelChannel(LidarChannel):

    def __init__(self):
        self.name = None
        self.resolution = None
        self.points = None
        self.wavelength = None
        self.laser_used = None

        self.rc = []
        self.raw_info = []
        self.laser_shots = []
        self.duration = []
        self.discriminator = []
        self.hv = []
        self.data = {}

    def append_file(self, current_file, file_channel):
        """ Append file to the current object """

        self._assign_properties(current_file, file_channel)

        self.binwidth = self.resolution * 2. / c  # in seconds
        self.z = file_channel.z

        self.data[current_file.start_time] = file_channel.data
        self.raw_info.append(file_channel.raw_info)

        self.duration.append(file_channel.duration)
        self.laser_shots.append(file_channel.number_of_shots)
        self.discriminator.append(file_channel.discriminator)
        self.hv.append(file_channel.hv)

    @property
    def number_of_shots(self):
        """ Redundant, kept here for backward compatibility """
        return self.laser_shots

    def _assign_properties(self, current_file, file_channel):
        self._assign_unique_property('name', file_channel.channel_name)
        self._assign_unique_property('resolution', file_channel.dz)
        self._assign_unique_property('wavelength', file_channel.wavelength)
        self._assign_unique_property('points', file_channel.data_points)
        self._assign_unique_property('adcbits', file_channel.adcbits)
        self._assign_unique_property('active', file_channel.active)
        self._assign_unique_property('laser_used', file_channel.laser_used)
        self._assign_unique_property('analog_photon_string', file_channel.analog_photon_string)
        self._assign_unique_property('latitude', current_file.latitude)
        self._assign_unique_property('longitude', current_file.longitude)

    def _assign_unique_property(self, property_name, value):

        current_value = getattr(self, property_name, None)

        if current_value is None:
            setattr(self, property_name, value)
        else:
            if current_value != value:
                raise ValueError('Cannot combine channels with different values of {0}.'.format(property_name))

    @property
    def is_analog(self):
        return self.analog_photon_string == 'an'

    @property
    def is_photon_counting(self):
        return self.analog_photon_string == 'ph'

    def __unicode__(self):
        return "<Licel channel: %s>" % self.name

    def __str__(self):
        return str(self).encode('utf-8')


class PhotodiodeChannel(LicelChannel):

    def _assign_properties(self, current_channel, file_channel):
        """ In contrast with normal channels, don't check for constant points."""
        self._assign_unique_property('name', file_channel.channel_name)
        self._assign_unique_property('resolution', file_channel.dz)
        self._assign_unique_property('wavelength', file_channel.wavelength)
        self._assign_unique_property('adcbits', file_channel.adcbits)
        self._assign_unique_property('active', file_channel.active)
        self._assign_unique_property('laser_used', file_channel.laser_used)
        self._assign_unique_property('adcbits', file_channel.adcbits)
        self._assign_unique_property('analog_photon_string', file_channel.analog_photon_string)


class LicelLidarMeasurement(BaseLidarMeasurement):
    file_class = LicelFile
    channel_class = LicelChannel
    photodiode_class = PhotodiodeChannel

    def __init__(self, file_list=None, use_id_as_name=False, get_name_by_order=False, licel_timezone='UTC'):
        self.raw_info = {}  # Keep the raw info from the files
        self.durations = {}  # Keep the duration of the files
        self.laser_shots = []

        self.use_id_as_name = use_id_as_name
        self.get_name_by_order = get_name_by_order
        self.licel_timezone = licel_timezone
        self.photodiodes = collections.OrderedDict()

        super(LicelLidarMeasurement, self).__init__(file_list)

    def _import_file(self, filename):

        if filename in self.files:
            logger.warning("File has been imported already: %s" % filename)
        else:
            logger.debug('Importing file {0}'.format(filename))
            current_file = self.file_class(filename, use_id_as_name=self.use_id_as_name,
                                           get_name_by_order=self.get_name_by_order,
                                           licel_timezone=self.licel_timezone)
            self.raw_info[current_file.file_path] = current_file.raw_info
            self.durations[current_file.file_path] = current_file.duration()

            file_laser_shots = []

            self._create_or_append_channel(current_file)

            self.laser_shots.append(file_laser_shots)
            self.files.append(current_file.file_path)

    def _create_or_append_channel(self, current_file):

        for channel_name, channel in current_file.channels.items():
            if channel_name not in self.channels:
                self.channels[channel_name] = self.channel_class()
            self.channels[channel_name].append_file(current_file, channel)

        for photodiode_name, photodiode in current_file.photodiodes.items():
            if photodiode_name not in self.photodiodes:
                self.photodiodes[photodiode_name] = self.photodiode_class()
            self.photodiodes[photodiode_name].append_file(current_file, photodiode)

    def append(self, other):

        self.start_times.extend(other.start_times)
        self.stop_times.extend(other.stop_times)

        for channel_name, channel in self.channels.items():
            channel.append(other.channels[channel_name])

    def _get_duration(self, raw_start_in_seconds):
        """ Return the duration for a given time scale. If only a single
        file is imported, then this cannot be guessed from the time difference
        and the raw_info of the file are checked.
        """

        if len(raw_start_in_seconds) == 1:  # If only one file imported
            duration = next(iter(self.durations.values()))  # Get the first (and only) raw_info
            duration_sec = duration
        else:
            duration_sec = np.diff(raw_start_in_seconds)[0]

        return duration_sec

    def _get_custom_variables(self, channel_names):

        daq_ranges = np.ma.masked_all(len(channel_names))
        for n, channel_name in enumerate(channel_names):
            channel = self.channels[channel_name]
            if channel.is_analog:
                unique_values = list(set(channel.discriminator))
                if len(unique_values) > 1:
                    logger.warning(
                        'More than one discriminator levels for channel {0}: {1}'.format(channel_name, unique_values))
                daq_ranges[n] = unique_values[0]

        laser_shots = []
        for channel_name in channel_names:
            channel = self.channels[channel_name]
            laser_shots.append(channel.laser_shots)

        try:
            laser_shots = np.vstack(laser_shots).T
        except Exception as e:
            logger.error('Could not read laser shots as an array. Maybe files contain different number of channels?')
            raise e

        params = [{
            "name": "DAQ_Range",
            "dimensions": ('channels',),
            "type": 'd',
            "values": daq_ranges,
        }, {
            "name": "Laser_Shots",
            "dimensions": ('time', 'channels',),
            "type": 'i',
            "values": laser_shots,
        },
        ]

        return params

    def _get_custom_global_attributes(self):
        """
        NetCDF global attributes that should be included
        in the final NetCDF file.

        Currently the method assumes that all files in the measurement object have the same altitude, lat and lon
        properties.
        """
        logger.debug('Setting custom global attributes')
        logger.debug('raw_info keys: {0}'.format(self.raw_info.keys()))

        params = [{
            "name": "Altitude_meter_asl",
            "value": float(self.raw_info[self.files[0]]["altitude"])
        }, {
            "name": "Latitude_degrees_north",
            "value": float(self.raw_info[self.files[0]]["latitude"])
        }, {
            "name": "Longitude_degrees_east",
            "value": float(self.raw_info[self.files[0]]["longitude"])
        },
        ]

        return params

    def subset_by_channels(self, channel_subset):
        """
        Create a measurement object containing only a subset of  channels.

        This method overrides the parent method to add some licel-spefic parameters to the new object.

        Parameters
        ----------
        channel_subset : list
           A list of channel names (str) to be included in the new measurement object.

        Returns
        -------
        m : BaseLidarMeasurements object
           A new measurements object
        """
        new_measurement = super(LicelLidarMeasurement, self).subset_by_channels(channel_subset)

        new_measurement.raw_info = copy.deepcopy(self.raw_info)
        new_measurement.durations = copy.deepcopy(self.durations)
        new_measurement.laser_shots = copy.deepcopy(self.laser_shots)

        return new_measurement

    def subset_by_time(self, channel_subset):
        """
        Subsetting by time does not work yet with Licel files.

        This requires changes in generic.py
        """
        raise NotImplementedError("Subsetting by time, not yet implemented for Licel files.")

    def print_channels(self):
        """ Print the available channel information on the screen.
        """
        keys = sorted(self.channels.keys())

        print("Name  Wavelength  Mode  Resolution  Bins ")

        for key in keys:
            channel = self.channels[key]
            print("{0:<3}  {1:<10}  {2:<4}  {3:<10}  {4:<5}".format(channel.name, channel.wavelength,
                                                                    channel.analog_photon_string, channel.resolution,
                                                                    channel.points))


class LicelDivaLidarMeasurement(DivaConverterMixin, LicelLidarMeasurement):
    pass
