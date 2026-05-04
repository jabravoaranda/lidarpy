import logging


import numpy as np

from .licel import LicelFile, LicelChannelData, LicelChannel, LicelLidarMeasurement, PhotodiodeChannel

logger = logging.getLogger(__name__)

c = 299792458.0  # Speed of light


class LicelChannelDataV2(LicelChannelData):
    """ A class representing a single channel found in a single Licel file."""

    def __init__(self, raw_info, raw_data, duration):
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
        """
        super(LicelChannelDataV2, self).__init__(raw_info=raw_info,
                         raw_data=raw_data,
                         duration=duration,
                         use_id_as_name=False)  # Obsolete

    def _assign_properties(self):
        """ Assign properties """
        laser_polarizations = {0: "none",
                               1: "vertical",
                               2: "horizontal",
                               3: "right circular",
                               4: "left circular"}

        self.laser_polarization_int = int(self.raw_info['laser_polarization'])
        self.laser_polarization_str = laser_polarizations[self.laser_polarization_int]

        self.bin_shift = self.raw_info['bin_shift']
        self.bin_shift_decimal_places = self.raw_info['bin_shift_decimal_places']
        super(LicelChannelDataV2, self)._assign_properties()

    @property
    def is_photodiode(self):
        return self.id[0:2] == 'PD'

    @property
    def is_powermeter(self):
        return self.id[0:2] == 'PM'

    @property
    def is_standard_deviation(self):
        return self.id[0:2] == 'S2'

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
        return "{0.id}_L{0.laser_used}_P{0.laser_polarization_int}".format(self)

    def calculate_physical(self):
        """ Calculate physically-meaningful data from raw channel data:

        * In case of analog signals, the data are converted to mV.
        * In case of photon counting signals, data are stored as number of photons.

        In addition, some ancillary variables are also calculated (z, dz, number_of_bins).
        """
        data = self.raw_data

        norm = data / float(self.number_of_shots)
        dz = self.bin_width

        if self.is_analog:
            # If the channel is in analog mode
            ADCrange = self.discriminator  # Discriminator value already in mV

            if self.is_photodiode and (self.adcbits == 0):
                logger.debug("Assuming adcbits equal 1. This is a bug in current licel format when storing photodiode data.")
                channel_data = norm * ADCrange / (2 ** self.adcbits)
            else:
                channel_data = norm * ADCrange / ((2 ** self.adcbits) - 1)  # Licel LabView code has a bug (does not account -1).

        else:
             channel_data = norm * self.number_of_shots

        # Calculate Z
        self.z = np.array([dz * bin_number + dz / 2.0 for bin_number in range(self.data_points)])
        self.dz = dz
        self.data = channel_data

    @property
    def is_analog(self):
        return self.analog_photon == '0'

    @property
    def is_photon(self):
        return self.analog_photon == '1'


class LicelFileV2(LicelFile):
    """ A class representing a single binary Licel file. """

    licel_file_header_format = ['filename',
                                'start_date start_time end_date end_time altitude longitude latitude zenith_angle azimuth_angle custom_field',
                                # Apart from Site that is read manually
                                'LS1 rate_1 LS2 rate_2 number_of_datasets LS3 rate_3 0000000 0000 controller_timestamp' ]
    licel_file_channel_format = 'active analog_photon laser_used number_of_datapoints laser_polarization HV bin_width wavelength d2 d3 bin_shift bin_shift_decimal_places ADCbits number_of_shots discriminator ID'

    channel_data_class = LicelChannelDataV2

    # If True, it corrects the old Raymetrics convention of zenith angle definition (zenith = -90 degrees)
    fix_zenith_angle = False

    def __init__(self, file_path, licel_timezone="UTC", get_name_by_order=False, import_now=True):
        """
        This is run when creating a new object.

        Parameters
        ----------
        file_path : str
           The path to the Licel file.
        licel_timezone : str
           The timezone of dates found in the Licel files. Should match the available
           timezones in the TZ database.
        import_now : bool
           If True, the header and data are read immediately. If not, the user has to call the
           corresponding methods directly. This is used to speed up reading files when only
           header information are required.
        """
        super(LicelFileV2, self).__init__(file_path=file_path,
                         use_id_as_name=False,  # Obsolete
                         get_name_by_order=get_name_by_order,
                         licel_timezone=licel_timezone,
                         import_now=import_now)

    def _assign_properties(self):
        """ Assign properties from the raw_info dictionary. """
        self.number_of_datasets = int(self.raw_info['number_of_datasets'])
        self.altitude = float(self.raw_info['altitude'])
        self.longitude = float(self.raw_info['longitude'])
        self.latitude = float(self.raw_info['latitude'])

        self.zenith_angle_raw = float(self.raw_info['zenith_angle'])

        if self.fix_zenith_angle:
            logger.debug('Fixing zenith angle.')
            self.zenith_angle = self._correct_zenith_angle(self.zenith_angle_raw)
        else:
            self.zenith_angle = self.zenith_angle_raw

        self.azimuth_angle = float(self.raw_info['azimuth_angle'])

        if "custom_field" in self.raw_info.keys():
            self.custom_field = self.raw_info['custom_field'].strip('"')

        self.laser1_shots = self.raw_info['LS1']
        self.laser1_reprate = self.raw_info['rate_1']
        self.laser2_shots = self.raw_info['LS2']
        self.laser2_reprate = self.raw_info['rate_2']
        self.laser3_shots = self.raw_info['LS3']
        self.laser3_reprate = self.raw_info['rate_3']

        if 'controller_timestamp' in self.raw_info.keys():
            self.controller_timestamp = self.raw_info['controller_timestamp']

    def import_file(self):
        """ Read the header info and data of the Licel file.
        """
        channels = {}
        photodiodes = {}
        powermeters = {}
        standard_deviations = {}

        with open(self.file_path, 'rb') as f:

            self.read_header(f)

            # Check the complete header is read
            f.readline()

            # Import the data
            for current_channel_info in self.channel_info:
                raw_data = np.fromfile(f, 'i4', int(current_channel_info['number_of_datapoints']))
                a = np.fromfile(f, 'b', 1)
                b = np.fromfile(f, 'b', 1)

                if (a[0] != 13) | (b[0] != 10):
                    logger.warning("No end of line found after record. File could be corrupt: %s" % self.file_path)
                    logger.warning('a: {0}, b: {1}.'.format(a, b))

                channel = self.channel_data_class(current_channel_info, raw_data, self.duration())

                # Assign the channel either as normal channel or photodiode
                if channel.is_photodiode:
                    if channel.channel_name in photodiodes.keys():
                        # Check if current naming convention produces unique files
                        raise IOError('Trying to import two photodiodes with the same name')
                    photodiodes[channel.channel_name] = channel
                elif channel.is_powermeter:
                    if channel.channel_name in powermeters.keys():
                        # Check if current naming convention produces unique files
                        raise IOError('Trying to import two powermeters with the same name')
                    powermeters[channel.channel_name] = channel
                elif channel.is_standard_deviation:
                    if channel.channel_name in channels.keys():
                        # Check if current naming convention does not produce unique files
                        raise IOError('Trying to import two standard deviations with the same name')
                    standard_deviations[channel.channel_name] = channel
                else:
                    if channel.channel_name in channels.keys():
                        # Check if current naming convention does not produce unique files
                        raise IOError('Trying to import two channels with the same name')
                    channels[channel.channel_name] = channel

        self.channels = channels
        self.photodiodes = photodiodes
        self.powermeters = powermeters
        self.standard_deviations = standard_deviations

        self._calculate_physical()

    def _calculate_physical(self):
        """ Calculate physical quantities from raw data for all channels in the file. """
        for channel in self.channels.values():
            channel.calculate_physical()

        for photodiode in self.photodiodes.values():
            photodiode.calculate_physical()

        for powermeter in self.powermeters.values():
            powermeter.calculate_physical()

        for standard_deviations in self.standard_deviations.values():
            standard_deviations.calculate_physical()

    @property
    def has_powermeter(self):
        return len(self.powermeters) != 0

    @property
    def has_standard_deviations(self):
        return len(self.standard_deviations) != 0


class LicelV2Channel(LicelChannel):

    def __init__(self):

        self.laser_polarization_int = None
        self.laser_polarization_str = None
        self.bin_shift = None
        self.bin_shift_decimal_places = None

        self.zenith_angles = []
        self.azimuth_angles = []

        super(LicelV2Channel, self).__init__()

    def append_file(self, current_file, file_channel):
        """ Append file to the current object """

        super(LicelV2Channel, self).append_file(current_file, file_channel)
        self.zenith_angles.append(current_file.zenith_angle)
        self.azimuth_angles.append(current_file.azimuth_angle)

    def _assign_properties(self, current_file, file_channel):

        self._assign_unique_property('laser_polarization_int', file_channel.laser_polarization_int)
        self._assign_unique_property('laser_polarization_str', file_channel.laser_polarization_str)
        self._assign_unique_property('bin_shift', file_channel.bin_shift)
        self._assign_unique_property('bin_shift_decimal_places', file_channel.bin_shift_decimal_places)

        super(LicelV2Channel, self)._assign_properties(current_file, file_channel)

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


class LicelLidarMeasurementV2(LicelLidarMeasurement):

    file_class = LicelFileV2
    channel_class = LicelV2Channel
    photodiode_class = PhotodiodeChannel
    powermeter_class = PhotodiodeChannel
    standard_deviation_class = LicelV2Channel

    def __init__(self, file_list=None, get_name_by_order=False, licel_timezone='UTC'):

        self.standard_deviations = {}
        self.powermeters = {}

        super(LicelLidarMeasurementV2, self).__init__(file_list=file_list,
                                                      get_name_by_order=get_name_by_order,
                                                      use_id_as_name=False,
                                                      licel_timezone=licel_timezone)

    def _create_or_append_channel(self, current_file):

        for channel_name, channel in current_file.channels.items():
            if channel_name not in self.channels:
                self.channels[channel_name] = self.channel_class()
            self.channels[channel_name].append_file(current_file, channel)

        for photodiode_name, photodiode in current_file.photodiodes.items():
            if photodiode_name not in self.photodiodes:
                self.photodiodes[photodiode_name] = self.photodiode_class()
            self.photodiodes[photodiode_name].append_file(current_file, photodiode)

        for powermeter_name, powermeter in current_file.powermeters.items():
            if powermeter_name not in self.powermeters:
                self.powermeters[powermeter_name] = self.powermeter_class()
            self.powermeters[powermeter_name].append_file(current_file, powermeter)

        for std_deviation_name, std_deviation in current_file.standard_deviations.items():
            if std_deviation_name not in self.photodiodes:
                self.standard_deviations[std_deviation_name] = self.standard_deviation_class()
            self.standard_deviations[std_deviation_name].append_file(current_file, std_deviation)

    def _import_file(self, filename):

        if filename in self.files:
            logger.warning("File has been imported already: %s" % filename)
        else:
            logger.debug('Importing file {0}'.format(filename))
            current_file = self.file_class(filename, licel_timezone=self.licel_timezone)
            self.raw_info[current_file.file_path] = current_file.raw_info
            self.durations[current_file.file_path] = current_file.duration()

            file_laser_shots = []

            self._create_or_append_channel(current_file)

            self.laser_shots.append(file_laser_shots)
            self.files.append(current_file.file_path)