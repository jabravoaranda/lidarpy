import logging
from pdb import set_trace

import numpy as np

from .licel import LicelLidarMeasurement

logger = logging.getLogger(__name__)


class LicelCalibrationMeasurement(LicelLidarMeasurement):

    def __init__(self, plus45_files=None, minus45_files=None, use_id_as_name=False, get_name_by_order=False, licel_timezone='UTC'):
        """Class to handle depolarization calibration measurements according to the SCC.


        Parameters
        ----------
        plus45_files : list of str
           List of paths for the plus 45 files.
        minus45_files : list of str
           List of paths for the minus 45 files.
        use_id_as_name : bool
           Defines if channels names are descriptive or transient digitizer IDs.
        get_name_by_order : bool
           If True, channels are named by the order they appear.
        licel_timezone : str
           String describing the timezone according to the tz database.
        """
        # Setup the empty class
        super(LicelCalibrationMeasurement, self).__init__(use_id_as_name=use_id_as_name, licel_timezone=licel_timezone,
                                                          get_name_by_order=get_name_by_order)

        self.plus45_files = plus45_files
        self.minus45_files = minus45_files

        if plus45_files and minus45_files:
            self.files = plus45_files + minus45_files

            self.check_equal_length()
            self.read_channel_data()
            self.update()

    def subset_by_scc_channels(self):
        m = super(LicelCalibrationMeasurement, self).subset_by_scc_channels()
        m.plus45_measurement = self.plus45_measurement.subset_by_scc_channels()
        m.minus45_measurement = self.minus45_measurement.subset_by_scc_channels()
        return m

    def update(self):
        """
        Correct timescales after each update.
        """
        super(LicelCalibrationMeasurement, self).update()
        self.correct_timescales()

    def check_equal_length(self):
        """
        Check if input time series have equal lengths.
        """
        len_plus = len(self.plus45_files)
        len_minus = len(self.minus45_files)
        if len_plus != len_minus:
            raise self.UnequalMeasurementLengthError(
                            "Input timeseries have different length: %s vs %s." % (len_plus, len_minus))

    def read_channel_data(self):
        # Read plus and minus 45 measurements
        self.plus45_measurement = LicelLidarMeasurement(self.plus45_files, use_id_as_name=self.use_id_as_name,
                                                        get_name_by_order=self.get_name_by_order,
                                                        licel_timezone=self.licel_timezone)
        self.plus45_measurement.extra_netcdf_parameters = self.extra_netcdf_parameters
        self.plus45_measurement.rename_channels(suffix='_p45')

        self.minus45_measurement = LicelLidarMeasurement(self.minus45_files, use_id_as_name=self.use_id_as_name,
                                                         get_name_by_order=self.get_name_by_order,
                                                         licel_timezone=self.licel_timezone)
        self.minus45_measurement.extra_netcdf_parameters = self.extra_netcdf_parameters
        self.minus45_measurement.rename_channels(suffix='_m45')

        # Combine them in this object
        self.channels = {}
        self.channels.update(self.plus45_measurement.channels)
        self.channels.update(self.minus45_measurement.channels)

        self.raw_info = self.plus45_measurement.raw_info.copy()
        self.raw_info.update(self.minus45_measurement.raw_info)

    def correct_timescales(self):
        self.check_timescales_are_two()
        self.combine_scales()

    def check_timescales_are_two(self):
        no_timescales = len(self.variables['Raw_Data_Start_Time'])
        if no_timescales != 2:
            raise self.WrongNumberOfTimescalesError("Wrong number of timescales: %s instead of 2." % no_timescales)

    def combine_scales(self):
        start_times, end_times = self.get_ordered_timescales()
        new_start_time = start_times[0]
        new_stop_time = end_times[1]
        self.variables['Raw_Data_Start_Time'] = [new_start_time, ]
        self.variables['Raw_Data_Stop_Time'] = [new_stop_time, ]
        self.reset_timescale_id()

        self.dimensions['nb_of_time_scales'] = 1

    # def _get_custom_global_attributes(self):
    #     """
    #     NetCDF global attributes that should be included in the final NetCDF file.
    #
    #     Using the values of just p45 measurements.
    #     """
    #     return self.plus45_measurement._get_custom_global_attributes()

    def reset_timescale_id(self):
        """
        Set all timescales to 0
        :return:
        """
        timescale_dict = self.variables['id_timescale']
        self.variables['id_timescale'] = dict.fromkeys(timescale_dict, 0)

    def get_ordered_timescales(self):
        scale_start_1, scale_start_2 = self.variables['Raw_Data_Start_Time']
        scale_end_1, scale_end_2 = self.variables['Raw_Data_Stop_Time']

        if scale_start_1[0] > scale_start_2[0]:
            scale_start_1, scale_start_2 = scale_start_2, scale_start_1

        if scale_end_1[0] > scale_end_2[0]:
            scale_end_1, scale_end_2 = scale_end_2, scale_end_1

        return (scale_start_1, scale_start_2), (scale_end_1, scale_end_2)

    def add_fake_measurements(self, no_profiles, variation=0.1):
        """
        Add a number of fake measurements. This is done to allow testing with single analog profiles.

        Adds a predefined variation in each new profile.
        """
        duration = self.info['duration']
        for channel_name, channel in self.channels.items():
            base_time = list(channel.data.keys())[0]
            base_data = channel.data[base_time]

            for n in range(no_profiles):
                random_variation = base_data * (np.random.rand(len(base_data)) * 2 - 1) * variation

                new_time = base_time + n * duration
                new_data = channel.data[base_time].copy() + random_variation
                if 'ph' in channel_name:
                    new_data = new_data.astype('int')
                channel.data[new_time] = new_data

        self.update()

    def subset_photoncounting(self):
        """
        Subset photoncounting channels.
        """
        ph_channels = [channel for channel in self.channels.keys() if 'ph' in channel]
        new_measurement = self.subset_by_channels(ph_channels)
        return new_measurement

    def _get_scc_channel_variables(self):
        """
        Get a list of variables to put in the SCC.

        It can be overridden e.g. in the depolarization product class.

        Returns
        -------

        channel_variables: dict
           A dictionary with channel variable specifications.
        """
        channel_variables = \
            {'Background_Low': (('channels',), 'd'),
             'Background_High': (('channels',), 'd'),
             'LR_Input': (('channels',), 'i'),
             'DAQ_Range': (('channels',), 'd'),
             'Pol_Calib_Range_Min': (('channels',), 'd'),
             'Pol_Calib_Range_Max': (('channels',), 'd'),
             }
        return channel_variables

    class UnequalMeasurementLengthError(RuntimeError):
        """ Raised when the plus and minus files have different length.
        """
        pass

    class WrongNumberOfTimescalesError(RuntimeError):
        """ Raised when timescales are not two.
        """
        pass


class DarkLicelCalibrationMeasurement(LicelCalibrationMeasurement):

    def __init__(self, dark_files=None, use_id_as_name=False, licel_timezone='UTC'):
        """Class to handle dark files for depolarization calibration measurements according to the SCC.

        It assumes that a single sent of dark measurements will be use for both plus and minus 45 channels.

        Parameters
        ----------
        dark_files : list of str
           List of paths for the dark measurement files.
        use_id_as_name : bool
           Defines if channels names are descriptive or transient digitizer IDs.
        licel_timezone : str
           String describing the timezone according to the tz database.
        """
        # Setup the empty class
        super(DarkLicelCalibrationMeasurement, self).__init__(dark_files, dark_files,
                                                              use_id_as_name=use_id_as_name,
                                                              licel_timezone=licel_timezone)

    def correct_timescales(self):
        """ For dark measuremetns, no need to correct timescales. """
        pass
