""" Code to read Raymetrics version of Licel binary files."""
import logging

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from .licel import LicelFile, LicelLidarMeasurement, LicelChannel, PhotodiodeChannel

logger = logging.getLogger(__name__)


class ScanningFile(LicelFile):
    """ Raymetrics is using a custom version of licel file format to store scanning lidar measurements.
     
    The file includes one extra line describing the scan strategy of the dataset. The extra parameters are:
    
    `azimuth_start`
        Start azimuth angle for the scan, relative to instrument zero position (degrees).
    
    `azimuth_stop`
        Stop azimuth angle for the scan, relative to instrument zero position (degrees).
        
    `azimuth_step` 
        Step of the azimuth scan (degrees).

    `zenith_start` 
        Start zenith angle for the scan, relative to *nadir* (degrees). Take care that this is actually
        nadir angle. Vertical measurements correspond to -90.

    `zenith_stop` 
        Stop zenith angle for the scan, relative to *nadir* (degrees). Take care that this is actually
        nadir angle. Vertical measurements correspond to -90.

    `zenith_step` 
        Step of the zenith scan (degrees).

    `azimuth_offset`
        Offset of instrument zero from North (degrees). Using this value you can convert `azimuth_start` and
        `azimuth_stop` to absolute values.

    Moreover, four new parameters are added in the second line of the file:

    `zenith_angle`
        Zenith angle of the current file. Take care that this is actually
        nadir angle. Vertical measurements correspond to -90.

    `azimuth_angle`
        Azimuth angle of the current file. Value relative to instrument zero position.

    `temperature`
        Ambient temperature (degrees C)

    `pressure`
        Ambient pressure (hPa)
    """

    # Specifications of the header lines.
    licel_file_header_format = ['filename',
                                'start_date start_time end_date end_time altitude longitude latitude zenith_angle azimuth_angle temperature pressure custom_info',
                                # Appart from Site that is read manually
                                'azimuth_start azimuth_stop azimuth_step zenith_start zenith_stop zenith_step azimuth_offset',
                                'LS1 rate_1 LS2 rate_2 number_of_datasets', ]

    # Specifications of the channel lines in the header
    licel_file_channel_format = 'active analog_photon laser_used number_of_datapoints 1 HV bin_width wavelength d1 d2 d3 d4 ADCbits number_of_shots discriminator ID'

    fix_zenith_angle = True
    skip_scan_overview_line = False  # Skip the 3d line containing azimuth_start, stop etc. Used to overcome a bug in some output files.

    def _read_rest_of_header(self, f):
        """ Read the third and fourth row of  of the header lines.

        The first two rows are read in the licel class.

        Parameters
        ----------
        f : file
           An open file-like object.

        Returns
        -------
        raw_info : dict
           A dictionary containing all parameters of the third and fourth line of the header.
        """
        raw_info = {}

        third_line = f.readline().decode()
        self.header_lines.append(third_line.strip())

        raw_info.update(self.match_lines(third_line, self.licel_file_header_format[2]))

        fourth_line = f.readline().decode()
        self.header_lines.append(fourth_line.strip())

        raw_info.update(self.match_lines(fourth_line, self.licel_file_header_format[3]))
        return raw_info

    def _assign_properties(self):
        """ Assign scanning-specific parameters found in the header as object properties."""
        super(ScanningFile, self)._assign_properties()
        self.azimuth_angle_raw = float(self.raw_info['azimuth_angle'])
        self.temperature = float(self.raw_info['temperature'])
        self.pressure = float(self.raw_info['pressure'])

        if not self.skip_scan_overview_line:
            self.azimuth_start_raw = float(self.raw_info['azimuth_start'])
            self.azimuth_stop_raw = float(self.raw_info['azimuth_stop'])
            self.azimuth_step = float(self.raw_info['azimuth_step'])
            self.zenith_start_raw = float(self.raw_info['zenith_start'])
            self.zenith_stop_raw = float(self.raw_info['zenith_stop'])
            self.zenith_step = float(self.raw_info['zenith_step'])
            self.azimuth_offset = float(self.raw_info['azimuth_offset'])
        else:
            self.azimuth_start_raw = np.nan
            self.azimuth_stop_raw = np.nan
            self.azimuth_step = np.nan
            self.zenith_start_raw = np.nan
            self.zenith_stop_raw = np.nan
            self.zenith_step = np.nan
            self.azimuth_offset = 0

        self.azimuth_angle = (self.azimuth_angle_raw + self.azimuth_offset) % 360
        self.azimuth_start = (self.azimuth_start_raw + self.azimuth_offset) % 360
        self.azimuth_stop = (self.azimuth_stop_raw + self.azimuth_offset) % 360

        if self.fix_zenith_angle:
            logger.debug('Fixing zenith start and zenith stop angles.')
            self.zenith_start = self._correct_zenith_angle(self.zenith_start_raw)
            self.zenith_stop = self._correct_zenith_angle(self.zenith_stop_raw)
        else:
            self.zenith_start = self.zenith_start_raw
            self.zenith_stop = self.zenith_stop_raw

        try:
            self.custom_info = self.raw_info['custom_info'].strip('"')
        except KeyError:
            self.custom_info = None

    def get_coordinates(self, channel_name):
        """
        Calculate the lat, lon, z coordinates for each measurement point.

        Parameters
        ----------
        channel_name : str
           The name of the channel. Only the channel object knows about the
           range resolution.

        Returns
        -------
        lat : array
           Latitude array
        lon : array
           Longitude array
        z : array
           Altitude array in meters
        """
        R_earth = 6378137  # Earth radius in meters

        # Shortcuts to make equations cleaner
        lat_center = self.latitude
        lon_center = self.longitude
        r = self.channels[channel_name].z
        azimuth = self.azimuth_angle
        zenith = self.zenith_angle

        # Convert all angles to radiants
        zenith_rad = np.deg2rad(zenith)[:, np.newaxis]
        azimuth_rad = np.deg2rad(azimuth)[:, np.newaxis]

        lat_center_rad = np.deg2rad(lat_center)
        lon_center_rad = np.deg2rad(lon_center)

        # Generate the mesh
        R, Zeniths = np.meshgrid(r, zenith_rad)
        R_ground = R * np.sin(Zeniths)
        Z = R * np.cos(Zeniths)

        # Equations from https://www.movable-type.co.uk/scripts/latlong.html
        delta = R_ground / R_earth
        lat_out_rad = np.arcsin(np.sin(lat_center_rad) * np.cos(delta)
                                + np.cos(lat_center_rad) * np.sin(delta) * np.cos(azimuth_rad))
        lon_out_rad = lon_center_rad + np.arctan2(np.sin(azimuth_rad) * np.sin(delta) * np.cos(lat_center_rad),
                                                  np.cos(delta) - np.sin(lat_center_rad) * np.sin(lat_out_rad))

        # Convert back to degrees
        lat_out = np.rad2deg(lat_out_rad)
        lon_out = np.rad2deg(lon_out_rad)

        return lat_out, lon_out, Z


class ScanningChannel(LicelChannel):
    """ A class representing measurements of a specific lidar channel, during a scanning measurement. """

    def __init__(self):
        super(ScanningChannel, self).__init__()

        self.azimuth_start = None
        self.azimuth_stop = None
        self.azimuth_step = None
        self.zenith_start = None
        self.zenith_stop = None
        self.zenith_step = None
        self.azimuth_offset = None
        self.zenith_angles = []
        self.azimuth_angles = []
        self.temperature = []
        self.pressure = []

    def append_file(self, current_file, file_channel):
        """ Keep track of scanning-specific variable properties of each file. """
        super(ScanningChannel, self).append_file(current_file, file_channel)
        self.zenith_angles.append(current_file.zenith_angle)
        self.azimuth_angles.append(current_file.azimuth_angle)
        self.temperature.append(current_file.temperature)
        self.pressure.append(current_file.pressure)

    def _assign_properties(self, current_file, file_channel):
        """ Assign scanning-specific properties as object properties. Check that these are unique,
        i.e. that all files belong to the same measurements set.

        Parameters
        ----------
        current_file : ScanningFile object
           A ScanningFile object being imported
        file_channel : LicelChannelData object
           A specific LicelChannelData object holding data found in the file.
        """
        super(ScanningChannel, self)._assign_properties(current_file, file_channel)
        self._assign_unique_property('azimuth_start', current_file.azimuth_start)
        self._assign_unique_property('azimuth_stop', current_file.azimuth_stop)
        self._assign_unique_property('azimuth_start_raw', current_file.azimuth_start_raw)
        self._assign_unique_property('azimuth_stop_raw', current_file.azimuth_stop_raw)
        self._assign_unique_property('azimuth_step', current_file.azimuth_step)
        self._assign_unique_property('zenith_start', current_file.zenith_start)
        self._assign_unique_property('zenith_stop', current_file.zenith_stop)
        self._assign_unique_property('zenith_start_raw', current_file.zenith_start_raw)
        self._assign_unique_property('zenith_stop_raw', current_file.zenith_stop_raw)
        self._assign_unique_property('zenith_step', current_file.zenith_step)

    def plot_ppi(self, figsize=(8, 4), signal_type='rc', z_min=0., z_max=12000., show_plot=True,
                 cmap=plt.cm.jet, title=None, vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1.):
        """
        Plot a vertical project of channel data.

        Parameters
        ----------
        figsize : tuple
           (width, height) of the output figure (inches)
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        self.draw_ppi(ax1, cmap=cmap, signal_type=signal_type, z_min=z_min, z_max=z_max, vmin=vmin, vmax=vmax,
                      mask_noise=mask_noise, noise_threshold=noise_threshold)

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("PPI scan")

        if show_plot:
            plt.show()

    def plot_rhi(self, figsize=(8, 4), signal_type='rc', z_min=0., z_max=12000., show_plot=True,
                 cmap=plt.cm.jet, title=None, vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1.):
        """
        Plot a vertical project of channel data.

        Parameters
        ----------
        figsize : tuple
           (width, height) of the output figure (inches)
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)

        projection_angle = self.draw_rhi(ax1, cmap=cmap, signal_type=signal_type, z_min=z_min, z_max=z_max, vmin=vmin,
                                         vmax=vmax,
                                         mask_noise=mask_noise, noise_threshold=noise_threshold)

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("RHI scan ({0}$^\circ$)".format(projection_angle))

        if show_plot:
            plt.show()

    def plot_scan(self, figsize=(8, 4), signal_type='rc', z_min=0., z_max=12000., show_plot=True,
                  cmap=plt.cm.jet, vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1., cb_format='%.0e',
                  box=False, grid=(1, 4), ax1_position=(0, 0), ax1_span=2, ax2_position=(0, 2), ax2_span=2):
        """
        Plot data as RHI and PPI scans.

        Parameters
        ----------
        figsize : tuple
           (width, height) of the output figure (inches)
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        """
        fig = plt.figure(figsize=figsize)

        ax1 = plt.subplot2grid(grid, ax1_position, colspan=ax1_span)
        ax2 = plt.subplot2grid(grid, ax2_position, colspan=ax2_span)

        self.draw_ppi(ax1, cmap=cmap, signal_type=signal_type, z_min=z_min, z_max=z_max, vmin=vmin, vmax=vmax,
                      mask_noise=mask_noise, noise_threshold=noise_threshold, add_colorbar=False, cb_format=cb_format,
                      box=box)

        projection_angle = self.draw_rhi(ax2, cmap=cmap, signal_type=signal_type, z_min=z_min, z_max=z_max, vmin=vmin,
                                         vmax=vmax,
                                         mask_noise=mask_noise, noise_threshold=noise_threshold, cb_format=cb_format,
                                         box=box)

        fig.suptitle("Channel {0}: {1} - {2}".format(self.name,
                                                     self.start_time.strftime('%Y%m%dT%H%M'),
                                                     self.stop_time.strftime('%Y%m%dT%H%M')))

        ax1.set_title('PPI')
        ax2.set_title("RHI ({0:.1f}$^\circ$)".format(projection_angle))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if show_plot:
            plt.show()

    def draw_ppi(self, ax1, cmap=plt.cm.jet, signal_type='rc',
                 z_min=0, z_max=12000., add_colorbar=True, cmap_label='a.u.', cb_format=None,
                 vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1., first_signal_bin=0, box=False):
        """
        Draw channel data as a PPI plot.

        Parameters
        ----------
        ax1 : axis object
           The axis object to draw.
        x : array
           X axis coordinates
        y : array
           Y axis coordinates
        cmap : cmap
           An instance of a matplotlib colormap to use.
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        first_signal_bin : int
           First signal bin. Can be used to fix analog bin shift of Licel channels.
        """
        x, y = self._polar_to_ground(self.z / 1000., self.azimuth_angles, self.zenith_angles)

        self.draw_projection(ax1, x, y, cmap=cmap, signal_type=signal_type,
                             z_min=z_min, z_max=z_max, add_colorbar=add_colorbar, cmap_label=cmap_label,
                             cb_format=cb_format, vmin=vmin, vmax=vmax, mask_noise=mask_noise,
                             noise_threshold=noise_threshold, first_signal_bin=first_signal_bin)

        if box:
            ax1.set_xlim(-z_max / 1000., z_max / 1000.)
            ax1.set_ylim(-z_max / 1000., z_max / 1000.)

        ax1.set_ylabel('South-North (km)')
        ax1.set_xlabel('West-East (km)')

    def draw_rhi(self, ax1, cmap=plt.cm.jet, signal_type='rc',
                 z_min=0, z_max=12000., add_colorbar=True, cmap_label='a.u.', cb_format=None,
                 vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1., first_signal_bin=0, box=False):
        """
        Draw channel data as a PPI plot.

        Parameters
        ----------
        ax1 : axis object
           The axis object to draw.
        x : array
           X axis coordinates
        y : array
           Y axis coordinates
        cmap : cmap
           An instance of a matplotlib colormap to use.
        signal_type : str
           If 'rc', the range corrected signal is plotted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        first_signal_bin : int
           First signal bin. Can be used to fix analog bin shift of Licel channels.
        """
        projection_angle = np.mean(self.azimuth_angles)
        x, y = self._polar_to_cross_section(self.z / 1000., self.azimuth_angles, self.zenith_angles, projection_angle)

        self.draw_projection(ax1, x, y, cmap=cmap, signal_type=signal_type,
                             z_min=z_min, z_max=z_max, add_colorbar=add_colorbar, cmap_label=cmap_label,
                             cb_format=cb_format, vmin=vmin, vmax=vmax, mask_noise=mask_noise,
                             noise_threshold=noise_threshold, first_signal_bin=first_signal_bin)

        padding = 0.5  # km

        if box:
            ax1.set_xlim(-z_max / 1000. - padding, z_max / 1000. + padding)
            ax1.set_ylim(-padding, z_max / 1000. + padding)

        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Height a.l. (km)')

        return projection_angle

    def draw_projection(self, ax1, x, y, cmap=plt.cm.jet, signal_type='rc',
                        z_min=0, z_max=12000., add_colorbar=True, cmap_label='a.u.', cb_format=None,
                        vmin=0, vmax=1.3 * 10 ** 7, mask_noise=True, noise_threshold=1.,
                        first_signal_bin=0):
        """
        Draw channel data as a PPI plot.

        Parameters
        ----------
        ax1 : axis object
           The axis object to draw.
        x : array
           X axis coordinates
        y : array
           Y axis coordiantes
        cmap : cmap
           An instance of a matplotlib colormap to use.
        signal_type : str
           If 'rc', the range corrected signal is ploted. Else, the raw signals are used.
        z_min : float
           Minimum z range
        z_max : float
           Maximum z range
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
        mask_noise : bool
           If True, remove noisy bins.
        noise_threshold : int
           Threshold to use in the noise masking routine.
        first_signal_bin : int
           First signal bin. Can be used to fix analog bin shift of Licel channels.
        """
        if signal_type == 'rc':
            if len(self.rc) == 0:
                self.calculate_rc()
            data = self.rc
        else:
            data = self.matrix

        if mask_noise:
            mask = self.noise_mask(threshold=noise_threshold)
            data = np.ma.masked_where(mask, data)

        z_min_idx = self._index_at_height(z_min)
        z_max_idx = self._index_at_height(z_max)

        data_min_idx = z_min_idx + first_signal_bin
        data_max_idx = z_max_idx + first_signal_bin

        im1 = ax1.pcolormesh(x[:, z_min_idx:z_max_idx], y[:, z_min_idx:z_max_idx], data[:, data_min_idx:data_max_idx],
                             cmap=cmap, vmin=vmin, vmax=vmax)

        ax1.set(adjustable='box', aspect='equal')

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

        # Centered axis in center: https://stackoverflow.com/a/31558968
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        # ax1.spines['left'].set_position('center')
        # ax1.spines['bottom'].set_position('center')
        #
        # # Eliminate upper and right axes
        # ax1.spines['right'].set_color('none')
        # ax1.spines['top'].set_color('none')
        #
        # # Show ticks in the left and lower axes only
        # ax1.xaxis.set_ticks_position('bottom')
        # ax1.yaxis.set_ticks_position('left')

    @staticmethod
    def _polar_to_ground(z, azimuth, zenith):
        """
        Convert polar coordinates to cartesian project for a PPI scan

        Parameters
        ----------
        z : array
           Distance array in meters
        azimuth : list
           List of profile azimuth angles in degrees
        zenith : list
           List of profile zenith angles in degrees

        Returns
        -------
        x : array
           X axis in meters
        y : array
           Y axis in meters
        """
        # Generate the mesh
        zenith_rad = np.deg2rad(zenith)
        azimuth_rad = np.deg2rad(azimuth)

        Z, Zeniths = np.meshgrid(z, zenith_rad)
        Z_ground = Z * np.sin(Zeniths)

        x = Z_ground * np.sin(azimuth_rad)[:, np.newaxis]
        y = Z_ground * np.cos(azimuth_rad)[:, np.newaxis]

        return x, y

    @staticmethod
    def _polar_to_cross_section(z, azimuth, zenith, cross_section_azimuth):
        """
        Convert polar coordinates to cartesian project for a PPI scan

        Parameters
        ----------
        z : array
           Distance array in meters
        azimuth : list
           List of profile azimuth angles in degrees
        zenith : list
           List of profile zenith angles in degrees
        cross_section_azimuth : float
           Azimuth angle of plane in degrees

        Returns
        -------
        x : array
           X axis in meters
        y : array
           Y axis in meters
        """

        zenith_rad = np.deg2rad(zenith)

        # The angle between measurements and the cross section plance
        azimuth_difference_rad = np.deg2rad(azimuth) - np.deg2rad(cross_section_azimuth)

        # Generate the mesh
        Z, Azimuth_differences = np.meshgrid(z, azimuth_difference_rad)

        x = Z * np.sin(zenith_rad)[:, np.newaxis] * np.cos(Azimuth_differences)
        y = Z * np.cos(zenith_rad)[:, np.newaxis]

        return x, y

    def get_coordinates(self,):
        """
        Calculate the lat, lon, z coordinates for each measurement point.

        Returns
        -------
        lat : array
           Latitude array
        lon : array
           Longitude array
        z : array
           Altitude array in meters
        """
        R_earth = 6378137  # Earth radius in meters

        # Shortcuts to make equations cleaner
        lat_center = self.latitude
        lon_center = self.longitude
        r = self.z
        azimuth = self.azimuth_angles
        zenith = self.zenith_angles

        # Convert all angles to radiants
        zenith_rad = np.deg2rad(zenith)[:, np.newaxis]
        azimuth_rad = np.deg2rad(azimuth)[:, np.newaxis]

        lat_center_rad = np.deg2rad(lat_center)
        lon_center_rad = np.deg2rad(lon_center)

        # Generate the mesh
        R, Zeniths = np.meshgrid(r, zenith_rad)
        R_ground = R * np.sin(Zeniths)
        Z = R * np.cos(Zeniths)

        # Equations from https://www.movable-type.co.uk/scripts/latlong.html
        delta = R_ground / R_earth
        lat_out_rad = np.arcsin(np.sin(lat_center_rad) * np.cos(delta)
                                + np.cos(lat_center_rad) * np.sin(delta) * np.cos(azimuth_rad))
        lon_out_rad = lon_center_rad + np.arctan2(np.sin(azimuth_rad) * np.sin(delta) * np.cos(lat_center_rad),
                                                  np.cos(delta) - np.sin(lat_center_rad) * np.sin(lat_out_rad))

        # Convert back to degrees
        lat_out = np.rad2deg(lat_out_rad)
        lon_out = np.rad2deg(lon_out_rad)

        return lat_out, lon_out, Z


class ScanningFileMissingLine(ScanningFile):
    skip_scan_overview_line = True


class ScanningLidarMeasurement(LicelLidarMeasurement):
    """ A class representing a scanning measurement set.

    It useses `ScanningFile` and `ScanningChannel` classes for handling the data.
    """
    file_class = ScanningFile
    channel_class = ScanningChannel
    photodiode_class = PhotodiodeChannel


class FixedPointingFile(LicelFile):
    """ Raymetrics is using a custom version of licel file format to store
    vertical lidar measurements.

    `azimuth_angle`
        The angle from North

    `temperature`
        Ambient temperature (degrees C)

    `pressure`
        Ambient pressure (hPa)
    """
    # Specifications of the header lines.
    licel_file_header_format = ['filename',
                                'start_date start_time end_date end_time altitude longitude latitude zenith_angle azimuth_angle temperature pressure custom_info',
                                # Appart from Site that is read manually
                                'LS1 rate_1 LS2 rate_2 number_of_datasets', ]

    fix_zenith_angle = True

    def __init__(self, file_path, use_id_as_name=False, get_name_by_order=False, licel_timezone="UTC", import_now=True, azimuth_offset=0):
        self.azimuth_offset = azimuth_offset

        super().__init__(file_path, use_id_as_name=use_id_as_name, get_name_by_order=get_name_by_order,
                         licel_timezone=licel_timezone, import_now=import_now)

    def _assign_properties(self):
        """ Assign scanning-specific parameters found in the header as object properties."""
        super(FixedPointingFile, self)._assign_properties()

        self.temperature = float(self.raw_info['temperature'])
        self.pressure = float(self.raw_info['pressure'])
        self.azimuth_angle_raw = float(self.raw_info['azimuth_angle'])

        azimuth_offset = getattr(self, 'azimuth_offset', None)

        # Add a option to manually define an azimuth offset, but still work if you don't
        if azimuth_offset is not None:
            self.azimuth_angle = (self.azimuth_angle_raw + azimuth_offset) % 360
        else:
            self.azimuth_angle = self.azimuth_angle_raw

        try:
            self.custom_info = self.raw_info['custom_info'].strip('"')
        except KeyError:
            self.custom_info = None


class FixedPointingChannel(LicelChannel):
    """ A class representing measurements of a specific lidar channel, during a fixed pointing measurement. """

    def __init__(self):
        super(FixedPointingChannel, self).__init__()
        self.zenith_angles = []
        self.azimuth_angles = []
        self.temperature = []
        self.pressure = []

    def append_file(self, current_file, file_channel):
        """ Keep track of scanning-specific variable properties of each file. """
        super(FixedPointingChannel, self).append_file(current_file, file_channel)
        self.zenith_angles.append(current_file.zenith_angle)
        self.azimuth_angles.append(current_file.azimuth_angle)
        self.temperature.append(current_file.temperature)
        self.pressure.append(current_file.pressure)


class FixedPointingLidarMeasurement(LicelLidarMeasurement):
    file_class = FixedPointingFile
    channel_class = FixedPointingChannel
