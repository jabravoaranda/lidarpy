from __future__ import annotations

from lidarpy.general_utils.io import read_yaml as general_read_yaml
from lidarpy.general_utils.smoothing import bin_rescale as general_bin_rescale
from lidarpy.utils.io import read_yaml as compat_read_yaml
from lidarpy.utils.smoothing import bin_rescale as compat_bin_rescale


def test_utils_io_reexports_general_utils_io():
    assert compat_read_yaml is general_read_yaml


def test_utils_smoothing_reexports_general_utils_smoothing():
    assert compat_bin_rescale is general_bin_rescale
