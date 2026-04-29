# [cómo usar restructuretext --> https://www.sphinx-doc.org/es/master/usage/restructuredtext/index.html]

from .lidar_preprocessing import preprocess
from . import gluing_de_la_rosa, lidar_gluing_bravo_aranda

__all__ = ["preprocess", "gluing_de_la_rosa", "lidar_gluing_bravo_aranda"]

__doc__ = """
This module contains a file `lidarpy.preprocessing.preprocess` which apply the following corrections to the lidar raw data:

* Dark measurement [AN only]
* Dead time correction [PC only]
* Trigger delay [PC only] [tbi]
* Zero bin [AN only]
* Background substraction
* Merge of polarized channels

The following files contains the depolarization calibation factors and the GHK parameters:
* `lidarpy.preprocessing.depolarization_calibration_mhc`
* `lidarpy.preprocessing.depolarization_calibration_alh`
* `lidarpy.preprocessing.depolarization_calibration_vlt`

.. warning::
   Be sure these files are updated.

### Gluing:

There are some gluing implementations already. The main two are:
- `lidarpy.preprocessing.gluing_de_la_rosa`: Calculates a windowed correlation coefficient in everypoint within a range.
   Then, selects the highest and with a linear regression scales analog channel to photocounting untis.
   It works, but needs consistency.  For a good adjustment, windows has to be large and it works even better if the adjustment window is even larger. For instance:
   ```
   win_an = rcs_an_whole[idx, gl_bin - str(hw * 1.5) : gl_bin + hw * 5]
   win_pc = rcs_pc_whole[idx, gl_bin - str(hw * 1.5) : gl_bin + hw * 5]
   ```
   For instance in the default configuration The day /2021/07/05 from MULHACEN has a gluing inconsistency at time profile 217.
   Neighbours are a bit higher, resulting in a visible vertical line in the quicklook.
   It is thought that it happens because with larger adjustment windows the regression captures better how the scale (a and b params of the linear regression)
   despite the noise at higher altitudes.

- `lidarpy.preprocessing.lidar_gluing_bravo_aranda`: Calculates a windowed linear regression in everypoing for both channels and within a range.
   Chooses the point where this is minimum  and make a gluing with another window larger than the linear regression.
   For larger adjustment windows it works better.

"""
