import numpy as np
import xarray as xr

from lidarpy.scc.plot.product import SCC_elda


def angstrom_exponent(product1: SCC_elda, product2: SCC_elda) -> xr.DataArray | None:
    """Retrive Angstrom exponent with scc products. 

    Args:
        product1 (SCC_elda): Elda product object
        product2 (SCC_elda): Elda product object.

    Returns:
        xr.DataArray | None
    """
    elda1, elda2 = xr.open_dataset(product1.path), xr.open_dataset(product2.path)
    backscatter1 = elda1.backscatter.sel(wavelength=product1.wavelength)
    backscatter2 = elda2.backscatter.sel(wavelength=product2.wavelength)
    if len(elda1.altitude) > len(elda2.altitude):
        interpolated_backscatter2 = elda2.backscatter.interp(altitude=elda1.backscatter.altitude, method='nearest')
        ratio = interpolated_backscatter2/backscatter1
        positive = np.logical_and(interpolated_backscatter2>0,backscatter1>0)
    else:
        interpolated_backscatter1 = elda1.backscatter.interp(altitude=elda2.backscatter.altitude, method='nearest')
        ratio = backscatter2/interpolated_backscatter1
        positive = np.logical_and(backscatter2>0,interpolated_backscatter1>0)
    
    ae = (-np.log(ratio.where(positive))/np.log(product2.wavelength/product1.wavelength)).squeeze()
    if len(ae.values)==0:
        ae = None
    return ae