import warnings
import numpy as np
from scipy import optimize

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma


def project(x, y, data, shape, weight=None):
    """Project x,y, data TOIs on a 2D grid

    Parameters
    ----------
    x, y : array_like
        input pixel indexes, 0 indexed convention
    data : array_like
        input data to project
    shape : int or tuple of int
        the shape of the output projected map
    weight : array_like
        weight to be use to sum the data (by default, ones)

    Returns
    -------
    proj_data : ndarray
        the projected data set

    Notes
    -----
    The pixel index must follow the 0 indexed convention, i.e. use `origin=0` in `*_worl2pix` methods from `~astropy.wcs.WCS`.

    >>> data = project([0], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])

    >>> data = project([-0.4], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])

    There is no test for out of shape data

    >>> data = project([-0.6, 1.6], [0, 0], [1, 1], 2)
    >>> data
    array([[nan, nan],
           [nan, nan]])
    Weighted means are also possible :

    >>> data = project([-0.4, 0.4], [0, 0], [0.5, 2], 2, weight=[2, 1])
    >>> data
    array([[ 1., nan],
           [nan, nan]))

    """
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape)

    assert len(shape) == 2, "shape must be a int or have a length of 2"

    if weight is None:
        weight = np.ones_like(data)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        weight = weight * ~data.mask

    _hits, _, _ = np.histogram2d(
        y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight
    )
    _data, _, _ = np.histogram2d(
        y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight * np.asarray(data)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _hits

    return output


def build_wcs(lon, lat, crval=None, ctype=("TLON-TAN", "TLAT-TAN"), cdelt=0.1, **kwargs):
    """Build the wcs for full projection.

    Arguments
    ---------
    lon, lat: array_like
        input longitude and latitude in degree
    crval: tuple of float
        the center of the projection in degree (default: None, computed from the data)
    ctype: tuple of str
        the type of projection (default: Az/El telescope in TAN projection)
    cdelt: float
        the size of the pixels in degree

    Returns
    -------
    wcs: `~astropy.wcs.WCS`
        the wcs for the projection
    shape: tuple of int
        the shape to porject all the data
    """

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype

    if ctype is ("TLON-TAN" "TLAT-TAN"):
        wcs.wcs.name = "Terrestrial coordinates"

    wcs.wcs.cdelt = (-cdelt, cdelt)

    if crval is None:
        # find the center of the projection
        crval = ((lon.max() + lon.min()) / 2, (lat.max() + lat.min()) / 2)

    wcs.wcs.crval = crval

    # Determine the center of the map to project all data
    x, y = wcs.all_world2pix(lon, lat, 0)
    x_min, y_min = x.min(), y.min()
    wcs.wcs.crpix = (-x_min, -y_min)

    shape = (np.round(y.max() - y.min()).astype(np.int) + 1, np.round(x.max() - x.min()).astype(np.int) + 1)

    return wcs, shape
