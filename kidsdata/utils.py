import warnings
import numpy as np
from scipy import optimize

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma


def project(x, y, data, shape, weight=None):
    """Project x,y, data TOIs on a 2D grid.

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
    data, weight, hits : ndarray
        the projected data set and corresponding weights and hits

    Notes
    -----
    The pixel index must follow the 0 indexed convention, i.e. use `origin=0` in `*_worl2pix` methods from `~astropy.wcs.WCS`.

    >>> data, weight, hits = project([0], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])
    >>> weight
    array([[1., 0.],
           [0., 0.]])
    >>> hits
    array([[1, 0],
           [0, 0]]))

    >>> data, _, _ = project([-0.4], [0], [1], 2)
    >>> data
    array([[ 1., nan],
           [nan, nan]])

    There is no test for out of shape data

    >>> data, _, _ = project([-0.6, 1.6], [0, 0], [1, 1], 2)
    >>> data
    array([[nan, nan],
           [nan, nan]])

    Weighted means are also possible :

    >>> data, weight, hits = project([-0.4, 0.4], [0, 0], [0.5, 2], 2, weight=[2, 1])
    >>> data
    array([[ 1., nan],
           [nan, nan]))
    >>> weight
    array([[3., 0.],
           [0., 0.]])
    >>> hits
    array([[2, 0],
           [0, 0]])
    """
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape)

    assert len(shape) == 2, "shape must be a int or have a length of 2"

    if weight is None:
        weight = np.ones_like(data)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        weight = weight * ~data.mask

    _hits, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)))

    _weight, _, _ = np.histogram2d(
        y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight
    )
    _data, _, _ = np.histogram2d(
        y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight * np.asarray(data)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weight

    return output, _weight, _hits.astype(np.int)


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
    x, y: list of floats
        the projected positions
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype

    if ctype == ("TLON-TAN" "TLAT-TAN"):
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

    return wcs, x, y


def elliptical_gaussian(X, amplitude, xo, yo, fwhm_x, fwhm_y, theta, offset):
    """Ellipcital gaussian function."""
    x, y = X
    xo = float(xo)
    yo = float(yo)
    sigma_x_sqr = (fwhm_x * gaussian_fwhm_to_sigma) ** 2
    sigma_y_sqr = (fwhm_y * gaussian_fwhm_to_sigma) ** 2
    a = np.cos(theta) ** 2 / (2 * sigma_x_sqr) + np.sin(theta) ** 2 / (2 * sigma_y_sqr)
    b = np.sin(2 * theta) / (4 * sigma_x_sqr) - np.sin(2 * theta) / (4 * sigma_y_sqr)
    c = np.sin(theta) ** 2 / (2 * sigma_x_sqr) + np.cos(theta) ** 2 / (2 * sigma_y_sqr)
    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def elliptical_gaussian_start_params(data):
    """Get starting parameters for ellipcital gaussian."""
    return [np.nanmax(data), *np.unravel_index(np.nanargmax(data, axis=None), data.shape)[::-1], 5, 5, 0, 0]


def elliptical_disk(X, amplitude, xo, yo, fwhm_x, fwhm_y, theta, offset):
    """Ellipcital disk function.

    Absolutely not optimal...

    """
    x, y = X
    xo = float(xo)
    yo = float(yo)
    sigma_x_sqr = (fwhm_x * gaussian_fwhm_to_sigma) ** 2
    sigma_y_sqr = (fwhm_y * gaussian_fwhm_to_sigma) ** 2
    a = np.cos(theta) ** 2 / (2 * sigma_x_sqr) + np.sin(theta) ** 2 / (2 * sigma_y_sqr)
    b = np.sin(2 * theta) / (4 * sigma_x_sqr) - np.sin(2 * theta) / (4 * sigma_y_sqr)
    c = np.sin(theta) ** 2 / (2 * sigma_x_sqr) + np.cos(theta) ** 2 / (2 * sigma_y_sqr)
    g = offset + np.select(
        [np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))) > 0.5], [amplitude]
    )
    return g.ravel()


def elliptical_disk_start_params(data):
    """Get starting parameters for ellipcital disk."""
    return [
        np.nanmax(data),
        *np.unravel_index(np.nanargmax(data, axis=None), data.shape)[::-1],
        10,
        10,
        0,
        np.nanmedian(data),
    ]


def fit_gaussian(data, weight=None, func=elliptical_gaussian):
    """Fit a gaussian on a map.

    Parameters
    ----------
    data : array_like
        the input 2D map
    weight : array_like (optional)
        the corresponding weights
    func : function
        the fitted function

    Returns
    -------
    tuple :
        the optimal parameters for the fitted function or nan if problem

    Notes
    -----
    a global function will also be used to get the starting parameters with name `{func}_start_params`
    for e.g (height, x, y, width_x, width_y, theta, offset) for ellipcital_gaussian
    """
    Y, X = np.indices(data.shape)
    mask = np.isnan(data)

    x = X[~mask].flatten()
    y = Y[~mask].flatten()
    d = data[~mask].flatten()
    if weight is None:
        s = np.ones_like(d)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = 1 / np.sqrt(weight[~mask].flatten())

    start_params_func = globals().get("{}_start_params".format(func.__name__))
    if start_params_func is not None:
        params = start_params_func(data)
    else:
        from inspect import signature

        n_params = len(signature(func).parameters) - 1
        params = np.zeros(n_params)

    params = np.asarray(params)
    try:
        popt, pcov = optimize.curve_fit(func, (x, y), d, sigma=s, p0=params)
    except RuntimeError:
        popt = [np.nan] * len(params)

    return popt
