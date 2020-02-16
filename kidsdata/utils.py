import warnings
import numpy as np
from scipy import optimize

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma


def project(x, y, data, shape, weights=None):
    """Project x,y, data TOIs on a 2D grid.

    Parameters
    ----------
    x, y : array_like
        input pixel indexes, 0 indexed convention
    data : array_like
        input data to project
    shape : int or tuple of int
        the shape of the output projected map
    weights : array_like
        weights to be use to sum the data (by default, ones)

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

    >>> data, weight, hits = project([-0.4, 0.4], [0, 0], [0.5, 2], 2, weights=[2, 1])
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

    if weights is None:
        weights = np.ones_like(data)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        weights = weights * ~data.mask

    _hits, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)))

    _weights, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weights)
    _data, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weights * np.asarray(data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weights

    return output, _weights, _hits.astype(np.int)


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
    g = offset + np.select([np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))) > 0.5], [amplitude])
    return g.ravel()


def elliptical_disk_start_params(data):
    """Get starting parameters for ellipcital disk."""
    return [np.nanmax(data), *np.unravel_index(np.nanargmax(data, axis=None), data.shape)[::-1], 10, 10, 0, np.nanmedian(data)]


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


def correlated_median_removal(data_array, iref=0):
    """Remove correlated data as median value per timestamp.
    
    Parameters
    ----------
    data_array : array_like
        the input 2D datablock (see note)
    iref : int
        the reference index for flat field
    
    Returns
    -------
    data_array : array_like
        the processed datablock without correlated signal, but with flat field and offset 
    flat_field : array_like
        the flat field derived from the data
    flat_offset : array_like
        the offsets derived from the data
    median_data : array_like
        the median signal removed from the data

    Notes
    -----
    detectors are on axis 0, time on axis=1
    if the reference detector is bad, this could corrump the procedure
    """

    # Remove median value in time
    flat_offset = np.nanmedian(data_array, axis=1)
    data_array = data_array - flat_offset[:, np.newaxis]

    # Compute the median flat field between all detectors
    flat_field = np.array([np.nanmedian(_data_array / data_array[iref]) for _data_array in data_array])

    # Remove the flat field value
    data_array /= flat_field[:, np.newaxis]

    # Remove the mean value per timestamp
    median_data = np.nanmedian(data_array, axis=0)
    data_array -= median_data

    # just in case
    data_array *= flat_field[:, np.newaxis]
    data_array += flat_offset[:, np.newaxis]

    return data_array, flat_field, flat_offset, median_data


# https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
def pca(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca, eigen_vals, eigen_vecs


def svd(X):
    # Data matrix X, X doesn't need to be 0-centered
    n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)  # It's not necessary to compute the full matrix of U or V
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd
