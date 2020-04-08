import warnings
import numpy as np
from scipy import optimize

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma

from multiprocessing import Pool, cpu_count

from itertools import zip_longest


# From https://docs.python.org/fr/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def interp1d_nan(arr):
    """Replace nan by interpolated value in 1d array.

    Parameters
    ----------
    arr : 1d array_like
        the 1d input array with potential nans

    Returns
    -------
    arr : 1d array_like
        the same 1d array with nan linearly interpolated
    """
    nans = np.isnan(arr)
    if np.any(nans):
        x = arr.nonzero()[0]
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


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

    kwargs = {"bins": shape, "range": ((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5))}
    _hits, _, _ = np.histogram2d(y, x, **kwargs)

    _weights, _, _ = np.histogram2d(y, x, weights=weights, **kwargs)
    _data, _, _ = np.histogram2d(y, x, weights=weights * np.asarray(data), **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weights

    return output, _weights, _hits.astype(np.int)


def build_wcs(lon, lat, freq=None, crval=None, ctype=("TLON-TAN", "TLAT-TAN"), cdelt=0.1, cunit="deg", **kwargs):
    """Build the wcs for full projection.

    Arguments
    ---------
    lon, lat: array_like
        input longitude and latitude in degree
    freq : array_like (optional)
        potential 3rd axis values
    crval: tuple of 2 float
        the center of the projection in degree (default: None, computed from the data)
    ctype: tuple of str
        the type of projection (default: Az/El telescope in TAN projection)
    cdelt: float (or tuple of 2 floats)
        the size of the pixels in degree or (size of the pixels in degree, size in the 3rd axis)

    Returns
    -------
    wcs: `~astropy.wcs.WCS`
        the wcs for the projection
    x, y (,z): list of floats
        the projected positions
    """
    wcs = WCS(naxis=len(ctype))
    wcs.wcs.ctype = ctype

    if isinstance(cdelt, (float, int, np.float, np.int)):
        cdelt = (cdelt,)

    if isinstance(cunit, (str)):
        cunit = (cunit,)

    assert len(cdelt) == len(cunit), "cdelt and cunit must have the same length"

    if ctype == ("TLON-TAN", "TLAT-TAN"):
        wcs.wcs.name = "Terrestrial coordinates"

    wcs.wcs.cdelt[0:2] = (-cdelt[0], cdelt[0])
    wcs.wcs.cunit[0] = wcs.wcs.cunit[1] = cunit[0]

    # If we are in Offsets or Terrestrial coordinate, do not flip the longitude axis
    if ctype[0][0] in ["O", "T"] and ctype[1][0] in ["O", "T"]:
        wcs.wcs.cdelt[0] = cdelt[0]

    if len(cdelt) == 2:
        wcs.wcs.cdelt[2] = cdelt[1]
        wcs.wcs.cunit[2] = cunit[1]

    if crval is None:
        # find the center of the projection
        crval = ((lon.max() + lon.min()) / 2, (lat.max() + lat.min()) / 2)
        wcs.wcs.crval[0:2] = crval
    else:
        wcs.wcs.crval = crval

    wcs.wcs.cunit[0] = wcs.wcs.cunit[1] = "deg"

    # Determine the center of the map to project all data
    x, y = wcs.celestial.all_world2pix(lon, lat, 0)
    x_min, y_min = x.min(), y.min()
    wcs.wcs.crpix[0:2] = (-x_min, -y_min)
    x -= x_min
    y -= y_min

    projected = (x, y)

    if freq is not None:
        z = wcs.swapaxes(0, 2).sub(1).all_world2pix(freq, 0)[0]
        z_min = z.min()
        wcs.wcs.crpix[2] = -z_min
        z -= z_min
        projected += (z,)

    return (wcs,) + projected


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
    data_array -= flat_offset[:, np.newaxis]

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
    # TBC: This assertion fails for big array of np.float32, even within the float32 tolerances...
    # assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space, enforce original dtype
    X_pca = np.dot(X, eigen_vecs).astype(X.dtype)
    return X_pca, eigen_vals, eigen_vecs


def svd(X):
    # Data matrix X, X doesn't need to be 0-centered
    n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(
        X, full_matrices=False, compute_uv=True
    )  # It's not necessary to compute the full matrix of U or V
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd


# https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean


def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):
        w = np.random.rand(components_nr)

        for j in range(iterations):
            w_new = calculate_new_w(w, X)
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            if distance < tolerance:
                break

        W[i, :] = w

    S = np.dot(W, X)

    return S


# From https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
def radii(xy, c):
    """Compute distances from xy to c.

    Parameters
    ----------
    xy: {2, N} array_like
        the position to compute radii
    c: {2} array_like
        the position of the center

    Returns
    -------
    R {N} array_like
        the radii from xy to c
    """
    x, y = xy
    c_x, c_y = c
    return np.sqrt((x - c_x) * (x - c_x) + (y - c_y) * (y - c_y))


def fit_circle_3pts(xs, ys):
    """Fit the center of a circle using algebraic approximation with only 3 points.

    Parameters
    ----------
    xs, ys: {3, N, ...} array_like
        The 3 points to fit the {N, ...} circle(s) on

    Returns
    -------
    x_c, y_c : {N, ...} ndarray
        The center position for the {N, ...} circle(s)

    Notes
    -----
    the returned arrays will be of the same shape of the input arrays
    """
    x1, x2, x3 = xs
    y1, y2, y3 = ys

    # Speed-up trick
    dist1_sqr = (x1 * x1) + (y1 * y1)
    dist2_sqr = (x2 * x2) + (y2 * y2)
    dist3_sqr = (x3 * x3) + (y3 * y3)
    diffy12 = y1 - y2
    diffy23 = y2 - y3
    diffy31 = y3 - y1

    den = 2.0 * (x1 * diffy23 + x2 * diffy31 + x3 * diffy12)
    x_c = dist1_sqr * diffy23 + dist2_sqr * diffy31 + dist3_sqr * diffy12
    y_c = dist1_sqr * (x3 - x2) + dist2_sqr * (x1 - x3) + dist3_sqr * (x2 - x1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_c /= den
        y_c /= den

    return x_c, y_c


def fit_circle_algebraic(xs, ys):
    """Fit the center of a circle using the algebraic approximation.

    Parameters
    ----------
    xs, ys: {M, N, ...} array_like
        The M points to fit the {N, ...} circle(s) on

    Returns
    -------
    x_c, y_c : {N, ...} ndarray
        The center position for the {N, ...} circle(s)

    Notes
    -----
    the returned arrays will be of the same shape of the input arrays
    """
    x_m = xs.mean(axis=0)
    y_m = ys.mean(axis=0)

    u = xs - x_m
    v = ys - y_m

    Suv = (u * v).sum(axis=0)
    Suu = (u * u).sum(axis=0)
    Svv = (v * v).sum(axis=0)
    Suuv = (u * u * v).sum(axis=0)
    Suvv = (u * v * v).sum(axis=0)
    Suuu = (u * u * u).sum(axis=0)
    Svvv = (v * v * v).sum(axis=0)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2
    uc, uv = np.linalg.solve(A.T, B.T).T

    x_c = x_m + uc
    y_c = y_m + uv

    return x_c, y_c


def f_2b(c, xy):
    """Algebraic distance between the 2D points and the mean circle."""
    Ri = radii(xy, c)
    return Ri - Ri.mean()


def Df_2b(c, xy):
    """Jacobian of f_2b."""
    c_x, c_y = c
    x, y = xy
    df2b_dc = np.empty((len(c), x.size))

    Ri = radii(xy, c)
    df2b_dc[0] = (c_x - x) / Ri
    df2b_dc[1] = (c_y - y) / Ri
    df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

    return df2b_dc


def _pool_f2b(x, y):

    center_estimate = fit_circle_algebraic(x, y)
    center_2d, ier = optimize.leastsq(f_2b, center_estimate, args=([x, y],), Dfun=Df_2b, col_deriv=True)
    return center_2d


def fit_circle_leastsq(xs, ys):
    """Fit the center of a circle.

    Parameters
    ----------
    xs, ys: {N} list of array_like
        the N circles

    Returns
    -------
    c : {2, N} ndarray
        the center position of the N circles

    Notes
    -----
    In xs and ys, different circles do not need to have the same number of points
    """

    with Pool(cpu_count()) as pool:
        center_2d = pool.starmap(_pool_f2b, zip(xs, ys))

    return np.array(center_2d).T
