import os
import warnings
import numpy as np
from scipy import optimize
import importlib

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma

import multiprocessing
from multiprocessing import Pool
from itertools import zip_longest


def cpu_count():
    """Proper cpu count on a SLURM cluster."""
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()
        # ncpus = len(os.sched_getaffinity(0))

    return ncpus


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

    kwargs = {"bins": shape, "range": tuple((-0.5, size - 0.5) for size in shape)}
    # TODO: Use a worker function to split y & x over n_CPUs
    _hits, _, _ = np.histogram2d(y, x, **kwargs)

    _weights, _, _ = np.histogram2d(y, x, weights=weights, **kwargs)
    _data, _, _ = np.histogram2d(y, x, weights=weights * np.asarray(data), **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weights

    return output, _weights, _hits.astype(np.int)


def project_3d(x, y, z, data, shape, weights=None):
    """Project x,y, data TOIs on a 2D grid.

    Parameters
    ----------
    x, y, z : array_like
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

    >>> data, weight, hits = project_3d([0], [0], [0], [1], 2)
    >>> data
    array([[[ 1., nan],
            [nan, nan]],

           [[nan, nan],
            [nan, nan]]])
    >>> weight
    array([[[1., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 0.]]])
    >>> hits
    array([[[1, 0],
            [0, 0]],

           [[0, 0],
            [0, 0]]])

    >>> data, _, _ = project_3d([-0.4], [0], [1], [1], 2)
    >>> data
    array([[[nan, nan],
            [nan, nan]]])

           [[ 1., nan],
            [nan, nan]]])

    There is no test for out of shape data

    >>> data, _, _ = project_3d([-0.6, 1.6], [0, 0], [0, 0], [1, 1], 2)
    >>> data
    array([[[nan, nan],
            [nan, nan]],

           [[nan, nan],
            [nan, nan]]])

    Weighted means are also possible :

    >>> data, weight, hits = project_3d([-0.4, 0.4], [0, 0], [0, 0], [0.5, 2], 2, weights=[2, 1])
    >>> data
    array([[[ 1., nan],
            [nan, nan]],

            [[nan, nan],
            [nan, nan]]])
    >>> weight
    array([[[3., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 0.]]])
    >>> hits
    array([[[2, 0],
            [0, 0]],

           [[0, 0],
            [0, 0]]])
    """
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape, shape)
    if len(shape) == 2:
        shape = (shape[0], shape[0], shape[1])

    assert len(shape) == 3, "shape must be a int or have a length of 2 or 3"

    if weights is None:
        weights = np.ones_like(data)

    if isinstance(data, np.ma.MaskedArray):
        # Put weights as 0 for masked data
        weights = weights * ~data.mask

    kwargs = {"bins": shape, "range": tuple((-0.5, size - 0.5) for size in shape)}

    # TODO: Use a worker function to split y & x over n_CPUs
    sample = (z, y, x)
    _hits, _ = np.histogramdd(sample, **kwargs)

    _weights, _ = np.histogramdd(sample, weights=weights, **kwargs)
    _data, _ = np.histogramdd(sample, weights=weights * np.asarray(data), **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _weights

    return output, _weights, _hits.astype(np.int)


def build_celestial_wcs(
    lon, lat, crval=None, crpix=None, ctype=("TLON-TAN", "TLAT-TAN"), cdelt=0.1, cunit="deg", **kwargs
):
    """Build a celestial wcs with square pixels for the projection.

    Arguments
    ---------
    lon, lat: array_like
        input longitude and latitude in degree
    crval: tuple of 2 float
        the center of the projection in world coordinate (in cunit), by default: None, computed from the data
    crpix : tuple of 2 float
        the center of the projecttion in pixel coordinates, by default: None, computed from the data
    ctype: tuple of str
        the type of projection (default: Az/El telescope in TAN projection)
    cdelt: float (or tuple of 2 floats)
        the size of the pixels in cunit, see below, by default 0.1
    cunit: str (or tuple of 2 str)
        the unit of cdelt, by default "deg"

    Returns
    -------
    wcs: `~astropy.wcs.WCS`
        the wcs for the projection
    x, y : list of floats
        the projected positions in pixel coordinates
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ctype

    if ctype == ("TLON-TAN", "TLAT-TAN"):
        wcs.wcs.name = "Terrestrial coordinates"

    if isinstance(cdelt, (int, np.int, float, np.float, np.float32, np.float64)):
        cdelt = [-cdelt, cdelt]

        # If we are in Offsets or Terrestrial coordinate, do not flip the longitude axis
        if ctype[0][0] in ["O", "T"] and ctype[1][0] in ["O", "T"]:
            cdelt[0] = -cdelt[0]

    if isinstance(cunit, str):
        cunit = (cunit, cunit)

    wcs.wcs.cdelt = cdelt
    wcs.wcs.cunit = cunit

    if crval is None:
        # find the center of the projection
        crval = ((lon.max() + lon.min()) / 2, (lat.max() + lat.min()) / 2)

    wcs.wcs.crval = crval

    # Determine the center of the map to project all data
    if crpix is None:
        x, y = wcs.all_world2pix(lon, lat, 0)
        x_min, y_min = x.min(), y.min()
        wcs.wcs.crpix[0:2] = (-x_min, -y_min)
        x -= x_min
        y -= y_min
    else:
        wcs.wcs.crpix = crpix
        x, y = wcs.all_world2pix(lon, lat, 0)

    return wcs, x, y


def extend_wcs(wcs, data, crval=None, crpix=None, ctype="OPD", cdelt=0.1, cunit="mm"):
    """Extend a celestial wcs with a new axis.

    Parameters
    ----------
    wcs : ~astropy.wcs.WCS
        the celestial wcs to be extended
    data : array_like
        the extra data to extend the wcs
    crval: float
        the center of the projection in world coordinate (in cunit), by default: None, computed from the data
    crpix : float
        the center of the projecttion in pixel coordinate, by default: None, computed from the data
    ctype : str, optional
        the type of axis, by default "OPD"
    cdelt : float, optional
        the size of the pixels of the new axis in cunit, by default 0.1
    cunit : str, optional
        the unit of the new axis pixel, by default "mm"

    Returns
    -------
    wcs: `~astropy.wcs.WCS`
        the extended wcs
    """
    # Tick is to go to header first to extend the wcs
    header = wcs.to_header()
    naxis = header["WCSAXES"]
    header["WCSAXES"] = naxis + 1
    wcs = WCS(header)

    wcs.wcs.ctype[naxis] = ctype
    wcs.wcs.cdelt[naxis] = cdelt
    wcs.wcs.cunit[naxis] = cunit

    data_min = data.min()
    data_max = data.max()

    if crval is None:
        # find the center of the projection
        crval = (data_max + data_min) / 2

    wcs.wcs.crval[naxis] = crval

    if crpix is None:
        (z,) = wcs.sub([naxis + 1]).all_world2pix([data_min, data_max], 0)
        z_min = z.min()
        wcs.wcs.crpix[naxis] = -z_min
    else:
        wcs.wcs.crpix[naxis] = crpix

    return wcs


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


def correlated_median_removal(data_array, iref=0, offset=True, flat=True):
    """Remove correlated data as median value per timestamp.

    Parameters
    ----------
    data_array : array_like
        the input 2D datablock (see note)
    iref : int
        the reference index for flat field
    offset : bool
        compute an internal offset, by default True
    flat : bool
        compute an internal flat field, by default True

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

    if offset:
        # Remove median value in time
        flat_offset = np.nanmedian(data_array, axis=1)
        data_array -= flat_offset[:, np.newaxis]
    else:
        flat_offset = None

    if flat:
        data_ref = data_array[iref]
    elif flat < 0:
        data_ref = np.nanmedian(data_array, axis=0)
    else:
        data_ref = None

    if data_ref is not None:
        # Compute the median flat field between all detectors
        flat_field = np.array([np.nanmedian(_data_array / data_ref) for _data_array in data_array])
        # Remove the flat field value
        data_array /= flat_field[:, np.newaxis]
    else:
        flat_field = None

    # Remove the mean value per timestamp
    median_data = np.nanmedian(data_array, axis=0)
    data_array -= median_data

    # just in case
    if flat:
        data_array *= flat_field[:, np.newaxis]
    if offset:
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
    try:
        center_estimate = fit_circle_algebraic(x, y)
        center_2d, _ = optimize.leastsq(f_2b, center_estimate, args=([x, y],), Dfun=Df_2b, col_deriv=True)
        return center_2d
    except (np.linalg.LinAlgError):
        return (np.nan, np.nan)


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


def roll_fft(a, shift, axis=None):
    """Roll array eleements along a given axis using fft.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or float
        The shift to be applied, can be a float
    axis : int, (0 or 1) optionnal
        Axis along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res: ndarray
        Output array, with the same shape as `a`.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_fft(x, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_fft(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_fft(x2, 1)
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_fft(x2, -1)
    array([[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 0]])
    >>> roll_fft(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> roll_fft(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> roll_fft(x2, 1, axis=1)
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])
    >>> roll_fft(x2, -1, axis=1)
    array([[1, 2, 3, 4, 0],
           [6, 7, 8, 9, 5]])

    Notes
    -----
    Only works for 1D or 2D inputs

    """
    a = np.asanyarray(a)
    if axis is None:
        return roll_fft(a.ravel(), shift, 0).reshape(a.shape)
    else:
        shape = a.shape
        assert len(shape) < 3, "Do not work for higher dimension"
        nu_shift = np.fft.fftfreq(shape[axis]) * shift
        if len(shape) > 1 and axis == 0:
            nu_shift = nu_shift[:, np.newaxis]
        shifted_fft_a = np.fft.fft(a, axis=axis) * np.exp(-2j * np.pi * nu_shift)
        return np.real(np.fft.ifft(shifted_fft_a, axis=axis)).reshape(shape).astype(a.dtype)


# To help importing functions/class from names
def _import_from(attribute, module=None):
    """import from a string

    Parameters
    ----------
    attribute : str
        the name of the attribute to import
    module : str, optionnal
        the module to import from

    Notes
    -----
    if no module is given, the function try to infer one from the attribute name
    """

    if module is None and "." in attribute:
        module = ".".join(attribute.split(".")[:-1])
        attribute = attribute.split(".")[-1]

    module = importlib.import_module(module)
    return getattr(module, attribute)


def interferograms_regrid(interferograms, laser, bins=10, flatten=False):
    """Regrid a given set of interferograms into a common grid

    Parameters
    ----------
    interferograms : 2D array (n_blocks, n_points_per_block)
        the interferograms to be regrided
    laser : 2D array (n_blocks, n_points_per_block)
        the laser position
    bins : int or sequence of scalars or str, optional
        the binning for the regrided interferograms (see `numpy.histogram` bins)
    flatten : bool
        do we flatten the interferograms, by default False

    Returns
    -------
    output : 2D array (n_blocks, n_bins) or (n_bins, ) if flatten is True
        The regrided interferograms
    binning: 1D array (n_bins, )
        The corresponding binning
    """

    if flatten:
        laser = [laser.flatten()]
        interferograms = [interferograms.flatten()]

    _, binning = np.histogram(laser, bins=bins)

    output = []
    for _laser, _interferograms in zip(laser, interferograms):
        histo, _ = np.histogram(_laser, weights=_interferograms, bins=binning)
        hits, _ = np.histogram(_laser, bins=binning)
        with np.errstate(divide="ignore", invalid="ignore"):
            output.append(histo / hits)
    return np.squeeze(output), binning


# From
# https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
# TODO: Add multiprocessing globals
def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # TODO: Pass the arr as global and indexes in the call function
    # Chunks for the mapping (only a few chunks):
    chunks = [
        (func1d, effective_axis, sub_arr, args, kwargs) for sub_arr in np.array_split(arr, multiprocessing.cpu_count())
    ]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    # TODO: Retrieve the global and apply indexes
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def mad_med(array):
    """Return median and median absolute deviation."""
    med = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - med))
    return med, mad


# From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)
