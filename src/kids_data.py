#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import astropy
import warnings
import numpy as np
from functools import lru_cache
from astropy.wcs import WCS
from . import read_kidsdata
from . import kids_calib
from . import kids_plots

try:
    # One should not rely on Labtools_JM_KISS
    from Labtools_JM_KISS.kiss_pointing_model import KISSPmodel
except ModuleNotFoundError:

    class KISSPmodel(object):
        def __init__(self, *args, **kwargs):
            warnings.warn('No pointing correction', Warning)
            pass

        def telescope2sky(self, *args):
            return args


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
    if isinstance(shape, (int, np.integer,)):
        shape = (shape, shape)

    assert len(shape) == 2, "shape must be a int or have a length of 2"

    if weight is None:
        weight = np.ones_like(data)

    _hits, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight)
    _data, _, _ = np.histogram2d(y, x, bins=shape, range=((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5)), weights=weight * np.asarray(data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = _data / _hits

    return output


class KidsData(object):
    """ General KISS data.

    Attributes
    ----------
    filename: str
        KISS data file name.

    """

    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.filename)


class KidsRawData(KidsData):
    """ Arrays of (I,Q) with assiciated information from KISS raw data.

    Attributes
    ----------
    kidpar: :obj: Astropy.Table
        KID parameter.
    param_c: :dict
        Global parameters.
    I: array
        Stokes I measured by KID detectors.
    Q: array
        Stokes Q measured by KID detectors.
    __dataSc: obj:'ScData', optional
        Sample data set.
    __dataSd:
        Sample data set.

    Methods
    ----------
    listInfo()
        Display the basic infomation about the data file.
    read_data(list_data = 'all')
        List selected data.
    """

    def __init__(self, filename):
        self.filename = filename
        info = read_kidsdata.read_info(self.filename)
        self.header, self.version_header, self.param_c, self.kidpar, self.names, self.nsamples = info
        self.ndet = len(self.kidpar[~self.kidpar['index'].mask])  # Number of detectors.
        self.nptint = self.header.nb_pt_bloc  # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms

        # Minimum dataset
        self.I = None
        self.Q = None
        self.A_masq = None

    def __len__(self):
        return self.nsamples

    @property
    @lru_cache(maxsize=1)
    def background(self):
        """ Background based on calibration factors"""
        assert (self.R0 is not None) & \
               (self.P0 is not None) & \
               (self.calfact is not None), "Calibration factors missing"
        return np.unwrap(self.R0 - self.P0, axis=1) * self.calfact

    @property
    def beamwcs(self):
        assert (self.az_sky is not None) & \
            (self.el_sky is not None), "Sky pointing missing."

        az_sky, el_sky, mask_tel = self.az_sky, self.el_sky, self.mask_tel

        cdelt = 6 / 60  # 5 arcmin pixel size   ???
        wcs = WCS(naxis=2)
#        wcs.ctype = ["Azimuth--DEG", "ELEVATION--DEG"]
        wcs.wcs.cdelt = (cdelt, cdelt)
        wcs.wcs.crval = ((az_sky[mask_tel].max() + az_sky[mask_tel].min()) / 2,
                         (el_sky[mask_tel].max() + el_sky[mask_tel].min()) / 2)
        x, y = wcs.all_world2pix(az_sky[mask_tel], el_sky[mask_tel], 0)
        # Determine the center of the map to project all data
        x_min, y_min = x.min(), y.min()
        wcs.wcs.crpix = (x_min, y_min)

        return wcs

    @property
    @lru_cache(maxsize=1)
    def bgrs(self):
        # Rough Baseline
        bgrd = self.background
        bgrd -= np.median(bgrd, axis=1)[:, np.newaxis]
        return bgrd

    @property
    def beammap(self):
        assert (self.beamwcs is not None), "Beam wcs missing."
        assert (self.az_sky is not None) & \
            (self.el_sky is not None), "Sky pointing missing."

        bgrd = self.bgrs
        az_sky, el_sky, mask_tel = self.az_sky, self.el_sky, self.mask_tel

        cdelt = 6 / 60  # 5 arcmin pixel size   ???
        wcs = WCS(naxis=2)
#        wcs.ctype = ["Azimuth--DEG", "ELEVATION--DEG"]
        wcs.wcs.cdelt = (cdelt, cdelt)
        wcs.wcs.crval = ((az_sky[mask_tel].max() + az_sky[mask_tel].min()) / 2,
                         (el_sky[mask_tel].max() + el_sky[mask_tel].min()) / 2)
        x, y = wcs.all_world2pix(az_sky[mask_tel], el_sky[mask_tel], 0)
        # Determine the center of the map to project all data
        x_min, y_min = x.min(), y.min()
        wcs.wcs.crpix = (x_min, y_min)
        x -= x_min
        y -= y_min

        # Determine the size of the map
        shape = (np.round(y.max()).astype(np.int) + 1,
                 np.round(x.max()).astype(np.int) + 1)
        print(shape, x.max(), y.max())

        return project(x, y, bgrd[self.testikid, mask_tel], shape)

    def listInfo(self):
        print("KISS RAW DATA")
        print("==================")
        print("File name: " + self.filename)

    def read_data(self, *args, **kwargs):
        self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd \
            = read_kidsdata.read_all(self.filename, *args, **kwargs)

        if 'indice' in self.__dataSc.keys():
            assert self.nptint == np.int(self.__dataSc['indice'].max() - self.__dataSc['indice'].min() + 1), \
                "Problem with 'indice' or header"
        # Expand keys
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]

        # Convert units azimuth and elevation to degs if present
        for ckey in ['F_azimuth', 'F_elevation']:
            if ckey in self.__dict__:
                self.__dict__[ckey] = np.rad2deg(self.__dict__[ckey] / 1000.0)

        if ['F_azimuth', 'F_elevation'] <= list(self.__dict__.keys()):
            self.mask_pointing = (self.F_azimuth != 0) & (self.F_elevation != 0)

            # Pointing have changed... from Interpolated in Sc to real sampling in Uc
            if self.F_azimuth.shape == (self.nint * self.nptint, ):
                warnings.warn("Interpolated positions", PendingDeprecationWarning)
                self.az_tel = np.median(self.F_azimuth.reshape((self.nint, self.nptint)), axis=1)
                self.el_tel = np.median(self.F_elevation.reshape((self.nint, self.nptint)), axis=1)
            elif self.F_azimuth.shape == (self.nint, ):
                self.az_tel = self.F_azimuth
                self.el_tel = self.F_elevation

            self.mask_tel = (self.az_tel != 0) & (self.el_tel != 0)

            # This is for KISS only
            self.az_sky, self.el_sky = KISSPmodel().telescope2sky(self.az_tel, self.el_tel)
            self.az_skyQ1, self.el_skyQ1 = KISSPmodel(model='Q1').telescope2sky(self.az_tel, self.el_tel)

    def calib_raw(self, *args, **kwargs):
        assert (self.I is not None) & \
               (self.Q is not None) & \
               (self.A_masq is not None), "I, Q or A_masq data not present"

        self.calfact, self.Icc, self.Qcc,\
            self.P0, self.R0, self.kidfreq = kids_calib.get_calfact(self, *args, **kwargs)

    def calib_plot(self, *args, **kwargs):
        return kids_plots.calibPlot(self, *args, **kwargs)

    def pointing_plot(self, *args, **kwargs):
        """ Plot azimuth and elevation to check pointing."""

        assert (self.F_azimuth is not None) & \
               (self.F_elevation is not None), "F_azimuth or F_elevation data not present"
        return kids_plots.checkPointing(self, *args, **kwargs)

    def photometry_plot(self, *args, **kwargs):
        """  """

        assert (self.F_azimuth is not None) & \
            (self.F_elevation is not None), "Pointing(F_azimuth or F_elevation) data not present"

        return kids_plots.photometry(self, *args, **kwargs)

    def beammap_plot(self, testikid, *args, **kwarys):
        self.testikid = testikid
        return kids_plots.show_maps(self)
