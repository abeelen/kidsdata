#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import astropy
import warnings
import numpy as np
from scipy import ndimage
from functools import lru_cache
from astropy.wcs import WCS
from . import read_kidsdata
from . import kids_calib
from . import kids_plots
from .utils import project, build_wcs

try:
    try:
        # One should not rely on Labtools_JM_KISS
        from Labtools_JM_KISS.kiss_pointing_model import KISSPmodel
    except ModuleNotFoundError:
        from .kiss_pointing_model import KISSPmodel
except ModuleNotFoundError:

    class KISSPmodel(object):
        def __init__(self, *args, **kwargs):
            warnings.warn('No pointing correction', Warning)
            pass

        def telescope2sky(self, *args):
            return args


class KidsData(object):
    """General KISS data.

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
    """ Arrays of (I,Q) with associated information from KIDs raw data.

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

        # Minimum dataset
        self.I = None
        self.Q = None
        self.A_masq = None

    def __len__(self):
        return self.nsamples

    def listInfo(self):
        print("RAW DATA")
        print("==================")
        print("File name: " + self.filename)

    def read_data(self, *args, **kwargs):
        self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd \
            = read_kidsdata.read_all(self.filename, *args, **kwargs)

        # Expand keys
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]


class KissRawData(KidsRawData):
    """Arrays of (I,Q) with associated information from KISS raw data.

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
        super().__init__(filename)
        self.nptint = self.header.nb_pt_bloc  # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms

    @property
    @lru_cache(maxsize=1)
    def continuum(self):
        """ Background based on calibration factors"""
        assert (self.R0 is not None) & \
               (self.P0 is not None) & \
               (self.calfact is not None), "Calibration factors missing"
        return np.unwrap(self.R0 - self.P0, axis=1) * self.calfact

    @lru_cache(maxsize=1)
    def continuum_pipeline(self, diff_mask=False):
        # Only a rough Baseline for now...
        bgrd = self.continuum
        if diff_mask:
            # Try to flag the saturated part : where the signal change too much
            diff_bgrd = np.gradient(bgrd, axis=1) / bgrd.std(axis=1)[:, np.newaxis]
            diff_mask = np.abs(diff_bgrd) > 3 * diff_bgrd.std()
            diff_mask = ndimage.binary_dilation(diff_mask, [[True, True, True]])
            bgrd = np.ma.array(bgrd, mask=diff_mask)
        bgrd -= np.median(bgrd, axis=1)[:, np.newaxis]
        return bgrd

    def continuum_beammaps(self, ikid=None, wcs=None, coord='sky', **kwargs):

        az_coord = 'F_{}_Az'.format(coord)
        el_coord = 'F_{}_El'.format(coord)
        assert (hasattr(self, az_coord) &
                (getattr(self, az_coord) is not None) &
                hasattr(self, el_coord) &
                (getattr(self, el_coord) is not None)), "Sky coordinate {} missing".format(coord)

        if ikid is None:
            ikid = slice(None)

        mask_tel = self.mask_tel

        az = getattr(self, az_coord)[mask_tel]
        el = getattr(self, el_coord)[mask_tel]

        # Pipeline is here : simple baseline for now
        bgrds = self.continuum_pipeline()[ikid, mask_tel]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        if wcs is None:
            wcs, shape = build_wcs(az, el, **kwargs)
            x, y = wcs.all_world2pix(az, el, 0)
        else:
            x, y = wcs.all_world2pix(az, el, 0)
            shape = (np.round(y.max()).astype(np.int) + 1,
                     np.round(x.max()).astype(np.int) + 1)

        outputs = []
        for bgrd in bgrds:
            outputs.append(project(x, y, bgrd, shape))

        return outputs, wcs

    def read_data(self, *args, **kwargs):
        super().read_data(*args, **kwargs)

        if 'indice' in self._KidsRawData__dataSc.keys():
            indice = self._KidsRawData__dataSc['indice']
            assert self.nptint == np.int(indice.max() - indice.min() + 1), \
                "Problem with 'indice' or header"

        # Convert units azimuth and elevation to degs if present
        for ckey in ['F_azimuth', 'F_elevation', 'F_tl_Az', 'F_tl_El', 'F_sky_Az', 'F_sky_El']:
            if ckey in self.__dict__:
                self.__dict__[ckey] = np.rad2deg(self.__dict__[ckey] / 1000.0)

        if 'F_azimuth' in self.__dict__ and 'F_elevation' in self.__dict__:
            self.mask_pointing = (self.F_azimuth != 0) & (self.F_elevation != 0)

            # Pointing have changed... from Interpolated in Sc to real sampling in Uc
            if self.F_azimuth.shape == (self.nint * self.nptint, ):
                warnings.warn("Interpolated positions", PendingDeprecationWarning)
                self.F_tl_Az = np.median(self.F_azimuth.reshape((self.nint, self.nptint)), axis=1)
                self.F_tl_El = np.median(self.F_elevation.reshape((self.nint, self.nptint)), axis=1)
            elif self.F_azimuth.shape == (self.nint, ):
                self.F_tl_Az = self.F_azimuth
                self.F_tl_El = self.F_elevation


            # This is for KISS only
            if 'F_sky_Az' not in self.__dict__ and 'F_sky_El' not in self.__dict__:
                self.F_sky_Az, self.F_sky_El = KISSPmodel().telescope2sky(self.F_tl_Az, self.F_tl_El)
                self.F_skyQ1_Az, self.F_skyQ1_El = KISSPmodel(model='Q1').telescope2sky(self.F_tl_Az, self.F_tl_El)

        if 'F_tl_Az' in self.__dict__ and 'F_tl_El' in self.__dict__:
            self.mask_tel = (self.F_tl_Az != 0) & (self.F_tl_El != 0)

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

    def beammap_plot(self, ikid, *args, **kwarys):
        return kids_plots.show_maps(self, ikid)
