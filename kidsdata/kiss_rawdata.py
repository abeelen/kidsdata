import warnings
import numpy as np
import datetime
import logging

from functools import lru_cache

from scipy.interpolate import interp1d
from scipy.optimize import OptimizeWarning

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, MaskedColumn
from astropy.utils.console import ProgressBar
from astropy.coordinates import Latitude, Longitude, EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates.name_resolve import NameResolveError
from astropy.io.fits import ImageHDU

import h5py
from autologging import logged

from . import kids_calib
from . import kids_plots
from .kids_rawdata import KidsRawData
from .kiss_object import get_coords
from .read_kidsdata import _to_hdf5, _from_hdf5
from .utils import _import_from, pprint_list

logger = logging.getLogger(__name__)

try:
    from .kiss_pointing_model import KISSPmodel

except ModuleNotFoundError:
    logger.warning("kiss_pointing_model not installed")

    class KISSPmodel(object):
        def __init__(self, *args, **kwargs):
            logger.warning("No pointing correction")
            pass

        def telescope2sky(self, *args):
            return args


# pylint: disable=no-member
@logged
class KissRawData(KidsRawData):
    """Class dealing with KISS raw data.

    Derive from KidsRawData and add specific attributes and methods

    Attributes
    ----------
    pointing_model : str
        the name of the pointing model to use
    f_mod : float, cached
        the modulation frequency to be used for calibration
    mod_mask : array_like, cached
        the modulation mask to be used for calibration
    __calib : dict
        The calibrated data
    _pdiff_Az _pdiff_El : array_like
        The proper position offset from the observed source

    Methods
    -------
    calib_raw(calib_func="kidsdata.kids__calib.get_calfact", clean_raw=False, **kwargs)
        Calibrate the data
    get_object_altaz(npoints=None), cached
        Compute the object alt az positions for the observations
    plot_kidpar(*args, **kwargs)
        Deprecated -- Plot the current kidpar
    plot_calib(*args, **kwargs)
        Deprecated -- Plot the calibration
    plot_pointing(*args, **kwargs)
        Deprecated -- Plot the pointing
    """

    pointing_model = None

    __calib = None

    def __init__(self, *args, pointing_model="KISSMateoNov2020", **kwargs):
        """
        Parameters
        ----------
        pointing_model : str
            the key used for the KISS pointing model
        """
        super().__init__(*args, **kwargs)

        self.pointing_model = pointing_model
        self.__calib = dict()

        # Add special position keys :
        self._KidsRawData__position_keys["pdiff"] = ("_pdiff_Az", "_pdiff_El")

    def _write_data(self, filename=None, mode="a", file_kwargs=None, **kwargs):
        """write internal data to hdf5 file

        Parameters
        ----------
        filename : [str]
            output filename, default None, use the cache file name
        dataSd : bool, optional
            flag to output the fully sampled data, by default False
        mode : str
            the open mode for the h5py.File object, by default 'a'
        file_kwargs, dict, optionnal
            additionnal keyword for h5py.File object, by default None
        **kwargs
            additionnal keyword argument for the h5py.Dataset, see Note

        Notes
        -----
        Usual kwargs could be :

        kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
        """
        super()._write_data(filename, mode=mode, file_kwargs=file_kwargs, **kwargs)

        if filename is None:
            filename = self._cache_filename

        if file_kwargs is None:
            file_kwargs = {}

        # writing extra data
        with h5py.File(filename, mode="a", **file_kwargs) as f:

            if self.pointing_model:
                self.__log.debug("Saving pointing model")
                _to_hdf5(f, "pointing_model", self.pointing_model, **kwargs)
            if self.__calib:
                self.__log.debug("Saving calibrated data")
                _to_hdf5(f, "calib", self.__calib, **kwargs)

    @property
    @lru_cache(maxsize=1)
    def mod_mask(self):
        # Check the *_masq values
        self.__log.debug("Checking the *_masq arrays")
        # Retrieve the kid boxes
        masq_names = np.unique(["{}_masq".format(item[1]) for item in self.list_detector])
        self.__check_attributes(masq_names, read_missing=False)
        # Check that they are all the same
        masqs = [getattr(self, masq) for masq in masq_names]

        if np.any(np.std(masqs, axis=0) != 0):
            self.__log.error("*_masq is varying -- Please check : {}".format(pprint_list(masq_names, "_masq")))
        # cast into 8 bit, is more than enough, only 3 bits used
        masq = masqs[0].astype(np.int8)

        # AB (#CONCERTO_DAQ January 11 13:02)
        # _flag_balayage_en_cours & _flag_blanking_synthe
        # Ainsi on aura la modulation en bit0 et 1 et le flag blanking en bit
        # AB (#CONCERTO_DAQ February 11 11:07)
        # bit 1 & 2 code the modulation as a signed integer -1 0 1 : 11 00 01 ie 3 0 1
        # bit 3 is a blanking bit, which does not exist for KISS, but should not be taken into account for CONCERTO

        # Thus as a temporary fix, let's clear the 3rd bit, actually a bad idea...
        # self.__log.warning("Temporary fix : clearing the 3rd bit of masq")
        # masq = masq & ~(1 << 2)

        return masq

    @property
    @lru_cache(maxsize=1)
    def fmod(self):
        # Check on frequency modulation values, in principle one should use the one corresponding on the array/crate, but depending on the files this could lead to wrong result
        self.__log.debug("Checking the *-modulFreq values")
        fmod_names = sorted([key for key in self.param_c.keys() if "modulFreq" in key])

        # Exclude null values (CONCERTO crate 1 has 0, has it does not read kids)
        fmods = [self.param_c[key] for key in fmod_names if self.param_c[key] != 0]
        if np.std(fmods) != 0:
            self.__log.warning("modulFreq are varying over crates  {}".format(dict(zip(fmod_names, fmods))))
        return fmods[0]

    def calib_raw(self, calib_func="kidsdata.kids__calib.get_calfact", clean_raw=False, **kwargs):
        """Calibrate the KIDS timeline."""

        if getattr(self, "__calib", None) is None:
            self.__log.debug("calibration using {}".format(calib_func))
            self.__check_attributes(["I", "Q"], read_missing=False)

            fmod = self.fmod
            masq = self.mod_mask

            # Check about the 3rd bit and the fix_masq keyword
            if np.any(masq & (1 << 2)) and kwargs.get("fix_masq") is True:
                self.__log.error("fix_masq should not be used when 3rd bit is set")

            self.__log.info("Calibrating with fmod={} and {}".format(fmod, kwargs))
            calib_func = _import_from(calib_func)
            self.__calib = calib_func(self.I, self.Q, masq, fmod=fmod, **kwargs)

        else:
            self.__log.error("calibrated data already present")

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for ckey in self.__calib.keys():
            self.__dict__[ckey] = self.__calib[ckey]

        if clean_raw:
            self._clean_data("_KidsRawData__dataSd")

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read so are computed...
    def __check_attributes(self, attr_list, **kwargs):
        """Check if the data has been read an attribute and read in it if not."""
        dependancies = [
            # I & Q will need A_masq
            (["I", "Q"], ["I", "Q", "A_masq"]),
            # Calibration data depends on the I, Q & A_masq raw data
            (["calfact", "Icc", "Qcc", "P0", "R0", "interferograms", "continuum"], ["I", "Q", "A_masq"]),
            # For any requested telescope position, read them all
            (["F_tl_Az", "F_tl_El", "F_sky_Az", "F_sky_El"], ["F_tl_Az", "F_tl_El"]),
        ]

        _dependancies = self._KidsRawData__check_attributes(attr_list, dependancies=dependancies, **kwargs)

        if _dependancies is not None:
            self.calib_raw()

    @lru_cache(maxsize=2)
    def get_object_altaz(self, npoints=None):
        """Get object position interpolator."""
        if npoints is None:
            anchor_time = self.obstime
        else:
            # Find npoints between first and last observing time
            anchor_time = Time(np.linspace(*self.obstime[[0, -1]].mjd, npoints), format="mjd", scale="utc")

        frames = AltAz(obstime=anchor_time, location=EarthLocation.of_site("KISS"))

        coords = get_coords(self.source, anchor_time).transform_to(frames)

        alts_deg, azs_deg = Latitude(coords.alt).to(u.deg).value, Longitude(coords.az).to(u.deg).value

        return interp1d(anchor_time.mjd, azs_deg), interp1d(anchor_time.mjd, alts_deg)

    @property
    @lru_cache(maxsize=1)
    def _pdiff_Az(self):
        """Return corrected diff Azimuths."""
        self.__check_attributes(["F_sky_Az", "F_sky_El"])

        # Fast interpolation
        obstime = self.obstime
        interp_az, _ = self.get_object_altaz(npoints=100)
        return (self.F_sky_Az - interp_az(obstime.mjd)) * np.cos(np.radians(self.F_sky_El))

    @property
    @lru_cache(maxsize=1)
    def _pdiff_El(self):
        """Return corrected diff Elevation."""
        self.__check_attributes(["F_sky_El"])

        # Fast interpolation
        obstime = self.obstime
        _, interp_el = self.get_object_altaz(npoints=100)
        return self.F_sky_El - interp_el(obstime.mjd)

    # Move most of that to __repr__ or __str__
    def info(self):
        super().info()
        print("No. of interfergrams:\t", self.nint)
        print("No. of points per interfergram:\t", self.nptint)
        print(
            "Typical size of undersampled data (MiB + mask):\t{:3.1f} (+{:3.1f})".format(
                self.nint * self.ndet * 32 / 8 / 1024 ** 2, self.nint * self.ndet * 8 / 8 / 1024 ** 2
            )
        )

    def plot_kidpar(self, *args, **kwargs):
        fig_geometry = kids_plots.show_kidpar(self)
        fig_fwhm = kids_plots.show_kidpar_fwhm(self)
        return fig_geometry, fig_fwhm

    def plot_calib(self, *args, **kwargs):
        self.__check_attributes(["Icc", "Qcc", "calfact", "interferograms"])
        return kids_plots.calibPlot(self, *args, **kwargs)

    def plot_pointing(self, *args, coord="tl", **kwargs):
        """Plot azimuth and elevation to check pointing."""
        # TODO: Generalize that function
        logger.warning("Deprecated function needs update.", DeprecationWarning)
        self.__check_attributes(["F_{}_az".format(coord), " F_{}_el".format(coord)])
        return kids_plots.checkPointing(self, *args, **kwargs)

    @property
    def meta(self):
        """Default meta data for products."""

        meta = super().meta

        # Specific cases
        meta["POINTING"] = self.pointing_model

        return meta

    def read_data(self, *args, cache=False, array=np.array, **kwargs):
        """Read raw data.

        Also read the calibrated data in the cache file if present.

        Notes
        -----

        The different Kiss telescope positions are also handled
        """
        super().read_data(*args, cache=cache, array=array, **kwargs)

        if cache and self._cache is not None:
            self.__log.info("Reading cached data :")
            datas = []
            for data in ["calib"]:
                datas.append(_from_hdf5(self._cache, data, array=array) if data in self._cache else {})

            (calib,) = datas

            self.__log.debug("Updating dictionnaries with cached data")
            self.__calib.update(calib)

            keys = [key for data in datas for key in data]
            self.__log.debug("Read cached data : {}".format(keys))

            # Expand keys :
            # Does not double memory, but it will not be possible to
            # partially free memory : All attribute read at the same time
            # must be deleted together

            for _dict in [self.__calib]:
                for ckey in _dict:
                    self.__dict__[ckey] = _dict[ckey]

            # TODO: list_detectors and nsamples

        # In case we do not read the full file, nsamples has changed
        self.nint = self.nsamples // self.nptint

        # Default telescope mask : keep all/undersampled position per block
        self.mask_tel = np.zeros(self.nint, dtype=np.bool)

        if hasattr(self, "indice"):
            indice = self.indice
            assert self.nptint == np.int(indice.max() - indice.min() + 1), "Problem with 'indice' or header"

        # Support for old parameters
        if "F_azimuth" in self.__dict__ and "F_elevation" in self.__dict__:
            logger.warning("F_azimuth and F_elevation are deprecated", DeprecationWarning)

            # Pointing have changed... from Interpolated in Sc to real sampling in Uc
            if self.F_azimuth.shape == (self.nint * self.nptint,):
                logger.warning("Interpolated positions", PendingDeprecationWarning)
                self.F_tl_Az = np.median(self.F_azimuth.reshape((self.nint, self.nptint)), axis=1)
                self.F_tl_El = np.median(self.F_elevation.reshape((self.nint, self.nptint)), axis=1)
            elif self.F_azimuth.shape == (self.nint,):
                self.F_tl_Az = self.F_azimuth
                self.F_tl_El = self.F_elevation

        # This is for KISS only, compute pointing model corrected values
        if (
            "F_sky_Az" not in self.__dict__
            and "F_sky_El" not in self.__dict__
            and "F_tl_Az" in self.__dict__
            and "F_tl_El" in self.__dict__
        ):
            self.F_sky_Az, self.F_sky_El = KISSPmodel(model=self.pointing_model).telescope2sky(
                np.array(self.F_tl_Az), np.array(self.F_tl_El)
            )

        if "F_tl_Az" in self.__dict__ and "F_tl_El" in self.__dict__:
            self.mask_tel = (self.F_tl_Az == 0) | (self.F_tl_El == 0)
