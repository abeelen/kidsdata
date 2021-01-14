import warnings
import numpy as np
import datetime

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
from .utils import _import_from

try:
    from .kiss_pointing_model import KISSPmodel

except ModuleNotFoundError:
    warnings.warn("kiss_pointing_model not installed", Warning)

    class KISSPmodel(object):
        def __init__(self, *args, **kwargs):
            warnings.warn("No pointing correction", Warning)
            pass

        def telescope2sky(self, *args):
            return args


# pylint: disable=no-member
@logged
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
    -------
    info()
        Display the basic infomation about the data file.
    read_data(list_data = 'all')
        List selected data.

    """

    def __init__(self, *args, pointing_model="KISSMateoNov2020", **kwargs):
        super().__init__(*args, **kwargs)

        self.pointing_model = pointing_model
        self.__calib = {}

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

    def calib_raw(self, calib_func="kidsdata.kids_calib.get_calfact", clean_raw=False, **kwargs):
        """Calibrate the KIDS timeline."""

        if getattr(self, "__calib", None) is None:
            self.__log.debug("calibration using {}".format(calib_func))
            self.__check_attributes(["I", "Q", "A_masq"], read_missing=False)
            calib_func = _import_from(calib_func)

            fmods = sorted([key for key in self.param_c.keys() if "modulFreq" in key])
            # Exclude null values
            fmods = [self.param_c[key] for key in fmods if self.param_c[key] != 0]
            if np.std(fmods) != 0:
                self.__log.warning("modulFreq are varying over crates  {}".format(fmods))
            fmod = fmods[0]
            self.__log.info("Calibrating with fmod={} and {}".format(fmod, kwargs))
            self.__calib = calib_func(self.I, self.Q, self.A_masq, fmod=fmod, **kwargs)
        else:
            self.__log.warning("calibrated data already present")

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for ckey in self.__calib.keys():
            self.__dict__[ckey] = self.__calib[ckey]

        if clean_raw:
            # Once calibrated, drop the raw data !!!
            for key in getattr(self, "_KidsRawData__dataSd").keys():
                delattr(self, key)
            del key
            delattr(self, "_KidsRawData__dataSd")
            # all references should be freed !!

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read so are computed...
    def __check_attributes(self, attr_list, **kwargs):
        """Check if the data has been read an attribute and read in it if not."""
        dependancies = [
            # I & Q will need A_masq
            (["I", "Q"], ["I", "Q", "A_masq"]),
            # Calibration data depends on the I, Q & A_masq raw data
            (["calfact", "Icc", "Qcc", "P0", "R0", "kidfreq", "continuum"], ["I", "Q", "A_masq"]),
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
        """List basic observation description and data set dimensions."""
        super().info()
        print("No. of interfergrams:\t", self.nint)
        print("No. of points per interfergram:\t", self.nptint)
        print("Typical size of undersampled data (MiB):\t{:3.1f}".format(self.nint * self.ndet * 32 / 8 / 1024 ** 2))

    def plot_kidpar(self, *args, **kwargs):
        fig_geometry = kids_plots.show_kidpar(self)
        fig_fwhm = kids_plots.show_kidpar_fwhm(self)
        return fig_geometry, fig_fwhm

    def plot_calib(self, *args, **kwargs):
        self.__check_attributes(["Icc", "Qcc", "calfact", "kidfreq"])
        return kids_plots.calibPlot(self, *args, **kwargs)

    def plot_pointing(self, *args, coord="tl", **kwargs):
        """Plot azimuth and elevation to check pointing."""
        # TODO: Generalize that function
        warnings.warn("Deprecated function needs update.", DeprecationWarning)
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

        Parameters
        ----------
        list_data : list of str or str
            list of data to read, see Notes
        cache : bool or 'only', optional
            use the cache file if present, by default False, see Notes
        array : function, (np.array|dask.array.from_array|None) optional
            function to apply to the largest cached value, by default np.array, if None return h5py.Dataset
        **kwargs
            additionnal parameters to be passed to the  `kidsdata.read_kidsdata.read_all`, in particular
                list_detector : list, optional
                    the list of detector indexes to be read, see Notes, by default None: read all detectors
                start : int
                    the starting block, default 0.
                end : type
                    the ending block, default full available dataset.
                silent : bool
                    Silence the output of the C library. The default is True
                diff_pps: bool
                    pre-compute pps time differences. The default is False
                correct_pps: bool
                    correct the pps signal. The default is False
                correct_time: bool or float
                    correct the time signal by interpolating jumps higher that given value in second. The default is False

        Notes
        -----
        if `cache=True`, the function reads all possible data from the cache file, and read the missing data from the raw binary file
        if `cache='only'`, the function reads all possible data from the cache file

        Depending on the `read_raw` flag when openning a kidsdata, the meaning of `list_data` and `list_detector` is changed:
        if raw is False:
            - `list_data` is a list of str within the data present in the files, see the `names` property.
            - `list_detector` is a list or array of detector names within the `kidpar` of the file. See also `get_list_detector`.
        if raw is True:
            - `list_data` is a list or array contains elements from ['Sc', 'Sd, 'Uc', 'Ud', 'Rg'].
            - `list_detector` is either a list or array of detector names within the `kidpar` of the file or
                - 'all' : to read all kids
                - 'one' or None : to read all kids of type 1
                - 'array?' : to read kids from crate/array '?'. '?' must be an int.
                - 'array_one?' : to read kids of type 1 from crate/array '?'. '?' must be an int.
                - 'box?' : to read kids from  box '?'. '?' must be an int or a letter
                - 'box_one?' : to read kids of type 1 from box '?'. '?' must be an int or a letter

                For CONCERTO, crate/array '?' must be from 2 to 3, or :
                - 'arrayT' : to read the kids from the array in transmission
                - 'arrayT_one' : to read the kids of type 1 from the array in transmission
                ' 'arrayR' : to read the kids from array in reflection
                - 'arrayR_one': to read the kids of type 1 from the array in reflection

        `None` or 'all' means read all data or detectors.
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

        if "indice" in self._KidsRawData__dataSc.keys():
            indice = self._KidsRawData__dataSc["indice"]
            assert self.nptint == np.int(indice.max() - indice.min() + 1), "Problem with 'indice' or header"

        # Support for old parameters
        if "F_azimuth" in self.__dict__ and "F_elevation" in self.__dict__:
            warnings.warn("F_azimuth and F_elevation are deprecated", DeprecationWarning)

            # Pointing have changed... from Interpolated in Sc to real sampling in Uc
            if self.F_azimuth.shape == (self.nint * self.nptint,):
                warnings.warn("Interpolated positions", PendingDeprecationWarning)
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
