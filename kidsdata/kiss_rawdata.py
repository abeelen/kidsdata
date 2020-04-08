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

from . import kids_calib
from . import kids_plots
from .kids_data import KidsRawData
from .kiss_object import get_coords

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

    def __init__(self, filename, pointing_model="KISSNov2019"):
        super().__init__(filename)
        self.nptint = self.header.nb_pt_bloc  # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms

        self.pointing_model = pointing_model

    def calib_raw(self, calib_func=kids_calib.get_calfact, *args, **kwargs):
        """Calibrate the KIDS timeline."""
        self.__check_attributes(["I", "Q", "A_masq"], read_missing=False)

        self.calfact, self.Icc, self.Qcc, self.P0, self.R0, self.kidfreq = calib_func(self, *args, **kwargs)

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read so are computed...
    def __check_attributes(self, attr_list, **kwargs):
        """Check if the data has been read an attribute and read in it if not."""
        dependancies = [
            # I & Q will need A_masq
            (["I", "Q"], ["I", "Q", "A_masq"]),
            # Calibration data depends on the I, Q & A_masq raw data
            (["calfact", "Icc", "Qcc", "P0", "R0", "kidfreq"], ["I", "Q", "A_masq"]),
            # For any requested telescope position, read them all
            (["F_tl_Az", "F_tl_El", "F_sky_Az", "F_sky_El"], ["F_tl_Az", "F_tl_El"]),
            (["mask_tel"], ["F_tl_Az", "F_tl_El"]),
        ]

        _dependancies = self._KidsRawData__check_attributes(attr_list, dependancies=dependancies, **kwargs)

        if _dependancies is not None:
            self.calib_raw()

    @property
    @lru_cache(maxsize=1)
    def obstime(self):
        """Recompute the proper obs time in UTC per interferograms."""
        times = ["A_hours", "A_time_pps"]
        self._KissRawData__check_attributes(times)

        # TODO: These Correction should be done at reading time
        idx = np.arange(self.nsamples)

        mask = self.A_time_pps == 0

        # Masked array, to be able to unwrap properly
        A_time = np.ma.array(self.A_time_pps + self.A_hours, mask=mask)
        A_time = np.ma.array(np.unwrap(A_time), mask=mask)

        # flag all data with time difference greater than 1 second
        bad = (np.append(np.diff(np.unwrap(A_time)), 0) > 1) | A_time.mask

        if any(bad) and any(~bad):
            func = interp1d(idx[~bad], np.unwrap(A_time)[~bad], kind="linear", fill_value="extrapolate")
            A_time[bad] = func(idx[bad])

        obstime = self.obsdate

        # Getting only time per interferograms here :
        return obstime + np.median(A_time.data.reshape((self.nint, self.nptint)), axis=1) * u.s

    @property
    def exptime(self):
        return (self.obstime[-1] - self.obstime[0]).to(u.s)

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
    def F_pdiff_Az(self):
        """Return corrected diff Azimuths."""
        self.__check_attributes(["F_sky_Az", "F_sky_El"])

        # Fast interpolation
        obstime = self.obstime
        interp_az, _ = self.get_object_altaz(npoints=100)
        return (self.F_sky_Az - interp_az(obstime.mjd)) * np.cos(np.radians(self.F_sky_El))

    @property
    @lru_cache(maxsize=1)
    def F_pdiff_El(self):
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

    def plot_kidpar(self, *args, **kwargs):
        fig_geometry = kids_plots.show_kidpar(self)
        fig_fwhm = kids_plots.show_kidpar_fwhm(self)
        return fig_geometry, fig_fwhm

    def plot_calib(self, *args, **kwargs):
        self.__check_attributes(["Icc", "Qcc", "calfact", "kidfreq"])
        return kids_plots.calibPlot(self, *args, **kwargs)

    def plot_pointing(self, *args, **kwargs):
        """Plot azimuth and elevation to check pointing."""
        self.__check_attributes(["F_tl_az", " F_tl_el"])
        return kids_plots.checkPointing(self, *args, **kwargs)

    def read_data(self, *args, **kwargs):
        super().read_data(*args, **kwargs)

        # In case we do not read the full file, nsamples has changed
        self.nint = self.nsamples // self.nptint

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
                self.F_tl_Az, self.F_tl_El
            )

        if "F_tl_Az" in self.__dict__ and "F_tl_El" in self.__dict__:
            self.mask_tel = (self.F_tl_Az != 0) & (self.F_tl_El != 0)
