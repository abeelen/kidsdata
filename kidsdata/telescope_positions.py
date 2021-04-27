"""
Module to deal with external telescope position, either from an ascii files or from a MBFits file
"""
import warnings

from astropy.io import fits
from astropy.table import Table
from pathlib import Path
import numpy as np
from functools import lru_cache
from abc import ABCMeta, abstractmethod

from autologging import logged

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage.morphology import binary_closing

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz
from astropy.coordinates import Latitude, Longitude, EarthLocation
from astropy.utils.metadata import MetaData
from astropy.stats import mad_std

from .utils import mad_med
from .kiss_object import get_coords

# MBFITs files comes into two flavor,
# - fits file
# - fits files with grouping.fits

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


_meta_doc = """`dict`-like : Additional meta information about the dataset."""


@logged
class TelescopePositions(metaclass=ABCMeta):
    """Class to handle telescope positions."""

    mask = False
    meta = MetaData(doc=_meta_doc, copy=False)

    @abstractmethod
    def _get_positions(self, **kwargs):
        """get the raw telescope position.

        Parameters
        ----------
        **kwargs, optionnal
            keywords to be used

        Returns
        -------
        mjd : astropy.time.Time array (M,)
            the mjd of the positions
        pos : numpy.ndarray (2, M)
            the longitude and latitude of the positions
        mask : numpy.array boolean (M,)
            the corresponding flag
        """
        raise NotImplementedError()

    def _speed(self, savgol_args=(11, 3), **kwargs):

        mjd, pos, mask = self._get_positions(**kwargs)

        # Mask anomalous speed :
        savgol_kwargs = {"deriv": 1}

        speed = (
            np.sqrt(
                savgol_filter(pos[0], *savgol_args, **savgol_kwargs) ** 2
                + savgol_filter(pos[1], *savgol_args, **savgol_kwargs) ** 2
            )
            * 60
            * 60
            / (np.median(np.diff(mjd.mjd)) * 24 * 60 * 60)
        )  # arcsec / s

        return mjd, speed, mask

    def plot_speed(self, **kwargs):

        mjd, speed, mask = self._speed(**kwargs)
        speed = np.ma.array(speed, mask=mask)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot_date(mjd.plot_date, speed)
        ax.set_ylabel("Speed [arcsec/sec]")
        fig.autofmt_xdate()
        return fig

    def flag_speed(self, sigma=5, min_speed=0, **kwargs):
        """flag the telescope position based on speed deviation.

        Parameters
        ----------
        sigma : float
            the sigma-kappa threshold for the speed
        min_speed : float in arcsec / s
            the minimum speed to flag before sigma clipping, default 0
        savgol_args : tuple
            the arguments of the savgol function
        """

        mjd, speed, mask = self._speed(**kwargs)

        # speed = np.sqrt(np.diff(tabx, append=0)**2 + np.diff(taby, append=0)**2)
        # Flag small speed which sometimes could represent most of the data
        mask |= speed < min_speed
        mask = binary_closing(mask, iterations=10, border_value=1)

        med_speed, mad_speed = mad_med(speed[~mask])
        if mad_speed > 0:
            mask |= np.abs(speed - med_speed) > sigma * mad_speed

        self.__log.info("Median speed : {} arcsec / sec".format(med_speed))

        # Remove large singletons (up to 10 appart)
        mask = binary_closing(mask, iterations=10, border_value=1)

        self.__log.warning("Masking {:3.0f} % of the data from speed".format(np.mean(mask) * 100))

        self.mask = mask

    def get_interpolated_positions(self, time, delta_time=None, num=50, **kwargs):

        _mjd, pos, _mask = self._get_positions(**kwargs)

        # Scale the two mjds
        mjd = Time(time, scale=_mjd.scale).mjd
        _mjd = _mjd.mjd

        if delta_time == "auto":
            delta_time = np.median(np.diff(mjd))

        # Oversample the requested time, for later average
        if delta_time is not None:
            mjd = np.array([np.linspace(mjd_ - delta_time / 2, mjd_ + delta_time / 2, num=num) for mjd_ in mjd])

        interp1d_kwargs = {"bounds_error": False, "fill_value": np.nan, "kind": "quadratic"}
        lon = interp1d(_mjd, pos[0], **interp1d_kwargs)(mjd)
        lat = interp1d(_mjd, pos[1], **interp1d_kwargs)(mjd)

        interp1d_kwargs = {"bounds_error": False, "fill_value": 0, "kind": "slinear"}
        if isinstance(_mask, bool):
            _mask = np.repeat(_mask, _mjd.shape[0])
        mask = interp1d(_mjd, _mask, **interp1d_kwargs)(mjd).astype(np.bool)
        mask |= np.isnan(lon) | np.isnan(lat)

        if delta_time is not None:
            lon = lon.mean(axis=1)
            lat = lat.mean(axis=1)

            mask = mask.mean(axis=1) > 0.6

        return lon, lat, mask


class BasicPositions(TelescopePositions):

    mjd = None
    pos = None

    def __init__(self, mjd, pos):
        self.mjd = mjd
        self.pos = pos

    def _get_positions(self, **kwargs):
        return self.mjd, self.pos, self.mask


@logged
class MBFitsPositions(TelescopePositions):

    filename = None

    def __init__(self, filename, position_key=None):
        self.filename = Path(filename)
        self.meta["position_key"] = position_key
        self.meta["MBFits"] = str(filename)

    @property
    def _is_grouping(self):
        return (self.filename / "GROUPING.fits").exists() | (self.filename / "GROUPING.fits.gz").exists()

    @lru_cache
    def _list_extensions(self, extname):
        filename = self.filename

        if self._is_grouping:
            for name in ["GROUPING.fits", "GROUPING.fits.gz"]:
                if (filename / name).exists():
                    filename = filename / name
                    break

            grouping = fits.getdata(filename, 0)
            mask = grouping["EXTNAME"] == extname
            return [(filename.parent / str(item), 1) for item in grouping["MEMBER_LOCATION"][mask]]
        else:
            with fits.open(filename, "readonly") as hdul:
                return [(filename, i) for i, hdu in enumerate(hdul[1:], 1) if hdu.header["EXTNAME"] == extname]

    @property
    @lru_cache
    def header(self):
        filename, extnum = self._list_extensions("SCAN-MBFITS")[0]
        try:
            header = fits.getheader(filename, extnum)
        except OSError:
            self.__log.error("Can not read SCAN-MBFITS")
            header = {"TIMESYS": "TAI"}
        return header

    def header_to_meta(self):
        """Extract meta information from MBFits header.

        Following the kidsdata.read_kidsdata.filename_to_name dictionnary
        """
        return {
            "OBJECT": self.header["OBJECT"],
            "OBSTYPE": self.header["SCANMODE"],
        }

    @property
    @lru_cache
    def _monpoint(self):
        monpoint = []

        for filename, extnum in self._list_extensions("MONITOR-MBFITS"):

            data = fits.getdata(filename, extnum, memmap=True)
            monpoint += list(data["MONPOINT"])
        return set(monpoint)

    @lru_cache
    def _read_monpoint(self, key):

        if key not in self._monpoint:
            raise ValueError("Unknown MONPOINT : {}".format(key))

        mjd = []
        values = []

        for filename, extnum in self._list_extensions("MONITOR-MBFITS"):

            data = fits.getdata(filename, extnum, memmap=True)
            mask = data["MONPOINT"] == key

            _mjd = data["MJD"][mask]
            _values = np.vstack(data["MONVALUE"][mask])

            mjd.append(_mjd)
            values.append(_values)

        mjd = Time(np.hstack(mjd), format="mjd", scale=self.header["TIMESYS"].lower())

        return mjd, np.vstack(values).T

    @lru_cache
    def _read_datapar(self, key):
        mjd = []
        values = []

        if not isinstance(key, tuple):
            key = (key,)

        for filename, extnum in self._list_extensions("DATAPAR-MBFITS"):

            data = fits.getdata(filename, extnum, memmap=True)

            _mjd = data["MJD"]
            _values = np.vstack([data[field] for field in key])

            mjd.append(_mjd)
            values.append(_values)

        mjd = Time(np.hstack(mjd), format="mjd", scale=self.header["TIMESYS"].lower())

        return mjd, np.hstack(values).T

    def _get_positions(self, key=None, **kwargs):
        if key is None:
            key = self.meta["position_key"]

        if key in self._monpoint:
            return self._read_monpoint(key) + (self.mask,)

        elif key == "OFFSET_AZ_EL" and "ANTENNA_AZ_EL" in self._monpoint and "REFERENCE_AZ_EL" in self._monpoint:
            antenna_mjd, antenna_az_el = self._read_monpoint("ANTENNA_AZ_EL")
            reference_mjd, reference_az_el = self._read_monpoint("REFERENCE_AZ_EL")
            if not np.all(antenna_mjd == reference_mjd):
                raise ValueError("ANTENNA and REFERENCE MJD differs")

            offset_az_el = antenna_az_el - reference_az_el
            return antenna_mjd, offset_az_el, self.mask

        elif key == "LONGOFF_LATOFF" and "ANTENNA_AZ_EL" in self._monpoint and "REFERENCE_AZ_EL" in self._monpoint:
            antenna_mjd, antenna_az_el = self._read_monpoint("ANTENNA_AZ_EL")
            reference_mjd, reference_az_el = self._read_monpoint("REFERENCE_AZ_EL")
            if not np.all(antenna_mjd == reference_mjd):
                raise ValueError("ANTENNA and REFERENCE MJD differs")

            longoff_latoff = antenna_az_el - reference_az_el
            # Eq 2 from APEX-MPI-ICD-0002-R1_63.pdf
            longoff_latoff[0] *= np.cos(np.radians(antenna_az_el[1]))
            return antenna_mjd, longoff_latoff, self.mask

        elif key == "OFFSET_RA_DEC" and "ACTUAL_RA_DEC" in self._monpoint and "REFERENCE_RA_DEC" in self._monpoint:
            actual_mjd, actual_ra_dec = self._read_monpoint("ACTUAL_RA_DEC")
            reference_mjd, reference_ra_dec = self._read_monpoint("REFERENCE_RA_DEC")
            if not np.all(actual_mjd == reference_mjd):
                raise ValueError("ANTENNA and REFERENCE MJD differs")

            offset_ra_dec = actual_ra_dec - reference_ra_dec
            return actual_mjd, offset_ra_dec, self.mask

        else:
            raise ValueError("Can not process {}".format(key))


@logged
class KissPositions(TelescopePositions):

    mjd = None
    pos = None

    def __init__(self, mjd, pos, position_key=None, pointing_model="KISSMateoNov2020", source=None):
        """Define the KISS positions object

        Parameters
        -------
        mjd : astropy.time.Time array (M,)
            the mjd of the positions
        pos : numpy.ndarray (2, M)
            the 'tl' positions
        position_key : str ('tl'|'sky'|'pdiff', None)
            the position to be used (see Notes)
        pointing_model : str
            the pointing model to compute sky and pdiff positions
        source : str
            the source name as defined in kiss_object
        """

        # TODO: Check mjd scale... utc or tai or gps ?
        # TODO: Check that F_sky_* is not present, and that F_tl is present
        self.mjd = Time(mjd, format="mjd", scale="utc")
        self.pos = pos

        assert position_key in ["tl", "sky", "pdiff", None], "position_key must be in ('tl'|'sky'|'pdiff'|None)"
        self.meta["position_key"] = position_key

        self.meta["pointing_model"] = pointing_model
        self.meta["source"] = source

    @lru_cache(maxsize=1)
    def get_object_altaz(self, npoints=None):
        """Get object position interpolator."""
        if npoints is None:
            anchor_time = self.mjd
        else:
            # Find npoints between first and last observing time
            anchor_time = Time(np.linspace(*self.mjd[[0, -1]].mjd, npoints), format="mjd", scale="utc")

        frames = AltAz(obstime=anchor_time, location=EarthLocation.of_site("KISS"))

        coords = get_coords(self.meta["source"], anchor_time).transform_to(frames)

        alts_deg, azs_deg = Latitude(coords.alt).to("deg").value, Longitude(coords.az).to(u.deg).value

        return interp1d(anchor_time.mjd, azs_deg), interp1d(anchor_time.mjd, alts_deg)

    @property
    @lru_cache(maxsize=1)
    def _pdiff(self):
        """Return corrected diff."""

        # Fast interpolation
        obstime = self.mjd
        interp_az, interp_el = self.get_object_altaz(npoints=100)

        # Get the F_sky_Az and F_sky_El, corrected from pointing model
        _, (lon, lat), _ = self._get_positions("sky")

        _lon = (lon - interp_az(obstime.mjd)) * np.cos(np.radians(lat))
        _lat = lat - interp_el(obstime.mjd)

        return np.array([_lon, _lat])

    @property
    @lru_cache(maxsize=1)
    def _sky(self):
        _, (lon, lat), _ = self._get_positions("tl")
        _lon, _lat = KISSPmodel(model=self.meta["pointing_model"]).telescope2sky(np.array(lon), np.array(lat))
        return np.array([_lon, _lat])

    def _get_positions(self, key=None, **kwargs):
        if key is None:
            key = self.meta["position_key"]

        if key == "tl":
            return self.mjd, self.pos, self.mask
        elif key == "sky":
            return self.mjd, self._sky, self.mask
        elif key == "pdiff":
            return self.mjd, self._pdiff, self.mask
        else:
            raise ValueError("Can not retrieve {}".format(key))


@logged
class InLabPositions(TelescopePositions):

    mjd = None
    pos = None

    def __init__(self, mjd, pos, position_key=None, delta_pix=540000):

        self.meta["delta_pix"] = delta_pix

        self.mjd = Time(mjd, format="mjd", scale="utc")
        self.pos = pos

        assert position_key in ["tab", "tabdiff", None], "position_key must be in ('tab'|'tabdiff'|None)"
        self.meta["position_key"] = position_key

    def _get_positions(self, key=None, **kwargs):
        if key is None:
            key = self.meta["position_key"]

        if key == "tab":
            return self.mjd, self.pos, self.mask
        elif key == "tabdiff":
            return self.mjd, self._tabdiff(delta_pix=self.meta["delta_pix"], **kwargs), self.mask
        else:
            raise ValueError("Can not retrieve {}".format(key))

    @lru_cache(maxsize=1)
    def _tabdiff(self, delta_pix=None, speed_sigma_clipping=5, min_speed=1, plot=False):

        ## Andrea Catalano Priv. Comm 20200113 : 30 arcsec == 4.5 mm
        # Scan X15_16_Tablebt_scanStarted_12 with Lxy = 90/120, we have (masked) delta_x = 180000.0, masked_delta_y = 240000.0,
        # so we have micron and Lxy is to be understood has half the full map
        # ((4.5*u.mm) / (30*u.arcsec)).to(u.micron / u.deg) = <Quantity 540000. micron / deg>
        # So with delta_pix = 540000 with should have proper degree in _tabdiff_Az and _tabdiff_El

        # Flag the positions with speed, to better evaluate the center (and potentially delta_pix)
        if speed_sigma_clipping is not None:
            self.__log.info("Flagging tab positions")
            _ = self.flag_speed(key="tab", sigma=speed_sigma_clipping, min_speed=min_speed)

        _, (tabx, taby), mask = self._get_positions(key="tab")

        tabx = np.ma.array(tabx, mask=mask)
        taby = np.ma.array(taby, mask=mask)

        if mad_std(tabx) == 0 and mad_std(taby) == 0:
            self.__log.info("Non moving planet")
            mask = np.zeros_like(tabx, dtype=bool)
            delta_pix = 1
        elif delta_pix is None:
            # Normalization of the tab position
            delta_x = tabx.max() - tabx.min()
            delta_y = taby.max() - taby.min()
            delta_pix = np.max([delta_x, delta_y]) / 0.5  # Everything within half a degree
            self.meta["delta_pix"] = delta_pix

        self.__log.info("Converting Table position with {} mm / deg".format(delta_pix / 1e3))

        tabdiff_Az = (tabx.data - tabx.mean()) / delta_pix
        tabdiff_El = (taby.data - taby.mean()) / delta_pix

        return np.array([tabdiff_Az, tabdiff_El])


if __name__ == "__main__":
    test_dir = Path("/home/abeelen/CONCERTO_reduction/tests_mbfits/")

    test_grouping = test_dir / "APEX-30-2021-01-06-O-PP.C-NNNN-YYYY"  # has wobbler....
    test_grouping_2 = test_dir / "APEX-48698-2011-08-04-T-087.F-0001-2011"
    test_grouped = test_dir / "APEX-23419-2008-06-20-E-081.A-0269A-2008.fits"
    test_grouped_2 = test_dir / "APEXBOL.2007-04-08T09:22:12.000.fits"  # Two subscans

    mb = MBFitsPositions(test_grouped_2, position_key="LONGOFF_LATOFF")

    # Projected data by the FitsWriter (do not use wobbler):
    datapar_mjd, (longoff, latoff) = mb._read_datapar(("LONGOFF", "LATOFF"))
    longoff = np.ma.array(longoff, mask=longoff == -999)
    latoff = np.ma.array(latoff, mask=latoff == -999)

    # LONGOFF LATOFF offsets are actually GLS projected values :
    monitor_mjd, offset_lon_lat, monitor_flag = mb._get_positions()

    # Check if 2 x 1d interpolation is enough, it is
    lon, lat, mask_tel = mb.get_telescope_positions(datapar_mjd)
    interp_lon = np.ma.array(lon, mask=mask_tel)
    interp_lat = np.ma.array(lat, mask=mask_tel)

    import matplotlib.pyplot as plt

    plt.ion()

    plt.plot(longoff, latoff, label="fitswriter")
    plt.plot(interp_lon, interp_lat, label="interp1d")
    plt.plot(offset_lon_lat[0], offset_lon_lat[1], "+", label="monitor")
    plt.legend()
