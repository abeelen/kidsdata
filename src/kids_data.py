#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import astropy
import warnings
import numpy as np
from pathlib import Path
from copy import copy

from functools import lru_cache
from itertools import chain

from dateutil.parser import parse
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, join
from astropy.utils.console import ProgressBar
from astropy.coordinates import Latitude, Longitude, get_body, AltAz, EarthLocation
from astropy.io.fits import ImageHDU

from . import pipeline
from . import read_kidsdata
from . import kids_calib
from . import kids_plots
from . import kids_log
from .utils import project, build_wcs, fit_gaussian

try:
    try:
        # One should not rely on Labtools_JM_KISS
        from Labtools_JM_KISS.kiss_pointing_model import KISSPmodel
    except ModuleNotFoundError:
        from .kiss_pointing_model import KISSPmodel
except ModuleNotFoundError:

    class KISSPmodel(object):
        def __init__(self, *args, **kwargs):
            warnings.warn("No pointing correction", Warning)
            pass

        def telescope2sky(self, *args):
            return args


# TODO: should not be defined here....
# Add used observatories
EarthLocation._get_site_registry()

# Alessandro Fasano Private Comm
EarthLocation._site_registry.add_site(
    ["Quijote", "KISS"], EarthLocation(lat=0.493931966 * u.rad, lon=-0.288155867 * u.rad, height=2395 * u.m)
)
# JMP code
EarthLocation._site_registry.add_site(
    ["Teide", "Tenerife"], EarthLocation(lat=28.7569444444 * u.deg, lon=-17.8925 * u.deg, height=2390 * u.m)
)
EarthLocation._site_registry.add_site(
    ["IRAM30m", "30m", "NIKA", "NIKA2"],
    EarthLocation(lat=37.066111111111105 * u.deg, lon=-3.392777777777778 * u.deg, height=2850 * u.m),
)


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
    filename: str
        Name of the raw data file.
    obs: dict
        Basic information in observation log.
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
        self.header, self.version_header, self.param_c, self._kidpar, self.names, self.nsamples = info
        self.list_detector = np.where(~self._kidpar["index"].mask)[0]
        self._extended_kidpar = None

    def __len__(self):
        return self.nsamples

    @property
    def ndet(self):
        return len(self.list_detector)

    @property
    @lru_cache(maxsize=1)
    def obsdate(self):
        """Return the obsdate of the observation, based on filename."""
        date = Path(self.filename).name[1:].split("_")[0]
        return Time(parse(date), scale="utc", out_subfmt="date")  # UTC ??

    @property
    @lru_cache(maxsize=1)
    def scan(self):
        """Return the scan number of the observation, based on filename."""
        return int(Path(self.filename).name[1:].split("_")[2][1:])

    @property
    @lru_cache(maxsize=1)
    def source(self):
        """Return the source name, based on filename."""
        return Path(self.filename).name[1:].split("_")[3]

    @property
    @lru_cache(maxsize=1)
    def obstype(self):
        """Return the observation type, based on filename."""
        return Path(self.filename).name[1:].split("_")[4]

    def info(self):
        print("RAW DATA")
        print("==================")
        print("File name:\t" + self.filename)
        print("------------------")
        print("Source name:\t" + self.source)
        print("Observed date:\t" + self.obsdate.iso)
        print("Description:\t" + self.obstype)
        print("Scan number:\t" + self.scan)

        print("------------------")
        print("No. of KIDS detectors:\t", self.ndet)
        print("No. of time samples:\t", self.nsamples)

    def read_data(self, *args, **kwargs):
        nb_samples_read, self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd = read_kidsdata.read_all(
            self.filename, *args, **kwargs
        )

        if "list_detector" in kwargs:
            self.list_detector = kwargs["list_detector"]

        if self.nsamples != nb_samples_read:
            self.nsamples = nb_samples_read

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read some are computed...
    def __check_attributes(self, attr_list, dependancies=None):
        """Check if attributes have been read, read them if needed.

        Parameters
        ----------
        attr_list: list
            list of parameter to check
        dependancies: list of tuple
            list the parameters dependancies

        Returns
        -------
        dependancies : list
            if the dependancies keyword is set, return the list of
            requested attributes for each dependancy (see Notes)

        Notes
        -----
        If some parameters checks are inter-dependant, you can declare
        them in the dependancies list. For example, any calibrated
        quantity depends on the raw data themselves, thus, when
        checking on 'P0' or 'R0', one should actually check for 'I', 'Q' and
        'A_masq', this dependancy can be described with :

        dependancies = [(['P0', 'R0'], ['I', 'Q', 'A_masq'])]

        """
        # Adapt the list if we have dependancies
        _dependancies = []
        if dependancies:
            for request_keys, depend_keys in dependancies:
                # Check if these attributes are requested...
                _dependancy = [attr_list.pop(attr_list.index(key)) for key in request_keys if key in attr_list]
                if _dependancy:
                    # Replace them by the dependancies..
                    attr_list += depend_keys
                _dependancies.append(_dependancy)

        missing = [attr for attr in attr_list if not hasattr(self, attr) or (getattr(self, attr) is None)]
        if missing:
            # TODO: check that there attributes are present in the file
            list_data = " ".join(missing)
            print("Missing data : ", list_data)
            print("-----Now reading--------")

            self.read_data(list_data=list_data, list_detector=self.list_detector)

        # Check that everything was read
        for key in attr_list:
            assert hasattr(self, key) & (getattr(self, key) is not None), "Missing data {}".format(key)

        # Check in _dependancies that everything was read, if not we are missing something
        if _dependancies:
            missing = [
                attr for attr in chain(*_dependancies) if not hasattr(self, attr) or (getattr(self, attr) is None)
            ]
            return missing or None
        return None

    def get_list_detector(self, pattern=None):
        """Retrieve the valid detector list given a pattern.

        Attributes
        ----------
        pattern: str
            any string pattern a KID name should match in a `in` operation

        Returns
        -------
        list_detector: list
            the list which should be used for the `.read_data()` method
        """
        list_detector = np.where(~self._kidpar["index"].mask & [pattern in name for name in self._kidpar["namedet"]])[0]
        return list_detector

    @property
    def kidpar(self):
        """Retrieve the kidpar of the observation.

        Notes
        -----
        If there is an extended kidpar, it will be merge with the data kidpar

        """
        if self._extended_kidpar:
            kidpar = copy(self._kidpar)
            # astropy.table.join fails when index is masked so..
            mask = kidpar["index"].mask
            kidpar["index"].mask = False
            kidpar = join(kidpar, self._extended_kidpar, join_type="outer", keys="index")
            kidpar["index"].mask = mask
        else:
            kidpar = self._kidpar

        return kidpar


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

    def __init__(self, filename):
        super().__init__(filename)
        self.nptint = self.header.nb_pt_bloc  # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms
        self.logger = kids_log.history_logger(self.__class__.__name__)

    def calib_raw(self, *args, **kwargs):
        """Calibrate the KIDS timeline."""
        self.__check_attributes(["I", "Q", "A_masq"])

        self.calfact, self.Icc, self.Qcc, self.P0, self.R0, self.kidfreq = kids_calib.get_calfact(self, *args, **kwargs)

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read so are computed...
    def __check_attributes(self, attr_list):
        """ Check if the data has been read an attribute and read in it if not."""
        dependancies = [
            # Calibration data depends on the I, Q & A_masq raw data
            (["calfact", "Icc", "Qcc", "P0", "R0", "kidfreq"], ["I", "Q", "A_masq"]),
            # For any requested telescope position, read them all
            (
                ["F_tl_Az", "F_tl_El", "F_sky_Az", "F_sky_El", "F_diff_Az", "F_diff_El"],
                ["F_tl_Az", "F_tl_El", "F_sky_Az", "F_sky_El", "F_diff_Az", "F_diff_El"],
            ),
        ]

        _dependancies = self._KidsRawData__check_attributes(attr_list, dependancies=dependancies)

        if _dependancies is not None:
            self.calib_raw()

    @property
    @lru_cache(maxsize=1)
    def continuum(self):
        """Background based on calibration factors."""
        self.__check_attributes(["R0", "P0", "calfact"])
        return np.unwrap(self.R0 - self.P0, axis=1) * self.calfact

    @lru_cache(maxsize=2)
    def continuum_pipeline(self, ikid, *args, pipeline_func=pipeline.basic_continuum, **kwargs):
        """Return the continuum data processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use
        pipeline_function : function
            Default: pipeline.basic_continuum.

        Notes
        -----
        Any other args and kwargs are given to the pipeline function.
        ikid *must* be a tuple when calling the function, for lru_cache to work

        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        return pipeline_func(self, ikid, *args, **kwargs)

    @property
    @lru_cache(maxsize=1)
    def obstime(self):
        """Recompute the proper obs time in UTC per interferograms."""
        times = ["A_hours", "A_time_pps"]
        self._KissRawData__check_attributes(times)

        # TODO: Correction should be done at reading time
        for time in times:
            idx = np.arange(self.nsamples)
            _time = getattr(self, time)
            bad = (np.abs(np.diff(_time, append=0)) > 2) | (_time == 0)
            if any(bad):
                # Still not correct anyway, the timing goes on and off....
                func = interp1d(idx[~bad], _time[~bad], kind="linear", fill_value="extrapolate")
                _time[bad] = func(idx[bad])

        obstime = self.obsdate

        # Getting only time per interferograms here :
        return obstime + np.median((self.A_hours + self.A_time_pps).reshape((self.nint, self.nptint)), axis=1) * u.s

    @lru_cache(maxsize=2)
    def get_object_altaz(self, npoints=None):
        """Get object position interpolator."""
        if npoints is None:
            anchor_time = self.obstime
        else:
            # Find npoints between first and last observing time
            anchor_time = Time(np.linspace(*self.obstime[[0, -1]].mjd, npoints), format="mjd", scale="utc")
        alts = []
        azs = []

        for time in ProgressBar(anchor_time):
            frame = AltAz(obstime=time, location=EarthLocation.of_site("KISS"))
            coord = get_body(self.source.lower(), time).transform_to(frame)
            alts.append(coord.alt)
            azs.append(coord.az)

        alts_deg = Latitude(alts).to(u.deg).value
        azs_deg = Longitude(azs).to(u.deg).value

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

    def continuum_map(self, ikid=None, wcs=None, coord="diff", **kwargs):
        """Project all data into one map."""

        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self.__check_attributes([az_coord, el_coord])

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        mask_tel = self.mask_tel

        az = getattr(self, az_coord)[mask_tel]
        el = getattr(self, el_coord)[mask_tel]

        # Pipeline is here : simple baseline for now
        bgrds = self.continuum_pipeline(tuple(ikid), **kwargs)[:, mask_tel]
        kidspars = self.kidpar[self.list_detector[ikid]]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        if wcs is None:
            wcs, dummy_x, dummy_y = build_wcs(az, el, ctype=("OLON-GLS", "OLAT-GLS"), crval=(0, 0), **kwargs)
            x, y = [az, el] / wcs.wcs.cdelt[:, np.newaxis]
            x_min, y_min = x.min(), y.min()
            wcs.wcs.crpix = (-x_min, -y_min)
            x -= x_min
            y -= y_min
        else:
            x, y = wcs.all_world2pix(az, el, 0)

        shape = (np.round(y.max() - y.min()).astype(np.int) + 1, np.round(x.max() - x.min()).astype(np.int) + 1)

        _x = (x[:, np.newaxis] - kidspars["x0"] / wcs.wcs.cdelt[0]).T
        _y = (y[:, np.newaxis] - kidspars["y0"] / wcs.wcs.cdelt[1]).T
        output, weight, hits = project(_x.flatten(), _y.flatten(), bgrds.flatten(), shape)

        return (
            ImageHDU(output, header=wcs.to_header(), name="data"),
            ImageHDU(weight, header=wcs.to_header(), name="weight"),
            ImageHDU(hits, header=wcs.to_header(), name="hits"),
        )

    def continuum_beammaps(self, ikid=None, wcs=None, coord="diff", **kwargs):
        """Project individual detectors into square map in AltAz coordinates."""
        assert "diff" in coord, "beammaps should be done of `diff` coordinates"

        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self.__check_attributes([az_coord, el_coord])

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        mask_tel = self.mask_tel

        az = getattr(self, az_coord)[mask_tel]
        el = getattr(self, el_coord)[mask_tel]

        # Pipeline is here : simple baseline for now
        bgrds = self.continuum_pipeline(tuple(ikid), **kwargs)[:, mask_tel]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        if wcs is None:
            wcs, dummy_x, dummy_y = build_wcs(az, el, ctype=("OLON-GLS", "OLAT-GLS"), crval=(0, 0), **kwargs)
            x, y = [az, el] / wcs.wcs.cdelt[:, np.newaxis]
            x_min, y_min = x.min(), y.min()
            wcs.wcs.crpix = (-x_min, -y_min)
            x -= x_min
            y -= y_min
        else:
            x, y = wcs.all_world2pix(az, el, 0)

        shape = (np.round(y.max() - y.min()).astype(np.int) + 1, np.round(x.max() - x.min()).astype(np.int) + 1)

        outputs = []
        popts = []
        for bgrd in bgrds:
            output, weight, _ = project(x, y, bgrd, shape)
            outputs.append(output)
            if np.any(~np.isnan(output)):
                popts.append(fit_gaussian(output, weight))
            else:
                popts.append([np.nan] * 7)

        idx = self._kidpar[self.list_detector[ikid]]["index"]
        popts = Table(np.array(popts), names=["amplitude", "x0", "y0", "fwhm_x", "fwhm_y", "theta", "offset"])
        for item in ["x0", "fwhm_x"]:
            popts[item] *= wcs.wcs.cdelt[0]
        for item in ["y0", "fwhm_y"]:
            popts[item] *= wcs.wcs.cdelt[1]
        popts.add_column(idx, 0)

        return outputs, wcs, popts

    # Move most of that to __repr__ or __str__
    def info(self):
        """List basic observation description and data set dimensions."""
        super().info()
        print("No. of interfergrams:\t", self.nint)
        print("No. of points per interfergram:\t", self.nptint)

    def plot_beammap(self, *args, **kwargs):
        datas, wcs, popts = self.continuum_beammaps(*args, **kwargs)
        return kids_plots.show_beammaps(self, datas, wcs, popts), datas, wcs, popts

    def plot_kidpar(self, *args, **kwargs):
        fig_geometry = kids_plots.show_kidpar(self)
        fig_fwhm = kids_plots.show_kidpar_fwhm(self)
        return fig_geometry, fig_fwhm

    def plot_calib(self, *args, **kwargs):
        self.__check_attributes(["Icc", "Qcc", "calfact", "kidfreq"])
        return kids_plots.calibPlot(self, *args, **kwargs)

    def plot_photometry(self, *args, **kwargs):
        return kids_plots.photometry(self, *args, **kwargs)

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

            # This is for KISS only
            if "F_sky_Az" not in self.__dict__ and "F_sky_El" not in self.__dict__:
                self.F_sky_Az, self.F_sky_El = KISSPmodel().telescope2sky(self.F_tl_Az, self.F_tl_El)
                self.F_skyQ1_Az, self.F_skyQ1_El = KISSPmodel(model="Q1").telescope2sky(self.F_tl_Az, self.F_tl_El)

        if "F_tl_Az" in self.__dict__ and "F_tl_El" in self.__dict__:
            self.mask_tel = (self.F_tl_Az != 0) & (self.F_tl_El != 0)

    def spectra(self):
        # Previous processings needed: calibration
        return
