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

from . import pipeline
from . import kids_calib
from . import kids_plots
from .utils import project, build_wcs, fit_gaussian
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
    def continuum(self):
        """Background based on calibration factors."""
        self.__check_attributes(["R0", "P0", "calfact"])
        # In order to catch the potential RuntimeWarning which happens when some data can not be calibrated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bgrd = np.unwrap(self.R0 - self.P0, axis=1) * self.calfact
        return bgrd

    @lru_cache(maxsize=2)
    def continuum_pipeline(self, ikid, *args, flatfield="amplitude", pipeline_func=pipeline.basic_continuum, **kwargs):
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

        # KIDs selection
        bgrd = self.continuum[ikid]

        # FlatField normalization
        if flatfield == "amplitude" and "amplitude" in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar["amplitude"]
        else:
            flatfield = np.ones(bgrd.shape[0])

        if isinstance(flatfield, MaskedColumn):
            flatfield = flatfield.filled(np.nan)

        bgrd *= flatfield[:, np.newaxis]

        return pipeline_func(self, bgrd, *args, **kwargs)

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

    def _kids_selection(self, pos_max=60, fwhm_dev=0.3, amplitude_dev=0.3, std_dev=None):
        """Select KIDs depending on their kidpar parameters.

        Parameters
        ----------
        pos_max : int, optional
            maximum position offset [arcmin] by default 60
        fwhm_dev : float, optional
            relative fwhm deviation from median [%], by default 0.3
        amplitude_dev : float, optional
            relative amplitude deviation from median [%], by default 0.3
        std_dev : float, optional
            relative devation of median background stddev [%], by default not used

        Returns
        -------
        array
            a boolean array for kids selection in the kidpar
        """
        # Retrieve used kidpar
        _kidpar = self.kidpar.loc[self.list_detector]

        pos = np.sqrt(_kidpar["x0"] ** 2 + _kidpar["y0"] ** 2) * 60  # arcmin
        fwhm = (np.abs(_kidpar["fwhm_x"]) + np.abs(_kidpar["fwhm_y"])) / 2 * 60
        median_fwhm = np.nanmedian(fwhm.filled(np.nan))
        median_amplitude = np.nanmedian(_kidpar["amplitude"].filled(np.nan))

        kid_mask = (
            (pos.filled(np.inf) < pos_max)
            & (np.abs(fwhm / median_fwhm - 1).filled(np.inf) < fwhm_dev)
            & (np.abs(_kidpar["amplitude"] / median_amplitude - 1).filled(np.inf) < amplitude_dev)
        )

        if std_dev is not None:
            # Catching warning when nan are present
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                std_cont = np.std(self.continuum, axis=1)
                kid_mask &= np.abs(std_cont / np.nanmedian(std_cont) - 1) < std_dev

        return kid_mask

    def _project_xy(self, ikid=None, wcs=None, coord="diff", cdelt=0.1, **kwargs):
        """Compute wcs and project the telescope position.

        Parameters
        ----------
        ikid : array, optional
            the selected kids index to consider, by default all
        wcs : ~astropy.wcs.WCS, optional
            the projection wcs if provided, by default None
        coord : str, optional
            coordinate type, by default "diff"
        cdelt: float
            the size of the pixels in degree

        Returns
        -------
        ~astropy.wcs.WCS, array, array, tuple
            the projection wcs, projected coordinates x, y and shape of the resulting map
        """
        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self.__check_attributes([az_coord, el_coord])

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        mask_tel = self.mask_tel

        az = getattr(self, az_coord)[mask_tel]
        el = getattr(self, el_coord)[mask_tel]

        _kidpar = self.kidpar.loc[self.list_detector[ikid]]

        # Need to include the extreme kidspar offsets
        kidspar_margin_x = (_kidpar["x0"].max() - _kidpar["x0"].min()) / cdelt
        kidspar_margin_y = (_kidpar["y0"].max() - _kidpar["y0"].min()) / cdelt

        if wcs is None:
            wcs, _, _ = build_wcs(az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), cdelt=cdelt, **kwargs)
            wcs.wcs.crpix += (kidspar_margin_x / 2, kidspar_margin_y / 2)

        az_all = (az[:, np.newaxis] + _kidpar["x0"]).T
        el_all = (el[:, np.newaxis] + _kidpar["y0"]).T

        x, y = wcs.all_world2pix(az_all, el_all, 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        return wcs, x, y, shape

    def continuum_map(self, ikid=None, wcs=None, shape=None, coord="diff", weights="std",  **kwargs):
        """Project all data into one map."""

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        mask_tel = self.mask_tel

        # Pipeline is here
        bgrds = self.continuum_pipeline(tuple(ikid), amplitude=None, **kwargs)[:, mask_tel]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        wcs, x, y, _shape = self._project_xy(ikid=ikid, wcs=wcs, coord=coord, **kwargs)

        if shape is None:
            shape = _shape

        if weights == "std":
            bgrd_weights = 1 / bgrds.std(axis=1) ** 2

        bgrd_weights = np.repeat(bgrd_weights, bgrds.shape[1]).reshape(bgrds.shape)

        output, weight, hits = project(x.flatten(), y.flatten(), bgrds.flatten(), shape, weights=bgrd_weights.flatten())

        # Add standard keyword to header
        header = wcs.to_header()
        header["OBJECT"] = self.source
        header["OBS-ID"] = self.scan
        header["FILENAME"] = self.filename
        header["EXPTIME"] = self.exptime.value
        header["DATE"] = datetime.datetime.now().isoformat()
        header["DATE-OBS"] = self.obstime[0].isot
        header["DATE-END"] = self.obstime[0].isot
        header["INSTRUME"] = self.param_c["nomexp"]
        header["AUTHOR"] = "KidsData"
        header["ORIGIN"] = "LAM"

        # Add extra keyword
        header["SCAN"] = self.scan

        return (
            ImageHDU(output, header=header, name="data"),
            ImageHDU(weight, header=header, name="weight"),
            ImageHDU(hits, header=header, name="hits"),
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
            wcs, x, y = build_wcs(az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), **kwargs)
        else:
            x, y = wcs.all_world2pix(az, el, 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        # Construct and fit all maps
        outputs = []
        popts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            for bgrd in ProgressBar(bgrds):
                output, weight, _ = project(x, y, bgrd, shape)
                outputs.append(output)
                if np.any(~np.isnan(output)):
                    popts.append(fit_gaussian(output, weight))
                else:
                    popts.append([np.nan] * 7)

        # Convert to proper kidpar in astropy.Table
        namedet = self._kidpar.loc[self.list_detector[ikid]]["namedet"]
        kidpar = Table(np.array(popts), names=["amplitude", "x0", "y0", "fwhm_x", "fwhm_y", "theta", "offset"])

        # Save relative amplitude
        kidpar["amplitude"] /= np.nanmedian(kidpar["amplitude"])

        # Positions (rather offets) are projected into the plane, backproject them to sky offsets...
        dlon, dlat = wcs.all_pix2world(kidpar["x0"], kidpar["y0"], 0)

        # wrap the angles
        dlon = (dlon + 180) % 360 - 180
        dlat = (dlat + 180) % 360 - 180

        # Remove overall pointing offsets
        pointing_offset = np.nanmedian(dlon), np.nanmedian(dlat)

        dlon = pointing_offset[0] - dlon
        dlat = pointing_offset[1] - dlat

        for item, value in zip(["x0", "y0"], [dlon, dlat]):
            kidpar[item] = value
            kidpar[item].unit = "deg"

        # Rough conversion to physical size for the widths
        for item, _cdelt in zip(["fwhm_x", "fwhm_y"], wcs.wcs.cdelt):
            kidpar[item] *= _cdelt
            kidpar[item].unit = "deg"

        kidpar["theta"].unit = "rad"
        kidpar.add_column(namedet, 0)

        kidpar.meta["scan"] = self.scan
        kidpar.meta["filename"] = self.filename
        kidpar.meta["created"] = datetime.datetime.now().isoformat()

        return outputs, wcs, kidpar, pointing_offset

    # Move most of that to __repr__ or __str__
    def info(self):
        """List basic observation description and data set dimensions."""
        super().info()
        print("No. of interfergrams:\t", self.nint)
        print("No. of points per interfergram:\t", self.nptint)

    def plot_beammap(self, *args, **kwargs):
        datas, wcs, kidpar, pointing_offset = self.continuum_beammaps(*args, **kwargs)
        return kids_plots.show_beammaps(self, datas, wcs, kidpar, pointing_offset), (datas, wcs, kidpar)

    def plot_contmap(self, *args, ikid=None, label=None, snr=False, **kwargs):
        """Plot continuum map(s), potentially with several KIDs selections."""
        if ikid is None:
            ikid = [None]
        elif isinstance(ikid[0], (int, np.int, np.int64)):
            # Default to a list of list to be able to plot several maps
            ikid = [ikid]

        if kwargs.get("wcs", None) is None and kwargs.get("shape", None) is None:
            # Need to compute the global wcs here...
            wcs, x, y, shape = self._project_xy(ikid=np.concatenate(ikid), **kwargs)
            kwargs["wcs"] = wcs
            kwargs["shape"] = shape

        data = []
        weights = []
        hits = []
        for _ikid in ikid:
            _data, _weights, _hits = self.continuum_map(ikid=_ikid, *args, **kwargs)
            data.append(_data)
            weights.append(_weights)
            hits.append(_hits)

        return kids_plots.show_contmap(self, data, weights, hits, label, snr=snr), (data, weights, hits)

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

    def spectra(self):
        # Previous processings needed: calibration
        return
