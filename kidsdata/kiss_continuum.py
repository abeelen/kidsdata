import warnings
import datetime
import numpy as np

from functools import lru_cache

from scipy.optimize import OptimizeWarning

from astropy.table import Table, MaskedColumn
from astropy.io.fits import ImageHDU
from astropy.utils.console import ProgressBar

from .kiss_data import KissRawData
from .utils import project, build_wcs, fit_gaussian

from . import common_mode as cm
from . import kids_plots


# pylint: disable=no-member
class KissContinuum(KissRawData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @lru_cache(maxsize=1)
    def continuum(self):
        """Background based on calibration factors."""
        self._KissRawData__check_attributes(["R0", "P0", "calfact"])
        # In order to catch the potential RuntimeWarning which happens when some data can not be calibrated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bgrd = np.unwrap(self.R0 - self.P0, axis=1) * self.calfact
        return bgrd

    @lru_cache(maxsize=3)
    def continuum_pipeline(self, ikid, *args, flatfield="amplitude", cm_func=cm.basic_continuum, **kwargs):
        """Return the continuum data processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use
        flatfield: str (None |'amplitude')
            the flatfield applied to the data prior to common mode removal (default: amplitude)
        cm_function : function
            the common mode removal function (default: common_mode.basic_continuum)

        Notes
        -----
        Any other args and kwargs are given to the pipeline function.
        ikid *must* be a tuple when calling the function, for lru_cache to work

        The flatfield values are taken from the amplitude column of the kidpar
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # KIDs selection
        bgrd = self.continuum[ikid]

        # FlatField normalization
        if flatfield is None:
            flatfield = np.ones(bgrd.shape[0])
        elif flatfield == "amplitude" and "amplitude" in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar["amplitude"]
        else:
            raise ValueError("Can not use this fieldfield : {}".format(flatfield))

        if isinstance(flatfield, MaskedColumn):
            flatfield = flatfield.filled(np.nan)

        bgrd *= flatfield[:, np.newaxis]

        return cm_func(bgrd, *args, **kwargs)

    def _project_xy(self, ikid=None, wcs=None, coord="diff", cdelt=0.1, cunit="deg", **kwargs):
        """Compute wcs and project the telescope position.

        Parameters
        ----------
        ikid : array, optional
            the selected kids index to consider, by default all
        wcs : ~astropy.wcs.WCS, optional
            the projection wcs if provided, by default None
        coord : str, optional
            coordinate type, by default "diff"
        cdelt : float
            the projected pixel size
        cunit : str
            the unit of the projected pixel size

        Returns
        -------
        ~astropy.wcs.WCS, array, array, tuple
            the projection wcs, projected coordinates x, y and shape of the resulting map
        """
        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self._KissRawData__check_attributes([az_coord, el_coord])

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
            wcs, _, _ = build_wcs(
                az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), cdelt=cdelt, cunit=cunit, **kwargs
            )
            wcs.wcs.crpix += (kidspar_margin_x / 2, kidspar_margin_y / 2)

        az_all = (az[:, np.newaxis] + _kidpar["x0"]).T
        el_all = (el[:, np.newaxis] + _kidpar["y0"]).T

        x, y = wcs.all_world2pix(az_all, el_all, 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        return wcs, x, y, shape

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

    def continuum_map(self, ikid=None, wcs=None, shape=None, coord="diff", weights="std", label=None, **kwargs):
        """Project the continuum data into one 2D map."""

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
        header["DATE-END"] = self.obstime[-1].isot
        header["INSTRUME"] = self.param_c["nomexp"]
        header["AUTHOR"] = "KidsData"
        header["ORIGIN"] = "LAM"

        # Add extra keyword
        header["SCAN"] = self.scan
        header["LABEL"] = label

        return (
            ImageHDU(output, header=header, name="data_{}".format(label) if label else "data"),
            ImageHDU(weight, header=header, name="weight_{}".format(label) if label else "weight"),
            ImageHDU(hits, header=header, name="hits_{}".format(label) if label else "hits"),
        )

    def continuum_beammaps(self, ikid=None, wcs=None, coord="diff", **kwargs):
        """Project individual detectors into square map in AltAz coordinates."""
        assert "diff" in coord, "beammaps should be done of `diff` coordinates"

        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self._KissRawData__check_attributes([az_coord, el_coord])

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

    def plot_beammap(self, *args, **kwargs):
        datas, wcs, kidpar, pointing_offset = self.continuum_beammaps(*args, **kwargs)
        return kids_plots.show_beammaps(self, datas, wcs, kidpar, pointing_offset), (datas, wcs, kidpar)

    def plot_contmap(self, *args, ikid=None, label=None, snr=False, **kwargs):
        """Plot continuum map(s), potentially with several KIDs selections."""
        if ikid is None or isinstance(ikid[0], (int, np.int, np.int64)):
            # Default to a list of list to be able to plot several maps
            ikid = [ikid]

        if kwargs.get("wcs", None) is None and kwargs.get("shape", None) is None:
            # Need to compute the global wcs here...
            if ikid[0] is None:
                ikid[0] = np.arange(len(self.list_detector))
            wcs, x, y, shape = self._project_xy(ikid=np.concatenate(ikid), **kwargs)
            kwargs["wcs"] = wcs
            kwargs["shape"] = shape

        data = []
        weights = []
        hits = []
        for _ikid, _label in zip(ikid, label or [None] * len(ikid)):
            _data, _weights, _hits = self.continuum_map(*args, ikid=_ikid, label=_label, **kwargs)
            data.append(_data)
            weights.append(_weights)
            hits.append(_hits)

        return kids_plots.show_contmap(self, data, weights, hits, label, snr=snr), (data, weights, hits)

    def plot_photometry(self, *args, **kwargs):
        return kids_plots.photometry(self, *args, **kwargs)
