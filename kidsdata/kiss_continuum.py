import os
import warnings
import datetime
import numpy as np


from functools import lru_cache, partial
from autologging import logged
from multiprocessing import Pool

from scipy.optimize import OptimizeWarning

from astropy.table import Table, Column, MaskedColumn
from astropy.stats import mad_std
from astropy.io.fits import ImageHDU
from astropy.utils.console import ProgressBar

from .kiss_data import KissRawData
from .utils import project, build_celestial_wcs, fit_gaussian
from .utils import _import_from, cpu_count

from . import kids_plots
from .db import RE_SCAN

from .continuumdata import ContinuumData
from astropy.nddata import InverseVariance
from matplotlib import mlab


# Helper functions to pass large arrays in multiprocessing.Pool

_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _remove_polynomial(ikids, deg=None):

    global _pool_global
    (bgrd,) = _pool_global

    idx = np.arange(bgrd.shape[1])
    p = np.polynomial.polynomial.polyfit(idx, bgrd[ikids].T, deg=deg)
    return bgrd.dtype.type(bgrd[ikids] - np.polynomial.polynomial.polyval(idx, p))


def remove_polynomial(bgrds, deg):

    _this = partial(_remove_polynomial, deg=deg)
    with Pool(cpu_count(), initializer=_pool_initializer, initargs=(bgrds,),) as pool:
        output = pool.map(_this, np.array_split(np.arange(bgrds.shape[0]), cpu_count()))

    return np.vstack(output)


def _sky_to_map(ikids):

    global _pool_global
    data, offsets, az, el, wcs, shape = _pool_global

    # TODO: Shall we provide weights within a kids ?

    outputs = []
    weights = []
    hits = []

    for ikid in ikids:
        x, y = wcs.all_world2pix(az + offsets[ikid]["x0"], el + offsets[ikid]["y0"], 0)
        if isinstance(data, np.ma.MaskedArray):
            output, weight, hit = project(x, y, data[ikid].filled(0), shape, weights=~data[ikid].mask)
        else:
            output, weight, hit = project(x, y, data[ikid], shape)

        outputs.append(output)
        weights.append(weight)
        hits.append(hit)

    return np.array(outputs), np.array(weights), np.array(hits)


def sky_to_map(data, offsets, az, el, wcs, shape):

    with Pool(cpu_count(), initializer=_pool_initializer, initargs=(data, offsets, az, el, wcs, shape),) as pool:
        items = pool.map(_sky_to_map, np.array_split(np.arange(data.shape[0]), cpu_count()))

    outputs = np.vstack([item[0] for item in items])
    weights = np.vstack([item[1] for item in items])
    hits = np.vstack([item[2] for item in items])

    return outputs, weights, hits


def _psd_cal(ikids):

    global _pool_global
    datas, Fs, rebin = _pool_global

    data_psds = []
    for ikid in ikids:
        data_psd = np.array(mlab.psd(datas[ikid], Fs=Fs, NFFT=datas.shape[1] // rebin, detrend='mean')[0])
        data_psds.append(data_psd)

    return np.array(data_psds)

def psd_cal(datas, Fs, rebin, _pool_global=None):
    """psd of data"""

    _, freq = mlab.psd(datas[0], Fs=Fs, NFFT=datas.shape[1] // rebin)

    with Pool(cpu_count(), initializer=_pool_initializer, initargs=(datas, Fs, rebin),) as pool:
        items = pool.map(_psd_cal, np.array_split(np.arange(datas.shape[0]), cpu_count()))
        
    return freq, np.vstack(items)



# pylint: disable=no-member
@logged
class KissContinuum(KissRawData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @lru_cache(maxsize=3)
    def continuum_pipeline(
        self,
        ikid,
        *args,
        flatfield="amplitude",
        baseline=None,
        cm_func="kidsdata.common_mode.basic_continuum",
        **kwargs
    ):
        """Return the continuum data processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use
        flatfield: str (None|'amplitude')
            the flatfield applied to the data prior to common mode removal (default: amplitude)
        baseline : int, optionnal
            the polynomial degree of scan wise baselines to be removed
        cm_func : str
            Function to use for the common mode removal, by default 'kidsdata.common_mode.basic_continuum'

        Notes
        -----
        Any other args and kwargs are given to the pipeline function.
        ikid *must* be a tuple when calling the function, for lru_cache to work

        The flatfield values are taken from columns of the kidpar
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        self._KissRawData__check_attributes(["continuum"])

        # KIDs selection, this copy the data
        bgrd = self.continuum[ikid].copy()

        # FlatField normalization
        self.__log.info("Applying flatfield : {}".format(flatfield))
        if flatfield is None:
            flatfield = np.ones(bgrd.shape[0])
        elif flatfield in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar[flatfield].data
        else:
            raise ValueError("Can not use this flatfield : {}".format(flatfield))

        if isinstance(flatfield, (MaskedColumn, np.ma.MaskedArray)):
            flatfield = flatfield.filled(np.nan)

        bgrd /= flatfield[:, np.newaxis]

        if cm_func is not None:
            self.__log.info("Common mode removal ; {}, {}".format(cm_func, kwargs))
            cm_func = _import_from(cm_func)
            bgrd = cm_func(bgrd, *args, **kwargs)

        if baseline is not None:
            self.__log.info("Polynomial baseline per kid  of deg {}".format(baseline))
            bgrd = remove_polynomial(bgrd, baseline)

        return bgrd

    def _build_2d_wcs(self, ikid=None, wcs=None, coord="diff", cdelt=0.1, cunit="deg", **kwargs):
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
        ~astropy.wcs.WCS, tuple
            the projection wcs and shape of the resulting map
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        az, el, mask_tel = self.get_telescope_position(coord)
        good_tel = ~mask_tel
        az, el = az[good_tel], el[good_tel]

        _kidpar = self.kidpar.loc[self.list_detector[ikid]]

        # Need to include the extreme kidspar offsets
        kidspar_margin_x = (_kidpar["x0"].max() - _kidpar["x0"].min()) / cdelt
        kidspar_margin_y = (_kidpar["y0"].max() - _kidpar["y0"].min()) / cdelt

        if wcs is None:
            # Project only the telescope position
            wcs, _, _ = build_celestial_wcs(
                az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), cdelt=cdelt, cunit=cunit
            )
            # Add marging from the kidpar offsets
            wcs.wcs.crpix += (kidspar_margin_x / 2, kidspar_margin_y / 2)

        # Actually way to big to pass to all_world2pix,
        # and use too much memory...
        # az_all = (az[:, np.newaxis] + _kidpar["x0"]).T
        # el_all = (el[:, np.newaxis] + _kidpar["y0"]).T

        # Alternatively decimate ...
        # az_all = az[::az.shape[0] // 100, np.newaxis] + _kidpar["x0"].T
        # el_all = el[::el.shape[0] // 100, np.newaxis] + _kidpar["x0"].T

        # Or use the maxima, which will be too big
        az_all = (np.array([az.min(), az.max()])[:, np.newaxis] + _kidpar["x0"]).T
        el_all = (np.array([el.min(), el.max()])[:, np.newaxis] + _kidpar["y0"]).T

        # Recompute the full projected coordinates
        x, y = wcs.all_world2pix(az_all, el_all, 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        return wcs, shape

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
                kid_mask &= np.array(np.abs(std_cont / np.nanmedian(std_cont, axis=0) - 1) < std_dev)

        return kid_mask

    def continuum_map(self, ikid=None, wcs=None, shape=None, coord="diff", kid_weights="std", label=None, **kwargs):
        """Project the continuum data into one 2D map.

        Parameters
        ----------
         ikid : array, optional
            the selected kids index to consider, by default all
        wcs : ~astropy.wcs.WCS, optional
            the projection wcs if provided, by default None
        shape : tuple of int
            the shape of the resulting map
        coord : str, optional
            coordinate type, by default "diff"
        weights : (None|std|mad|key)
            the inter kid weight to use (see Note)
        label: str
            a label to insert in the meta data
        **kwargs :
            any keyword accepted by `_build_2d_wcs` or `continuum_pipeline`

        Returns
        -------
        data : `~kidsdata.continuumdata.ContinuumData`
            the resulting map

        Notes
        -----
        The weights are used to combine different kids together :
        - None : do not apply weights
        - std : standard deviation of each timeline (actually 1 / std**2)
        - mad : median absolute deviation of each timeline (actually 1 / mad**2)
        - key : any key from the kidpar table
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        az, el, mask_tel = self.get_telescope_position(coord)
        good_tel = ~mask_tel
        az, el = az[good_tel], el[good_tel]

        if wcs is None:
            self.__log.info("Computing WCS")
            wcs, _shape = self._build_2d_wcs(ikid=ikid, wcs=wcs, coord=coord, **kwargs)

        if shape is None:
            shape = _shape

        # Pipeline is here
        bgrds = self.continuum_pipeline(tuple(ikid), **kwargs)[:, good_tel]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        self.__log.info("Projecting data")
        # At this stage we have maps per kids
        offsets = self.kidpar.loc[self.list_detector[ikid]]["x0", "y0"]
        outputs, weights, hits = sky_to_map(bgrds, offsets, az, el, wcs, shape)

        self.__log.info("Computing kids weights")
        if kid_weights is None:
            kid_weights = np.ones(bgrds.shape[0])
        elif kid_weights == "std":
            kid_weights = 1 / bgrds.std(axis=1) ** 2
        elif kid_weights == "mad":
            kid_weights = 1 / mad_std(bgrds, axis=1) ** 2
        elif kid_weights in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            kid_weights = _kidpar[weights].data
        else:
            raise ValueError("Unknown weights : {}".format(weights))

        bad_weights = np.isnan(kid_weights) | np.isinf(kid_weights)
        if np.any(bad_weights):
            kid_weights[bad_weights] = 0

        if isinstance(kid_weights, np.ma.MaskedArray):
            kid_weights = kid_weights.filled(0)

        # Combine all kids, including inter kid weights
        weight = np.nansum(weights * kid_weights[:, None, None], axis=0)
        output = np.nansum(outputs * weights * kid_weights[:, None, None], axis=0) / weight
        hits = np.nansum(hits, axis=0)

        # Add standard keyword to header
        meta = self.meta

        # Add extra keyword
        meta["LABEL"] = label
        meta["N_KIDS"] = len(ikid)

        return ContinuumData(output, uncertainty=InverseVariance(weight), hits=hits, wcs=wcs, meta=meta)

    def continuum_beammaps(self, ikid=None, wcs=None, coord="diff", **kwargs):
        """Project individual detectors into square map in AltAz coordinates."""
        assert "diff" in coord, "beammaps should be done of `diff` coordinates"

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        az, el, mask_tel = self.get_telescope_position(coord)
        good_tel = ~mask_tel
        az, el = az[good_tel], el[good_tel]

        # Pipeline is here : simple baseline for now
        kwargs["flatfield"] = None
        bgrds = self.continuum_pipeline(tuple(ikid), **kwargs)[:, good_tel]

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        self.__log.info("Computing WCS")
        if wcs is None:
            wcs, x, y = build_celestial_wcs(az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), **kwargs)
        else:
            x, y = wcs.all_world2pix(az, el, 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        self.__log.info("Projecting data")
        # Null offsets for the beammap
        offsets = Table([Column(np.zeros_like(ikid), name="x0"), Column(np.zeros_like(ikid), name="y0")])
        outputs = sky_to_map(bgrds, offsets, az, el, wcs, shape)

        self.__log.info("Fitting maps")
        # Fit each of the maps
        popts = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            for data, weight in zip(outputs[0], outputs[1]):
                if np.any(~np.isnan(data)):
                    popts.append(fit_gaussian(data, weight))
                else:
                    popts.append([np.nan] * 7)

        # Convert to proper kidpar in astropy.Table
        namedet = self._kidpar.loc[self.list_detector[ikid]]["namedet"]
        kidpar = Table(np.array(popts), names=["amplitude", "x0", "y0", "fwhm_x", "fwhm_y", "theta", "offset"])

        meta = self.meta
        kidpar.meta = meta
        # By definition remove the kidpar key...
        if "KIDPAR" in kidpar.meta:
            del kidpar.meta["KIDPAR"]

        # Add additionnal keywords for database extraction
        kidpar.meta["db-start"] = kidpar.meta.get("DATE-OBS", "")
        kidpar.meta["db-end"] = kidpar.meta.get("DATE-END", "")

        # Save relative amplitude
        kidpar.meta["median_amplitude"] = np.nanmedian(kidpar["amplitude"])
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

        return outputs, wcs, kidpar, pointing_offset

    def plot_beammap(self, *args, **kwargs):
        outputs, wcs, kidpar, pointing_offset = self.continuum_beammaps(*args, **kwargs)
        return kids_plots.show_beammaps(self, outputs[0], wcs, kidpar, pointing_offset), (outputs, wcs, kidpar)

    def plot_contmap(self, *args, ikid=None, label=None, snr=False, **kwargs):
        """Plot continuum map(s), potentially with several KIDs selections."""
        if ikid is None or isinstance(ikid[0], (int, np.int, np.int64)):
            # Default to a list of list to be able to plot several maps
            ikid = [ikid]

        if kwargs.get("wcs", None) is None and kwargs.get("shape", None) is None:
            # Need to compute the global wcs here...
            if ikid[0] is None:
                ikid[0] = np.arange(len(self.list_detector))
            wcs, shape = self._build_2d_wcs(ikid=np.concatenate(ikid), **kwargs)
            kwargs["wcs"] = wcs
            kwargs["shape"] = shape

        datas = []
        for _ikid, _label in zip(ikid, label or [None] * len(ikid)):
            datas.append(self.continuum_map(*args, ikid=_ikid, label=_label, **kwargs))

        return kids_plots.show_contmap(self, datas, label, snr=snr), datas

    def plot_photometry(self, *args, **kwargs):
        return kids_plots.photometry(self, *args, **kwargs)


