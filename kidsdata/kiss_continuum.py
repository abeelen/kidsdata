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
from .utils import psd_cal

from . import kids_plots

from .continuumdata import ContinuumData
from astropy.nddata import InverseVariance
from matplotlib import mlab


# Helper functions to pass large arrays in multiprocessing.Pool

_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _remove_polynomial(ikids, deg=None, threshold=10, niter=2, **kwargs):

    global _pool_global
    (bgrd,) = _pool_global

    idx = np.arange(bgrd.shape[1])

    _w = ~np.mean(bgrd[ikids].mask, axis=(0,)).astype(np.bool)  # mean axis mask for all kids (mostly pointing flag)

    p = np.polynomial.polynomial.polyfit(idx, bgrd[ikids].T, deg=np.abs(deg), w=_w)
    residuals = bgrd.dtype.type(bgrd[ikids] - np.polynomial.polynomial.polyval(idx, p))

    if deg < 0:
        for _ in range(niter):
            # median absolute deviation threshold for source detection
            med = np.ma.median(residuals, axis=1)
            mad = np.ma.median(np.abs(residuals - med[:, None]), axis=1)
            weights = np.abs(residuals - med[:, None]) / mad[:, None] < threshold
            residuals = []
            for _bgrd, _w in zip(bgrd[ikids], weights):
                p = np.polynomial.polynomial.polyfit(idx, _bgrd, deg=np.abs(deg), w=_w)
                residuals.append(bgrd.dtype.type(_bgrd - np.polynomial.polynomial.polyval(idx, p)))
            residuals = np.ma.array(residuals)

    return residuals


def remove_polynomial(bgrds, deg, **kwargs):
    # disable mkl parallelization, more efficient on kids for large number of kids
    import mkl

    mkl_threads = mkl.set_num_threads(1)

    _this = partial(_remove_polynomial, deg=deg, **kwargs)
    with Pool(
        cpu_count(),
        initializer=_pool_initializer,
        initargs=(bgrds,),
    ) as pool:
        output = pool.map(_this, np.array_split(np.arange(bgrds.shape[0]), cpu_count()))

    mkl.set_num_threads(mkl_threads)

    return np.ma.vstack(output)


def _sky_to_map(ikids):

    global _pool_global
    data, az, el, offsets, wcs, shape = _pool_global

    # TODO: Shall we provide weights within a kids ?

    outputs = []
    weights = []
    hits = []

    for ikid in ikids:
        x, y = wcs.all_world2pix(az + offsets[ikid]["x0"], el + offsets[ikid]["y0"], 0)
        output, weight, hit = project(x, y, data[ikid], shape)

        outputs.append(output)
        weights.append(weight)
        hits.append(hit)

    return np.array(outputs), np.array(weights), np.array(hits)


def sky_to_map(data, az, el, offsets, wcs, shape):

    with Pool(
        cpu_count(),
        initializer=_pool_initializer,
        initargs=(data, az, el, offsets, wcs, shape),
    ) as pool:
        items = pool.map(_sky_to_map, np.array_split(np.arange(data.shape[0]), cpu_count()))

    outputs = np.vstack([item[0] for item in items if len(item[0]) != 0])
    weights = np.vstack([item[1] for item in items if len(item[1]) != 0])
    hits = np.vstack([item[2] for item in items if len(item[2]) != 0])

    return outputs, weights, hits


def _fit_beammaps(ikids):

    global _pool_global
    datas, weights, _ = _pool_global

    if len(ikids) == 0:
        return np.array([])

    popts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for data, weight in zip(datas[ikids], weights[ikids]):
            if np.any(~np.isnan(data)):
                popts.append(fit_gaussian(data, weight))
            else:
                popts.append([np.nan] * 7)

    return np.array(popts)


def fit_beammaps(datas):

    with Pool(cpu_count(), initializer=_pool_initializer, initargs=(datas)) as pool:
        items = pool.map(_fit_beammaps, np.array_split(np.arange(len(datas[0])), cpu_count()))

    return np.vstack([item for item in items if len(item) != 0])


# pylint: disable=no-member
@logged
class KissContinuum(KissRawData):
    """This Class deals with continuum data in KISS.

    Methods
    -------
    continuum_pipeline(*args, **kwargs)
        return the continuum data processed by given pipeline
    continuum_map(**kwargs)
        project the continuum data into one 2D map
    continuum_beammaps(**kwargs)
        project individual detectors into square map in AltAz coordinates
    plot_beammap(*args, **kwargs)
        plot beammaps
    plot_contmap(*args, **kwargs)
        plot continuum map(s), potentially with several KIDs selections.
    plot_photometry(*args, **kwargs)
        DEPRECATED -- ?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @lru_cache(maxsize=2)
    def continuum_pipeline(
        self, ikid=None, flatfield="amplitude", baseline=None, cm_func="kidsdata.common_mode.basic_continuum", **kwargs
    ):
        """Return the continuum data processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use
        flatfield: str (None|'amplitude')
            the flatfield applied to the data prior to common mode removal (default: amplitude)
        baseline : int, optionnal
            the polynomial degree of scan wise baselines to be removed, see Notes
        cm_func : str
            Function to use for the common mode removal, by default 'kidsdata.common_mode.basic_continuum'

        Notes
        -----
        Any other args and kwargs are given to the pipeline function.
        ikid *must* be a tuple when calling the function, for lru_cache to work

        The flatfield values are taken from columns of the kidpar

        A negative baseline value, means a iterative median absolute deviation thresholding, which can be controled by `threshold` and `niter` keyword
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        self._KissRawData__check_attributes(["continuum"])

        self.__log.debug("Copy continuum")
        # KIDs selection, this copy the data
        bgrd = self.continuum[ikid]

        # Add the mask from the positions
        # bgrd.mask |= self.mask_tel[None, :]
        bgrd.mask[:, self.mask_tel] = True

        # FlatField normalization
        self.__log.info("Applying flatfield : {}".format(flatfield))
        if flatfield in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar[flatfield].data
        elif flatfield is not None:
            raise ValueError("Can not use this flatfield : {}".format(flatfield))

        if isinstance(flatfield, (MaskedColumn, np.ma.MaskedArray)):
            flatfield = flatfield.filled(np.nan)

        if flatfield is not None:
            bgrd /= flatfield[:, np.newaxis]

        if cm_func is not None:
            self.__log.info("Common mode removal ; {}, {}".format(cm_func, kwargs))
            cm_func = _import_from(cm_func)
            bgrd = cm_func(bgrd, **kwargs)

        if baseline is not None:
            self.__log.info("Polynomial baseline per kid  of deg {}".format(baseline))
            bgrd = remove_polynomial(bgrd, baseline, **kwargs)

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
                std_cont = np.ma.std(self.continuum, axis=1)
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
        The kid weights are used to combine different kids together :
        - None : do not apply weights
        - std : standard deviation of each timeline (actually 1 / std**2)
        - mad : median absolute deviation of each timeline (actually 1 / mad**2)
        - key : any key from the kidpar table
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        az, el, _ = self.get_telescope_position(coord)

        if wcs is None:
            self.__log.info("Computing WCS")
            wcs, _shape = self._build_2d_wcs(ikid=ikid, wcs=wcs, coord=coord, **kwargs)

        if shape is None:
            shape = _shape

        # Pipeline is here
        self.__log.info("Continuum pipeline")
        data = self.continuum_pipeline(tuple(ikid), **kwargs)

        # In case we project only one detector
        if len(data.shape) == 1:
            data = [data]

        self.__log.info("Projecting data")
        # At this stage we have maps per kids
        offsets = self.kidpar.loc[self.list_detector[ikid]]["x0", "y0"]
        outputs, weights, hits = sky_to_map(data, az, el, offsets, wcs, shape)

        self.__log.info("Computing kids weights")
        if kid_weights is None:
            kid_weights = np.ones(data.shape[0])
        elif kid_weights == "std":
            kid_weights = 1 / data.std(axis=1) ** 2
        elif kid_weights == "mad":
            kid_weights = 1 / mad_std(data, axis=1) ** 2
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
        """Project individual detectors into square map in AltAz coordinates.

        Parameters
        ----------
        ikid : array, optional
            the selected kids index to consider, by default all
        wcs : ~astropy.wcs.WCS, optional
            the projection wcs if provided, by default None
        coord : str, optional
            coordinate type, by default "diff"
        **kwargs : dict
            Additionnal keyword arguments accepted by
            * :meth:`~kidsdata.kiss_continuum.KissContinuum.continuum_pipeline` or
            * :func:`~kidsdata.utils.build_celestial_wcs`

        Returns
        -------
        outputs : tuple of 3 numpy.ndarray (maps, weights, hits) each (nkids, ny, nx)
            the individual maps with ikid order
        wcs : ~astropy.wcs.WCS
            the corresponding wcs
        kidpar : ~astropy.table.Table
            the fitted geometry kidpar
        pointing_offset : tuple of 2 float
            the additionnal offset to the kidpar (0, 0)
        """
        assert "diff" in coord, "beammaps should be done of `diff` coordinates"

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        az, el, mask_tel = self.get_telescope_position(coord)

        self.__log.info("Continuum pipeline without flatfield")
        kwargs["flatfield"] = None
        bgrds = self.continuum_pipeline(tuple(ikid), **kwargs)

        # In case we project only one detector
        if len(bgrds.shape) == 1:
            bgrds = [bgrds]

        self.__log.info("Computing WCS")
        if wcs is None:
            wcs, x, y = build_celestial_wcs(
                az[~mask_tel], el[~mask_tel], ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), **kwargs
            )
        else:
            x, y = wcs.all_world2pix(az[~mask_tel], el[~mask_tel], 0)

        shape = (np.round(y.max()).astype(np.int) + 1, np.round(x.max()).astype(np.int) + 1)

        self.__log.info("Projecting data")
        # Null offsets for the beammap
        offsets = Table([Column(np.zeros_like(ikid), name="x0"), Column(np.zeros_like(ikid), name="y0")])
        outputs = sky_to_map(bgrds, az, el, offsets, wcs, shape)

        self.__log.info("Fitting kidpar")
        popts = fit_beammaps(outputs)

        # Convert to proper kidpar in astropy.Table
        namedet = self._kidpar.loc[self.list_detector[ikid]]["namedet"]
        kidpar = Table(popts, names=["amplitude", "x0", "y0", "fwhm_x", "fwhm_y", "theta", "offset"])

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

        fig = kids_plots.show_contmap(datas, label, snr=snr)
        fig.suptitle(self.filename)

        return fig, datas

    def plot_photometry(self, *args, **kwargs):
        return kids_plots.photometry(self, *args, **kwargs)

    def continuum_psds(self, ikid=None, rebin=1, **kwargs):
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        datas = self.continuum_pipeline(ikid=ikid, **kwargs)

        # TODO: datas is a masked array, needs to fill the blanck somehow
        datas = datas.filled(0)

        Fs = self.param_c["acqfreq"]

        freq, psds = psd_cal(datas, Fs, rebin)

        return freq, psds

    def plot_continuum_psds(self, ikid=None, rebin=1, **kwargs):
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        freq, psds = self.continuum_psds(ikid=ikid, rebin=rebin, **kwargs)

        return (
            kids_plots.plot_psd(psds, freq, self.list_detector[ikid], **kwargs),
            freq,
            psds,
        )


@logged
def kidpar_fixup(
    kidpar, amplitude=(0, 10), position=(-0.5, 0.5), fwhm=(0, 0.3), eccentricity=(0, 0.75), fixup=True, plot=False
):

    # Select based on the kidpar gaussian fit (Must be done on kidpar NOT kd.kidpar)
    fwhms = np.array([np.abs(kidpar["fwhm_x"]), np.abs(kidpar["fwhm_y"])]).T
    mean_fwhms = np.nanmean(fwhms, axis=1)

    eccentricities = np.sqrt(1 - np.min(fwhms, axis=1) ** 2 / np.max(fwhms, axis=1) ** 2)
    eccentricities[np.isnan(eccentricities)] = 1

    select_amplitude = (kidpar["amplitude"] > amplitude[0]) & (kidpar["amplitude"] < amplitude[1])
    select_positions = (
        (kidpar["x0"] > position[0])
        & (kidpar["x0"] < position[1])
        & (kidpar["y0"] > position[0])
        & (kidpar["y0"] < position[1])
    )
    select_fwhm = (mean_fwhms > fwhm[0]) & (mean_fwhms < fwhm[1])
    select_ellipticities = (eccentricities > eccentricity[0]) & (eccentricities < eccentricity[1])
    # 0.75, select bad beams, but low resolution planet maps produce bad beams

    select_kids = select_amplitude
    kidpar_fixup._log.info("amplitude select {:3.2f} %".format(select_amplitude.mean() * 100))
    select_kids &= select_positions
    kidpar_fixup._log.info(
        "positions select {:3.2f} % ({:3.2f} %)".format(select_positions.mean() * 100, select_kids.mean() * 100)
    )
    select_kids &= select_fwhm
    kidpar_fixup._log.info(
        "fwhm select {:3.2f} % ({:3.2f} %)".format(select_fwhm.mean() * 100, select_kids.mean() * 100)
    )
    select_kids &= select_ellipticities
    kidpar_fixup._log.info(
        "eccentricity select {:3.2f} % ({:3.2f} %)".format(select_ellipticities.mean() * 100, select_kids.mean() * 100)
    )

    # select_kids = select_amplitude & select_positions & select_fwhm & select_ellipticities

    if fixup:
        kidpar_fixup._log.info("Fixing positions and amplitudes")
        # Recentering :
        kidpar["x0"] -= np.median(kidpar[select_kids]["x0"])
        kidpar["y0"] -= np.median(kidpar[select_kids]["y0"])
        kidpar["amplitude"] /= np.nanmedian(kidpar[select_kids]["amplitude"])

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(ncols=3)
        axes[0].scatter(mean_fwhms, eccentricities)
        axes[0].scatter(mean_fwhms[select_kids], eccentricities[select_kids])
        axes[0].set_xlabel("mean fwhms")
        axes[0].set_xlim(0, 0.5)

        axes[0].set_ylabel("eccentricities")

        axes[1].scatter(kidpar["amplitude"], eccentricities)
        axes[1].scatter(kidpar["amplitude"][select_kids], eccentricities[select_kids])
        axes[1].set_xlabel("amplitudes")
        axes[1].set_xlim(0, 10)
        axes[1].set_ylabel("ellipiticies")

        axes[2].scatter(kidpar["x0"], kidpar["y0"])
        axes[2].scatter(kidpar["x0"][select_kids], kidpar["y0"][select_kids])
        axes[2].set_xlim(-0.5, 0.5)
        axes[2].set_ylim(-0.5, 0.5)

    return kidpar, select_kids
