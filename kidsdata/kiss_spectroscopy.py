import logging
import warnings
import numpy as np
import datetime
from enum import Enum
from copy import deepcopy

from functools import lru_cache, partial

import matplotlib.pyplot as plt

from scipy.special import erfcinv
from scipy.ndimage.morphology import binary_dilation, binary_opening
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, fftconvolve

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits
from astropy.table import MaskedColumn
from astropy.nddata import NDDataArray, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.nddata.ccddata import _known_uncertainties, _unc_name_to_cls, _unc_cls_to_name

from .kiss_data import KissRawData
from .utils import roll_fft, build_celestial_wcs, extend_wcs
from .kids_calib import ModulationValue, A_masq_to_flag

# from .utils import _pool_global, _pool_initializer
from . import common_mode as cm
from .ftsdata import FTSData

from multiprocessing import Pool, cpu_count
from autologging import logged


class LaserDirection(Enum):
    FORWARD = 1
    BACKWARD = 2


# Helper functions to pass large arrays in multiprocessing.Pool
_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _pool_opd(i):
    """Helper function to compute optical path differences in parallel."""
    global _pool_global
    laser, laser_mask, max_mask = _pool_global
    return [np.median(_laser[_that]) for _laser, _that in zip(laser[:, laser_mask], max_mask[i])]


def _pool_project_xyz_worker(ikid, **kwargs):
    """worker function to compute 3D histogram for a given detector"""
    global _pool_global
    data, weights, x, y, z = _pool_global

    ndet, nint, nptint = data.shape

    # Match weights to the data shape
    weights_shape_length = len(weights.shape)
    if weights_shape_length == 1:
        # Weights per detector
        weights = weights[:, None, None]
        repeat_weights = nint * nptint
    elif weights_shape_length == 2:
        # Weights per detector and per interferograms
        weights = weights[:, :, None]
        repeat_weights = nptint

    # x/y are repeated has we do not have the telescope position at 4 kHz
    # The np.repeat here will consume a LOT of memory
    sample = [z[ikid].flatten(), np.repeat(y[ikid], nptint), np.repeat(x[ikid], nptint)]

    hits, _ = np.histogramdd(sample, **kwargs)
    data, _ = np.histogramdd(sample, **kwargs, weights=(data[ikid] * weights[ikid]).flatten())
    weights, _ = np.histogramdd(sample, **kwargs, weights=np.repeat(weights[ikid].flatten(), repeat_weights))

    return hits, data, weights


def _pool_project_xyz(ikids, **kwargs):
    """Helper function to compute 3D histogram for a given detector list"""
    if isinstance(ikids, (int, np.int)):
        ikids = [ikids]

    ikids = iter(ikids)

    hits, data, weights = _pool_project_xyz_worker(next(ikids), **kwargs)
    for ikid in ikids:
        _hits, _data, _weights = _pool_project_xyz_worker(ikid, **kwargs)
        hits += _hits
        data += _data
        weights += _weights

    return hits, data, weights


def _pool_find_lasershifts_brute_worker(roll, _roll_func=np.roll):
    """Worker function to compute 3D histogram for a given detector list"""

    global _pool_global
    interferograms, lasers, laser_mask_forward, laser_mask_backward = _pool_global

    chi2s = []
    # Loop on interferograms
    for interferogram, laser in zip(interferograms.swapaxes(0, 1), _roll_func(lasers, roll)):
        # For all detectors
        forward = interferogram[:, laser_mask_forward]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backward = interp1d(
                laser[laser_mask_backward], interferogram[:, laser_mask_backward], fill_value=0, bounds_error=False
            )(laser[laser_mask_forward])
        chi2s.append(np.sum((forward - backward) ** 2, axis=1))
    return np.asarray(chi2s)


def _pool_find_lasershifts_brute(rolls, **kwargs):
    """Helper function to compute 3D histogram for a given detector list"""

    if isinstance(rolls, (int, np.int, float, np.float)):
        rolls = [rolls]

    if len(rolls) == 0:
        return None

    chi2 = []
    for roll in rolls:
        chi2.append(_pool_find_lasershifts_brute_worker(roll, **kwargs))

    return chi2


@logged
class KissSpectroscopy(KissRawData):
    def __init__(self, *args, laser_shift=None, optical_flip=True, mask_glitches=True, glitches_threshold=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.__laser_shift = laser_shift
        self.__optical_flip = optical_flip
        self.__mask_glitches = mask_glitches
        self.__glitches_threshold = glitches_threshold

    # TODO: This could probably be done more elegantly
    @property
    def laser_shift(self):
        return self.__laser_shift

    @laser_shift.setter
    def laser_shift(self, value):
        self.__laser_shift = value
        KissSpectroscopy.laser.fget.cache_clear()

    @property
    def optical_flip(self):
        return self.__optical_flip

    @optical_flip.setter
    def optical_flip(self, value):
        self.__optical_flip = value
        KissSpectroscopy.interferograms.fget.cache_clear()

    @property
    def mask_glitches(self):
        return self.__mask_glitches

    @mask_glitches.setter
    def mask_glitches(self, value):
        self.__mask_glitches = value
        KissSpectroscopy.interferograms.fget.cache_clear()

    @property
    def glitches_threshold(self):
        return self.__glitches_threshold

    @glitches_threshold.setter
    def glitches_threshold(self, value):
        self.__glitches_threshold = value
        KissSpectroscopy.interferograms.fget.cache_clear()

    @property
    @lru_cache(maxsize=1)
    def interferograms(self):
        """Retrieve the interferograms as a masked array.

        Returns
        -------
        interferograms : ndarray
            all the interferograms as (ndet, nint, nptint) masked array

        Notes
        -----
        The modulation and flagged from A_masq are used to mask the interferograms,
        glitches are optionnaly removed with a simple median absolute deviation threshold

        Class properties which can change its behavior :

        optical_flip : bool
            flip the B array to get the same sign in KA/KB
        mask_glitches : bool
            mask the glitches
        glitches_threshold : float
            fake detection threshold
        """
        self._KissRawData__check_attributes(["A_masq", "kidfreq"])

        self.__log.debug("Masking modulation phases")

        # TODO: Should be done elsewhere
        A_masq = self.A_masq

        # Make sure we have no issues with A_masq
        structure = np.zeros((3, 3), np.bool)
        structure[1] = True

        # A_masq has problems when != (0,1,3), binary_closing opening,
        # up to 6 iterations (see scan 800 iint=7)
        A_masq = binary_opening(A_masq * 4, structure, iterations=4)

        # Remove a bit more from A_masq, will also remove some good data : TBC
        A_masq = binary_dilation(A_masq, structure, iterations=2)

        # Make kidfreq into a masked array (copy data just in case here, should not be needed)
        # This copy the data...
        interferograms = np.ma.array(
            self.kidfreq, mask=np.tile(A_masq, self.ndet).reshape(self.kidfreq.shape), fill_value=0
        )

        if self.optical_flip:
            # By optical construction KB = -KA
            # KA = [det.startswith('KA') for det in self.list_detector]
            # KB = [det.startswith('KB') for det in self.list_detector]
            KA = np.char.startswith(self.list_detector, "KA")
            # KB = np.char.startswith(self.list_detector, 'KB')
            interferograms[KA] *= -1

        # MOVE this to interferogram_pipeline
        if self.mask_glitches:
            self.__log.debug("Masking glitches")

            # Tests -- Rough Deglitching
            # Remove the mean value per interferogram index -> left with variations only
            # BUT this do not work as there are jumps
            # kidfreq_norm = kd.kidfreq - kd.kidfreq.mean(axis=1)[:, None, :]
            # Use a gaussian filter with a width of 3 interferograms to get smooth variations along the interferograms indexes
            # from scipy.ndimage import gaussian_filter1d, gaussian_laplace
            # might flag real source
            # # kidfreq_norm = kd.kidfreq - gaussian_filter1d(kd.kidfreq, 3, axis=1)

            # Try something else on the modulation flagged interferograms
            max_abs_interferogram = np.abs(interferograms).max(axis=2)

            # Threshold for gaussian statistic, along the interferogram axis only :
            sigma = erfcinv(self.glitches_threshold / np.product(interferograms.shape[1])) * np.sqrt(2)

            cutoffs = max_abs_interferogram.mean(axis=1) + sigma * max_abs_interferogram.std(axis=1)
            glitches_mask = np.abs(interferograms) > cutoffs[:, None, None]

            interferograms.mask = interferograms.mask | glitches_mask

        return interferograms

    @property
    @lru_cache(maxsize=1)
    def laser(self):
        """Retrieve the laser position with missing value interpolated.

        Returns
        -------
        laser : ndarray (nint, nptint)
            the laser positions with shape

        Notes
        -----
        This depends on the `laser_shift` property.
        """
        self._KissRawData__check_attributes(["C_laser1_pos", "C_laser2_pos"])

        self.__log.debug("Computing laser position with {} shift".format(self.laser_shift))

        laser1 = self.C_laser1_pos.flatten()
        laser2 = self.C_laser2_pos.flatten()

        # Mirror positions are acquired at half he acquisition frequency thus...
        assert np.all(laser1[::2] == laser1[1::2])
        assert np.all(laser2[::2] == laser2[1::2])

        # ... we can sum the two positions to lower the noise ...
        laser = (laser1 + laser2) / 2

        # and quadratictly interpolate the second position...
        laser[1::2] = interp1d(range(len(laser) // 2), laser[::2], kind="quadratic", fill_value="extrapolate")(
            np.arange(len(laser) // 2) + 0.5
        )

        if self.laser_shift is not None:
            ## TODO: different cases depending on the shape of laser_shift
            laser = roll_fft(laser, self.laser_shift)

        # Same shape as kidfreq[0]
        laser = laser.reshape(self.nint, self.nptint)

        return laser

    @property
    @lru_cache(maxsize=1)
    def laser_directions(self):
        """Get laser forward/backward mask.

        Returns
        -------
        laser_directions : array_like (nptint)
            the laser directions as listed in `LaserDirection`

        Notes
        -----
        This depends on the `laser_shift` property.
        """

        laser = self.laser

        # Find the forward and backward phases
        # rough cut on the the mean mirror position : find the large peaks of the derivatives
        turnovers, _ = find_peaks(-np.abs(np.diff(laser.mean(axis=0))), width=[100])

        assert len(turnovers) == 2

        laser_directions = np.zeros(self.nptint, dtype=np.int8)
        laser_directions[turnovers[0] :] = LaserDirection.FORWARD.value
        laser_directions[turnovers[1] :] = LaserDirection.BACKWARD.value

        return laser_directions

    def find_lasershifts_brute(
        self, ikid=None, min_roll=-10, max_roll=10, n_roll=21, roll_func=np.roll, plot=False, mode="single"
    ):
        """Find potential shift between mirror position and interferograms timeline.

        Brute force approach by computing min Chi2 value between rolled forward and backward interferograms

        Parameters
        ----------
        ikid : tuple (optional)
            The list of kid index in self.list_detector to use (default: all)
        min_roll, max_roll : float or int
            the minimum and maximum shift to consider
        n_roll : int
            the number of rolls between those two values
        roll_func : function (np.roll|roll_fft)
            the rolling function to be used np.roll for integer rolls and roll_fft for floating values
        plot : bool
            display so debugging plots
        mode : str (single|per_det|per_int|per_det_int)
            return value mode (see Notes)

        Returns
        -------
        lasershifts : array_like
            the lasershift which shape depends on the selected mode
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        interferograms = self.interferograms[ikid].filled(0)
        lasers = self.laser
        laser_mask = self.laser_directions

        # TODO: Remove polynomial baseline
        # DO NOT WORK
        # int_idx = np.arange(_interferograms.shape[-1])
        # _this = partial(lambda x, p: np.polyval(p, x), int_idx)
        # for _interferogram in _interferograms:
        #     p = np.polyfit(int_idx, _interferogram.T, deg=2)
        #     baseline = np.asarray(list(map(_this, p.T)))
        #     _interferogram -= baseline

        if roll_func is np.roll:
            rolls = np.linspace(min_roll, max_roll, n_roll, dtype=int)
        elif roll_func is roll_fft:
            rolls = np.linspace(min_roll, max_roll, n_roll)

        # laser_rolls = []
        # for roll in rolls:
        #     laser_rolls.append(find_shift__roll_chi2(interferograms, laser, laser_mask, roll))
        _this = partial(_pool_find_lasershifts_brute, _roll_func=roll_func)
        with Pool(
            cpu_count(),
            initializer=_pool_initializer,
            initargs=(
                interferograms,
                lasers,
                laser_mask == LaserDirection.FORWARD.value,
                laser_mask == LaserDirection.BACKWARD.value,
            ),
        ) as pool:
            laser_rolls = pool.map(_this, np.array_split(rolls, cpu_count()))

        # At this stages (n_roll, nint, ndet) -> (ndet, nint, n_roll)
        laser_rolls = np.concatenate([_this for _this in laser_rolls if _this is not None]).transpose(2, 1, 0)

        lasershifts = rolls[np.argmin(laser_rolls, axis=-1)]

        if mode == "single":
            lasershifts = np.median(lasershifts)
        elif mode == "per_det":
            lasershifts = np.median(lasershifts, axis=1)
        elif mode == "per_int":
            lasershifts = np.median(lasershifts, axis=0)
        elif mode == "per_det_int":
            pass
        else:
            raise ValueError("Unknown mode : {}".format(mode))

        if plot:
            from matplotlib.gridspec import GridSpec

            fig, axes = plt.subplots(ncols=2)
            axes[0].imshow(
                np.log(laser_rolls.mean(axis=1)),
                aspect="auto",
                extent=(np.min(rolls) - 0.5, np.max(rolls) + 0.5, 0, len(laser_rolls)),
            )
            axes[0].set_xlabel("sample roll")
            axes[0].set_title("Chi2 with rolling")
            axes[0].axvline(1, c="r")
            axes[1].semilogy(rolls, laser_rolls.mean(axis=(0, 1)))
            axes[1].axvline(1, c="r")
            axes[1].axvline(rolls[np.argmin(laser_rolls.mean(axis=(0, 1)))], c="r", linestyle="--")
            axes[1].set_xlabel("sample roll")
            axes[1].set_ylabel("Chi2 [abu]")
            axes[1].set_title("Chi2 with rolling")
            fig.suptitle(self.filename)

            shifts = rolls[np.argmin(laser_rolls, axis=-1)]
            fig = plt.figure()
            gs = GridSpec(3, 2, height_ratios=[0.05, 1, 0.2], width_ratios=[1, 0.2], hspace=0, wspace=0)
            ax = plt.subplot(gs[1, 0])
            im = ax.imshow(shifts, aspect="auto", origin="bottom")
            ax.set_ylabel("kid index")
            ax.set_xticklabels([])
            cbax = plt.subplot(gs[0, 0])
            fig.colorbar(im, cax=cbax, orientation="horizontal", ticklocation="top")
            axh = plt.subplot(gs[2, 0])
            axh.plot(np.median(shifts, axis=0))
            axh.set_xlim(0, shifts.shape[1])
            axh.set_xlabel("time index")
            axv = plt.subplot(gs[1, 1])
            axv.plot(np.median(shifts, axis=1), np.arange(shifts.shape[0]))
            axv.yaxis.tick_right()
            axv.set_ylim(0, shifts.shape[0])
            fig.suptitle(self.filename)

        # return rolls[np.argmin(laser_rolls.mean(axis=1), axis=1)]

        return lasershifts

    @lru_cache(maxsize=1)
    def opds(self, plot=False, mode="common"):
        """Retrieve the optical path differences for each detector.

        Parameters
        ----------
        laser_shift : float
            The shift that must be applied to the laser wrt detector timestream
        mode : str ('common' | 'per_det' | 'per_int')
            See notes
        plot : bool
            plot intermediate results

        Returns
        -------
        opds : array_like (ndet, nint, nptint)
            the optical path difference for all the detectors
        zpds : array_like
            the zero path difference positions

        Notes
        -----
        Three modes are possible :
            * 'common' : one single zpd for all the detectors
            * 'per_det' : one zpd per detector
            * 'per_int' : one zpd per detector and interferogram

        The 'per_int' mode is too noisy and do not work

        Notes
        -----
        This depends on the `laser_shift` property.
        """
        # TODO: Check output dtype.....

        # Raw interferograms, without pipeline to get residuals peaks
        interferograms = self.interferograms

        # Global Shift MUST be applied
        laser = self.laser
        laser_directions = self.laser_directions

        laser_forward_mask = laser_directions == LaserDirection.FORWARD.value
        laser_backward_mask = laser_directions == LaserDirection.BACKWARD.value

        import dask.array as da

        dainterferograms = da.from_array(interferograms, name=False)
        interferograms_forward = dainterferograms[..., laser_forward_mask]
        interferograms_backward = dainterferograms[..., laser_backward_mask]

        max_forward = interferograms_forward == da.max(interferograms_forward, axis=2, keepdims=1)
        max_backward = interferograms_backward == da.max(interferograms_backward, axis=2, keepdims=1)
        max_forward = da.ma.filled(max_forward, False).compute()
        max_backward = da.ma.filled(max_backward, False).compute()

        # bad_max = (max_forward.sum(axis=2) > 1) | (max_backward.sum(axis=2) > 1)
        # max_forward[bad_max, :] = False
        # max_backward[bad_max, :] = False

        # Find the maximum of interferograms in forward and backward interferograms (Note : this is a duplicate from find_delay)
        # interferograms_forward = interferograms[..., laser_forward]
        # interferograms_backward = interferograms[..., laser_backward]
        # max_forward = interferograms_forward == np.max(interferograms_forward, axis=2, keepdims=1)
        # max_backward = interferograms_backward == np.max(interferograms_backward, axis=2, keepdims=1)

        # Sometimes interferograms have several maxima, impossible to disantangle, need to flag them here
        # bad_max = (max_forward.sum(axis=2) > 1) | (max_backward.sum(axis=2) > 1)
        # max_forward.mask[bad_max, :] = True
        # max_backward.mask[bad_max, :] = True

        # Retrieve the max per interferogram to set the 0
        with Pool(initializer=_pool_initializer, initargs=(laser, laser_forward_mask, max_forward)) as pool:
            max_laser_forward = np.asarray(pool.map(_pool_opd, np.arange(max_forward.shape[0])))

        with Pool(initializer=_pool_initializer, initargs=(laser, laser_backward_mask, max_backward)) as pool:
            max_laser_backward = np.asarray(pool.map(_pool_opd, np.arange(max_backward.shape[0])))

        zpds = np.array([max_laser_backward, max_laser_forward]).mean(axis=0)

        if plot:
            fig, axes = plt.subplots(ncols=2, sharex=True)
            kwargs = {"range": (82, 88), "bins": 200, "density": True}
            axes[0].hist(max_laser_forward.flatten(), label="forward", alpha=0.3, **kwargs)
            axes[0].hist(max_laser_backward.flatten(), label="backward", alpha=0.3, **kwargs)
            axes[0].hist(
                np.concatenate([max_laser_forward, max_laser_backward]).flatten(), label="full", alpha=0.7, **kwargs
            )
            axes[0].axvline(np.median(zpds), c="r")
            axes[0].set_title("overall max. laser position")
            axes[0].legend()
            if mode != "common":
                axes[1].hist(zpds.flatten(), **kwargs)
            axes[1].axvline(np.median(zpds), c="r")
            axes[1].set_title("per det. max. laser position")

        if mode == "common":
            # overall median
            zpds = np.nanmedian(zpds)
            opds = np.broadcast_to(laser - zpds, interferograms.shape)
        elif mode == "per_det":
            # per detector median
            zpds = np.nanmedian(zpds, axis=1)
            opds = np.broadcast_to(laser, interferograms.shape) - zpds[:, None, None]
        elif mode == "per_int":
            opds = np.broadcast_to(laser, interferograms.shape) - zpds[:, :, None]

        # Optical path differences are actually twice the laser position difference
        opds = 2 * opds

        return opds, zpds

    @lru_cache(maxsize=3)
    def interferograms_pipeline(self, ikid=None, flatfield="amplitude", cm_func=cm.pca_filtering, **kwargs):
        """Return the interferograms processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use (default: all)
        flatfield: str (None|'amplitude'|'interferograms'|'specFF')
            the flatfield applied to the data prior to common mode removal (default: amplitude)
        cm_function : function
            Default: common_mode.pca_filtering

        Returns
        -------
        interferograms : ndarray (ndet, nint, nptint)
            the masked interferograms with common mode removed

        Notes
        -----
        Any other args and kwargs are given to the pipeline function.
        ikid *must* be a tuple when calling the function, for lru_cache to work

        The flatfield values are taken from the amplitude column of the kidpar

        Masked value are set to 0 prior to common mode removal
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # KIDs selection
        interferograms = self.interferograms[ikid]

        # FlatField normalization
        if flatfield is None:
            flatfield = np.ones(interferograms.shape[0])
        elif flatfield in ["amplitude", "interferogram", "specFF"] and flatfield in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar[flatfield].data
        else:
            raise ValueError("Can not use this flat field : {}".format(flatfield))

        if isinstance(flatfield, MaskedColumn):
            flatfield = flatfield.filled(np.nan)

        logging.debug("Flatfielding")
        interferograms *= flatfield[:, np.newaxis, np.newaxis]
        shape = interferograms.shape

        logging.debug("Common mode removal")
        if cm_func is not None:
            output = cm_func(interferograms.reshape(shape[0], -1).filled(0), **kwargs).reshape(shape)
            # Put back the original mask
            logging.debug("Masking back")
            output = np.ma.array(output, mask=interferograms.mask)
        else:
            output = interferograms

        return output

    def _project_xyz(
        self,
        ikid=None,
        opd_mode="common",
        wcs=None,
        coord="diff",
        ctype3="opd",
        cdelt=(0.1, 0.2),
        cunit=("deg", "mm"),
        **kwargs,
    ):
        """Compute wcs and project the telescope position and optical path differences.

        Parameters
        ----------
        ikid : array, optional
            the selected kids index to consider (default: all)
        wcs : ~astropy.wcs.WCS, optional
            the projection wcs if provided, by default None
        coord : str, optional
            coordinate type, by default "diff"
        cdelt : tuple of 2 float
            the size of the pixels in degree and delta opd width
        cunit : tupe of 2 str
            the units of the above cdelt

        Returns
        -------
        ~astropy.wcs.WCS, array, array, array, tuple
            the projection wcs, projected coordinates x, y, z and shape of the resulting cube
        """
        az_coord = "F_{}_Az".format(coord)
        el_coord = "F_{}_El".format(coord)

        self._KissRawData__check_attributes([az_coord, el_coord])

        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        mask_tel = self.mask_tel

        # Retrieve data
        az = getattr(self, az_coord)[mask_tel]
        el = getattr(self, el_coord)[mask_tel]
        opds = self.opds(mode=opd_mode)[0][ikid]

        _kidpar = self.kidpar.loc[self.list_detector[ikid]]

        # Need to include the extreme kidspar offsets
        kidspar_margin_x = (_kidpar["x0"].max() - _kidpar["x0"].min()) / cdelt[0]
        kidspar_margin_y = (_kidpar["y0"].max() - _kidpar["y0"].min()) / cdelt[0]

        if wcs is None:
            # Project only the telescope position
            wcs, _, _ = build_celestial_wcs(
                az, el, crval=(0, 0), ctype=("OLON-SFL", "OLAT-SFL"), cdelt=cdelt[0], cunit=cunit[0],
            )

            # Add marging from the kidpar offsets
            wcs.wcs.crpix[0:2] += (kidspar_margin_x / 2, kidspar_margin_y / 2)

        if wcs.is_celestial:
            # extend the wcs for a third axis :
            if ctype3.lower() == "opd":
                wcs, z = extend_wcs(wcs, opds.flatten(), crval=0, ctype=ctype3, cdelt=cdelt[1], cunit=cunit[1])
                # Round the crpix3
                crpix3 = wcs.wcs.crpix[2]
                crpix3_offset = np.round(crpix3) - crpix3
                wcs.wcs.crpix[2] = np.round(crpix3)
                z = z + crpix3_offset
            else:
                wcs, z = extend_wcs(wcs, opds.flatten(), ctype=ctype3, cdelt=cdelt[1], cunit=cunit[1])
        else:
            # Full WCS given....
            z = wcs.sub([3]).all_world2pix(opds.flatten(), 0)[0]

        z = z.reshape(opds.shape)

        az_all = (az[:, np.newaxis] + _kidpar["x0"]).T
        el_all = (el[:, np.newaxis] + _kidpar["y0"]).T

        # Recompute the full projected coordinates
        x, y = wcs.celestial.all_world2pix(az_all, el_all, 0)

        shape = (
            np.round(z.max()).astype(np.int) + 1,
            np.round(y.max()).astype(np.int) + 1,
            np.round(x.max()).astype(np.int) + 1,
        )

        return wcs, x, y, z, shape

    def _modulation_std(self, ikid=None, flatfield="amplitude"):
        """Compute the median standard deviation of kids within the modulation.

        Parameters
        ----------
        ikid : tuple (optional)
            The list of kid index in self.list_detector to use (default: all)
        flatfield: str (None|'amplitude'|'interferograms'|'specFF')
            the flatfield applied to the data prior to common mode removal (default: amplitude)

        Returns
        -------
        weights ; array_like
            the corresponding weights
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # KIDs selection
        interferograms = self.interferograms_pipeline(ikid=tuple(ikid), flatfield=flatfield, cm_func=None)

        if isinstance(interferograms, np.ma.MaskedArray):
            interferograms = interferograms.data

        A_masq = self.A_masq

        A_high = A_masq_to_flag(A_masq, ModulationValue.high)
        A_low = A_masq_to_flag(A_masq, ModulationValue.low)

        mad_std = []
        for itgs, low, high in zip(interferograms.swapaxes(0, 1), A_low, A_high):
            itgs[:, low] -= np.median(itgs[:, low])
            itgs[:, high] -= np.median(itgs[:, high])

            mad_std.append(np.median(np.abs(itgs[:, high | low]), axis=1))

        return np.asarray(mad_std).T

    def interferograms_cube(
        self, ikid=None, wcs=None, shape=None, coord="diff", weights=None, opd_mode="common", opd_trim=None, **kwargs
    ):
        """Project the interferograms into one 3D cube.

        Parameters
        ----------
        ikid : tuple (optional)
            The list of kid index in self.list_detector to use (default: all)
        wcs : ~astropy.wcs.WCS (optional)
            The 3D wcs to be used to project the data
        shape : tuple (optional)
            The output shape to be used for the projected cube
        coord : str
            The coordinates type to be used (default: 'diff')
        weights : str (None|'std'|'continuum_std'|'modulation_std')
            The weights computation mode
        opd_mode : str ('common' | 'per_det' | 'per_int')
            See `KissSpectroscopy.opds`
        opd_trim : float
            Fraction of spaxels in the 3d dimension to be kept

        Returns
        -------
        output : FTSData
            cube of projected interferograms

        Notes
        -----
        Any keyword arguments from `KissSpectroscopy._project_xyz` or `KissSpectroscopy.interferogram_pipeline` can be used
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        mask_tel = self.mask_tel

        # opds and #interferograms should be part of the object
        self.__log.debug("Interferograms pipeline")
        sample = self.interferograms_pipeline(tuple(ikid), **kwargs)[:, mask_tel, :]

        ## We do not have the telescope position at 4kHz, but we NEED it !
        # TODO: Shall we make interpolation or leave it like that ? This would require changes in _pool_project_xyz
        logging.debug("Computing projected quantities")
        wcs, x, y, z, _shape = self._project_xyz(
            ikid=ikid, opd_mode=opd_mode, wcs=wcs, shape=shape, coord=coord, ctype3="opd", **kwargs
        )

        if shape is None:
            shape = _shape

        self.__log.debug("Computing weights")
        if weights is None:
            sample_weights = np.ones(sample.shape[0])
        elif weights == "std":
            logging.warnings("Using std weights in spectroscopy is probably a bad idea")
            with np.errstate(divide="ignore"):
                # Compute the weight per kid as the std of the median per interferogram
                # Probably NOT a good idea !!!!
                sample_weights = 1 / sample.std(axis=(1, 2)) ** 2
        elif weights == "continuum_std":
            with np.errstate(divide="ignore"):
                sample_weights = 1 / self.continuum_pipeline(tuple(ikid), **kwargs).std(axis=1) ** 2
        elif weights == "modulation_std":
            with np.errstate(divide="ignore"):
                sample_weights = 1 / self._modulation_std(ikid=ikid, flatfield=kwargs.get("flatfield", None)) ** 2
        else:
            raise ValueError("Unknown weights : {}".format(weights))

        bad_weight = np.isnan(sample_weights) | np.isinf(sample_weights)
        if np.any(bad_weight):
            sample_weights[bad_weight] = 0

        if isinstance(sample, np.ma.MaskedArray):
            sample = sample.filled(0)

        if isinstance(sample_weights, np.ma.MaskedArray):
            sample_weights = sample_weights.filled(0)

        histdd_kwargs = {
            "bins": shape,
            "range": ((-0.5, shape[0] - 0.5), (-0.5, shape[1] - 0.5), (-0.5, shape[2] - 0.5)),
        }

        # One single histogram : HUGE memory usage for x/y/z
        # all_x = np.repeat(x, nptint).reshape(ndet, nint, nptint) # This seems to be views
        # all_y = np.repeat(y, nptint).reshape(ndet, nint, nptint)
        # all_z = np.broadcast_to(z, (ndet, nint, nptint))
        # sample = [all_z.flatten(), all_y.flatten(), all_x.flatten()] # This here seems to allocate memory
        # hits, _ = np.histogramdd(sample, **kwargs)
        # data, _ = np.histogramdd(sample, **kwargs, weights=(cleaned_interferograms_test * interferogram_weights[:, None, None]).flatten())
        # weight, _ = np.histogramdd(sample, **kwargs, weights=np.repeat(interferogram_weights, nint*nptint).flatten())

        # Alternatively, construct the sample list by rounding and downcasting x, y, z
        # By construction x, y, z > -0.5, and can be rounded to int downcasting to proper uint
        # assert (x.min() > -0.5) & (y.min() > -0.5 ) & (z.min() > -0.5), "Some data are projected outside of the cube"
        # uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        # uint_max = [np.iinfo(uint_type).max for uint_type in uint_types]
        # # We need the same data type for all 3 for histogramdd...
        # data_max = np.max([np.round(data).max() for data in [z, y, x]])
        # rounded_dtype = uint_types[np.argwhere(data_max < uint_max)[0][0]]

        # Global histogramdd, need to blow up x/y -> important memory consumption, monoprocess
        # 3min 56s ± 281 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # hist_sample = np.array([np.round(z).astype(rounded_dtype).flatten(),
        #                         np.repeat(np.round(y).astype(rounded_dtype), self.nptint).flatten(),
        #                         np.repeat(np.round(x).astype(rounded_dtype), self.nptint).flatten(), ]).T
        # # del(x, y, z)
        # hits, _ = np.histogramdd(hist_sample, **histdd_kwargs)
        # data, _ = np.histogramdd(hist_sample, **histdd_kwargs, weights=(sample * sample_weights[:, None, None]).flatten())
        # weight, _ = np.histogramdd(hist_sample, **histdd_kwargs, weights=np.repeat(sample_weights, self.nint * self.nptint).flatten())

        # Attempt to parallelize the histogram
        # Failed at passing a second argument which is not the same shape as the first one
        # import dask.array as da
        # da_hist_sample = da.from_array(np.asarray([np.round(z).astype(rounded_dtype).flatten(),
        #                                np.repeat(np.round(y).astype(rounded_dtype), self.nptint).flatten(),
        #                                np.repeat(np.round(x).astype(rounded_dtype), self.nptint).flatten(), ]).T)
        # dask_histogramdd = lambda da, *args, **kwargs: np.histogramdd(da, **kwargs)[0] if args is () else np.histogram(da, weights=args[0], **kwargs)
        # res = da.map_blocks(dask_histogramdd, da_hist_sample, dtype=np.int, chunks=shape, **histdd_kwargs)
        # hits = res.reshape(shape[0], res.numblocks[1], shape[1], shape[2]).sum(axis=1)
        # dask_histogramdd_weights = lambda da, **kwargs: np.histogramdd(da[0:3], weights=da[3], **kwargs)[0]
        # da_weights = (da.from_array(sample.reshape(sample.shape[0], -1)) * sample_weights[:, None]).flatten().rechunk(da_hist_sample.chunksize[0])
        # res = da.map_blocks(dask_histogramdd_weights, da_hist_sample, da_weights, dtype=np.float32, chunks=shape, **histdd_kwargs)

        # Do it detector by detector, combine at the end... Can be parallelized
        # Lower memory footprint, longer execution, Need 2 * np.product(shape)
        # 3min 32s ± 1.28 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # global _pool_global
        # _pool_global = sample, sample_weights, x, y, z
        # from astropy.utils.console import ProgressBar
        # hits, data, weight = _pool_project_xyz(0, **histdd_kwargs)
        # for idet in ProgressBar(range(1, sample.shape[0])):
        #     _hits, _data, _weight = _pool_project_xyz(idet, **histdd_kwargs)
        #     hits += _hits
        #     data += _data
        #     weight += _weight

        # Higher memory footprint, but much much faster Need n_cpu * np.product(shape)
        self.__log.debug("Computing histograms")

        _this = partial(_pool_project_xyz, **histdd_kwargs)
        with Pool(cpu_count(), initializer=_pool_initializer, initargs=(sample, sample_weights, x, y, z)) as pool:
            results = pool.map(_this, np.array_split(range(sample.shape[0]), cpu_count()))
        hits = [result[0] for result in results if result is not None]
        data = [result[1] for result in results if result is not None]
        weight = [result[2] for result in results if result is not None]
        del results

        # Here we have results per kid data is actuall data*weight
        hits = np.asarray(hits).sum(axis=0)
        data = np.asarray(data).sum(axis=0)
        weight = np.asarray(weight).sum(axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data /= weight

        # Add standard keyword to header
        meta = {}
        meta["OBJECT"] = self.source
        meta["OBS-ID"] = self.scan
        meta["FILENAME"] = str(self.filename)
        meta["EXPTIME"] = self.exptime.value
        meta["DATE"] = datetime.datetime.now().isoformat()
        meta["DATE-OBS"] = self.obstime[0].isot
        meta["DATE-END"] = self.obstime[-1].isot
        meta["INSTRUME"] = self.param_c["nomexp"]
        meta["AUTHOR"] = "KidsData"
        meta["ORIGIN"] = "LAM"

        # Add extra keyword
        meta["SCAN"] = self.scan

        # TODO: CUT hits/data/weight and wcs here
        if opd_trim is not None and isinstance(opd_trim, (int, np.int, float, np.float)):
            mostly_good_opd = (~np.isnan(data)).sum(axis=(1, 2))
            mostly_good_opd = np.abs(mostly_good_opd / np.median(mostly_good_opd) - 1) < opd_trim
            _slice = slice(*np.nonzero(mostly_good_opd)[0][[0, -1]])
            hits = hits[_slice]
            data = data[_slice]
            weight = weight[_slice]
            wcs.wcs.crpix[2] -= _slice.start

        output = FTSData(data, wcs=wcs, meta=meta, mask=np.isnan(data), uncertainty=InverseVariance(weight), hits=hits)

        return output
