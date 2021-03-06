import os
import logging
import warnings
import numpy as np
import datetime
import re
from enum import Enum
from copy import deepcopy

import scipy.signal as signal
from scipy import interpolate

from functools import lru_cache, partial

import matplotlib.pyplot as plt

from scipy.special import erfcinv
from scipy.ndimage.morphology import binary_dilation, binary_opening
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, fftconvolve, savgol_filter
from multiprocessing import Pool
from autologging import logged

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
from astropy.nddata import NDDataArray, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.nddata.ccddata import _known_uncertainties, _unc_name_to_cls, _unc_cls_to_name
from astropy.stats import mad_std
from astropy.utils.console import ProgressBar

from .kiss_data import KissRawData
from .utils import cpu_count
from .utils import roll_fft, build_celestial_wcs, extend_wcs
from .utils import _import_from
from .utils import interferograms_regrid, project_3d
from .utils import multipolyfit, multipolyval

from .utils import psd_cal
from . import kids_plots

from .kids_calib import ModulationValue, mod_mask_to_flag
from .ftsdata import FTSData


class LaserDirection(Enum):
    FORWARD = 1
    BACKWARD = 2


N_CPU = cpu_count()

# Helper functions to pass large arrays in multiprocessing.Pool
_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _remove_polynomial(ikids, idx=None, deg=None):
    """Worker function to remove polynomial over interferograms

    data is passed as a global parameter. this function works on data[ikids] only

    Parameters
    ----------
    idx : ndarray or tuple of numpy.ndarray (nint, nptint)
        the indexes on which we need to compute the polynomials
    deg : int of tuple of int
        the degree of polynomial degree

    Returns
    -------
    output : ndarray (ikids.shape[0], nint, nptint)
        the baseline removed array of a portion of the input data
    coeff : ndarray (ikids.shape[0], nint, sum(deg))
        the coefficients of the baselines

    Notes
    -----
    the coefficient needs to be unpacked in the multi polynomial case.
    """

    global _pool_global
    (data,) = _pool_global

    # Output type:
    otype = data.dtype.type

    if len(ikids) == 0:
        return None

    idx = np.array(idx)

    # If using mulitple indexes, swap the first two axes for the loop on nint
    if idx.ndim == 3:
        idx = idx.swapaxes(0, 1)

    multi = isinstance(deg, (tuple, list, np.ndarray))

    _data = data[ikids]

    baselines = []
    coeff_s = []
    # loop on nint, parallel for all kids
    for _idx, _input in zip(idx, _data.swapaxes(0, 1)):
        # Rough (0/1) mask to avoid fitting for the modulation 0s
        _w = ~np.mean(_input.mask, axis=(0,)).astype(np.bool)

        if multi:
            coeff = multipolyfit(_idx, _input.filled(0).T, deg=deg, w=_w)
            baselines.append(otype(multipolyval(_idx, coeff)))
            coeff = np.vstack(coeff)  # for the returned values... needs to be unpacked if used
        else:
            coeff = np.polynomial.polynomial.polyfit(_idx, _input.filled(0).T, deg=deg, w=_w)
            baselines.append(otype(np.polynomial.polynomial.polyval(_idx, coeff)))
        coeff_s.append(coeff.T)

    _data -= np.array(baselines).swapaxes(0, 1)
    coeff_s = np.array(coeff_s).swapaxes(0, 1)

    return _data, coeff_s


def remove_polynomial(data, idx, deg):
    """Remove polynomial over interferograms in paralllel.

    parallelization is made on kids index

    Parameters
    ----------
    data : numpy.ndarray (ndet, nint, nptint)
        input array
    idx : ndarray or tuple of numpy.ndarray (nint, nptint)
        the indexes on which we need to compute the polynomials
    deg : int of tuple of int
        the degree of polynomial degree

    Returns
    -------
    output : ndarray (ndet, nint, nptint)
        the baseline removed array
    coeff : ndarray (ikids.shape[0], nint, sum(deg))
        the coefficients of the baselines

    Notes
    -----
    the coefficient needs to be unpacked in the multi polynomial case.

    """
    # disable mkl parallelization, more efficient on kids for large number of kids
    import mkl

    mkl_threads = mkl.set_num_threads(2)

    _this = partial(_remove_polynomial, idx=idx, deg=deg)
    with Pool(
        N_CPU,
        initializer=_pool_initializer,
        initargs=(data,),
    ) as pool:
        outputs = pool.map(_this, np.array_split(range(data.shape[0]), N_CPU))

    mkl.set_num_threads(mkl_threads)

    # global _pool_global
    # _pool_global = (data,)
    # outputs = list(ProgressBar.map(_this, np.array_split(range(data.shape[0]), N_CPU)))

    return np.ma.vstack([output[0] for output in outputs if output is not None]), np.vstack(
        [output[1] for output in outputs if output is not None]
    )


def _pool_find_lasershifts_brute_worker(roll, _roll_func="numpy.roll"):
    """Worker function to compute 3D histogram for a given detector list"""

    global _pool_global
    interferograms, lasers, laser_mask_forward, laser_mask_backward = _pool_global

    _roll_func = _import_from(_roll_func)

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


def _pool_interferograms_regrid(i_kid, bins=None):
    """Regrid interferograms to a common grid."""

    global _pool_global

    interferograms, laser = _pool_global

    return interferograms_regrid(interferograms[i_kid], laser, bins=bins)[0]


def _sky_to_cube(ikids):

    global _pool_global
    data, opds, az, el, offsets, wcs, shape = _pool_global

    # TODO: Shall we provide kid_weights within a kids ?

    # TODO: az & el are per bloc here... need to interpolate.....

    outputs = []
    kid_weights = []
    hits = []

    for ikid in ikids:
        # TODO: az & el are per bloc here... need to interpolate.....
        # for now repeat....
        nptint = opds.shape[2]
        x, y, z = wcs.all_world2pix(
            np.repeat(az + offsets[ikid]["x0"], nptint),
            np.repeat(el + offsets[ikid]["y0"], nptint),
            opds[ikid].flatten(),
            0,
        )
        output, weight, hit = project_3d(x, y, z, data[ikid].flatten(), shape)

        outputs.append(output)
        kid_weights.append(weight)
        hits.append(hit)

    return np.array(outputs), np.array(kid_weights), np.array(hits)


def sky_to_cube(data, opds, az, el, offsets, wcs, shape):

    with Pool(
        cpu_count(),
        initializer=_pool_initializer,
        initargs=(data, opds, az, el, offsets, wcs, shape),
    ) as pool:
        items = pool.map(_sky_to_cube, np.array_split(np.arange(data.shape[0]), cpu_count()))

    outputs = np.vstack([item[0] for item in items if len(item[0]) != 0])
    kid_weights = np.vstack([item[1] for item in items if len(item[1]) != 0])
    hits = np.vstack([item[2] for item in items if len(item[2]) != 0])

    return outputs, kid_weights, hits


@logged
class KissSpectroscopy(KissRawData):
    """This Class deals with spectroscopic data in KISS.

    Attributes
    ----------
    laser_keys : str
        list of keys to be used to derive laser position, default "auto"
    laser_shift : float, optionnal
        the number or sample to shift the laser position wrt to interferograms
    optical_flip : str
        regular expression to select kids to sign flip, by default auto
    interferograms : numpy.ma.MaskedArray (ndet, nint, nptint cached)
        the interferograms with optical flip and glitches removed if needed.
    laser : array_like (nint, nptint, cached)
        Retrieve the laser position with missing value interpolated.
    laser_directions : array_like (nptint, cached)
        the laser directions as listed in `LaserDirection`

    Methods
    -------
    find_lasershifts_brute(**kwargs)
        find potential shift between mirror position and interferograms timeline
    interferograms_pipeline(**kwargs)
        return the interferograms processed by given pipeline
    interferogram_cube(**kwargs)
        project the interferograms into one 3D cube
    interferogram_beamcubes(**kwargs)
        project the interferograms into individual 3D cubes
    """

    __laser_shift = None
    __laser_keys = None
    __laser_reduction = None

    __optical_flip = None
    __mask_glitches = None
    __glitches_threshold = None

    def __init__(
        self,
        *args,
        laser_keys="auto",
        laser_reduction="numpy.mean",
        laser_shift=None,
        optical_flip="auto",
        mask_glitches=True,
        glitches_threshold=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        laser_keys : str
            list of keys to be used to derive laser position, default "auto"
        laser_reduction : str
            function to use to reduce the (potential) many laser position to one, by default "numpy.mean"
        laser_shift : float, optionnal
            the number or sample to shift the laser position wrt to interferograms
        optical_flip : str
            regular expression to select kids to sign flip, by default auto
        mask_glitches : bool
            to flag the glitches in the interferograms, default True
        glitches_threshold : float
            the sigma threshold to flag the glitches

        Notes
        -----
        optical_flip auto mode will determine if data is from KISS or CONCERTO by looking at the provided self._kidpar namedet column
        """
        super().__init__(*args, **kwargs)

        self.__log.debug("KissSpectroscopy specific kwargs")
        self.laser_keys = laser_keys
        self.laser_reduction = laser_reduction
        self.laser_shift = laser_shift
        self.optical_flip = optical_flip
        self.__mask_glitches = mask_glitches
        self.__glitches_threshold = glitches_threshold

    # TODO: This could probably be done more elegantly
    @property
    def laser_shift(self):
        return self.__laser_shift

    @laser_shift.setter
    def laser_shift(self, value):
        if self.__laser_shift is None:
            self.__laser_shift = value
        else:
            self.__laser_shift += value
        KissSpectroscopy.laser.fget.cache_clear()
        KissSpectroscopy.laser_directions.fget.cache_clear()
        KissSpectroscopy.opds.cache_clear()

    @property
    def optical_flip(self):
        return self.__optical_flip

    @optical_flip.setter
    def optical_flip(self, value):
        KissSpectroscopy.opds.cache_clear()

        if value == "auto":
            boxes = {name[0:2] for name in self._kidpar["namedet"] if name[0] == "K"}
            if boxes == {"KA", "KB"}:
                self.__log.info("Found Kiss data, flipping KB")
                self.__optical_flip = "KB"
            elif boxes == {"KA", "KB", "KC", "KD", "KE", "KF", "KG", "KH", "KI", "KJ", "KK", "KL"}:
                self.__log.info("Found CONCERTO data, flipping KG - KL")
                self.__optical_flip = "K(G|H|I|J|K|L)"
            else:
                self.__log.error("Can  not determine the instrument for optical flip : disabled")
                self.__optical_flip = None
        elif isinstance(value, str) or value is None:
            self.__optical_flip = value
        else:
            raise ValueError("value must be a regex string, 'auto' or None")

    @property
    def laser_keys(self):
        return self.__laser_keys

    @laser_keys.setter
    def laser_keys(self, value):
        if value == "auto":
            # Look for laser position keys
            keys = self.names.DataSc + self.names.DataSd + self.names.DataUc + self.names.DataUd
            laser_keys = [key for key in keys if "laser" in key and key.endswith("pos")]
            if not laser_keys:
                self.__log.error("Could not find laser position keys")
            self.__laser_keys = laser_keys
        else:
            if not isinstance(value, list):
                value = list(value)
            self.__laser_keys = value
        KissSpectroscopy.laser.fget.cache_clear()

    @property
    def laser_reduction(self):
        return self.__laser_reduction

    @laser_reduction.setter
    def laser_reduction(self, value):
        self.__laser_reduction = value
        KissSpectroscopy.laser.fget.cache_clear()

    @property
    def mask_glitches(self):
        return self.__mask_glitches

    @mask_glitches.setter
    def mask_glitches(self, value):
        self.__mask_glitches = value
        KissSpectroscopy.opds.cache_clear()

    @property
    def glitches_threshold(self):
        return self.__glitches_threshold

    @glitches_threshold.setter
    def glitches_threshold(self, value):
        self.__glitches_threshold = value
        KissSpectroscopy.opds.cache_clear()

    @property
    def meta(self):
        meta = super().meta

        # Specific cases
        meta["LASER_SHIFT"] = self.laser_shift or 0

        return meta

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
        This depends on the `laser_shift` and `laser_reduction` property.
        """
        laser_keys = self.laser_keys
        self._KissRawData__check_attributes(laser_keys)

        self.__log.info(
            "Computing {} laser position from {} with {} shift".format(
                self.laser_reduction, laser_keys, self.laser_shift
            )
        )

        laser = [getattr(self, key).flatten() for key in laser_keys]

        # Check laser consistancy:
        # TODO: should be made elsewhere...
        if len(laser) > 1 and self.laser_reduction == "numpy.mean":
            # Differences between different laser measurements
            diff_laser = np.diff(laser, axis=0)
            std_diff_laser = np.std(diff_laser)
            if std_diff_laser > 1:  # More than 1mm variation in time between the two laser
                self.__log.error("Varying differences between {} : {}".format(laser_keys, std_diff_laser))

        # combine laser position(s)
        red_fonction = _import_from(self.laser_reduction)
        laser = red_fonction(laser, axis=0)

        # Check the
        # If 99.99% of the data is similar....
        same = np.mean(laser[::2] == laser[1::2])
        if (1 - same) < 1e-5:
            self.__log.debug("Interpolating mirror positions")
            if same != 1:
                self.__log.warning("{:f} % of mirror positions differs".format((1 - same) * 100))
            # Mirror positions are acquired at half he acquisition frequency thus...
            # we can quadratictly interpolate the second position...
            # laser[1::2] = interp1d(range(len(laser) // 2), laser[::2], kind="quadratic", fill_value="extrapolate")(
            #     np.arange(len(laser) // 2) + 0.5
            # )
            # Middle of the two points
            ref_index = np.arange(len(laser) // 2) * 2 + 0.5
            ref_laser = laser[::2]
            laser = interp1d(ref_index, ref_laser, kind="quadratic", fill_value="extrapolate")(np.arange(len(laser)))

        if self.laser_shift is not None:
            self.__log.debug("Shifting mirror positions by {} sample".format(self.laser_shift))
            ## TODO: different cases depending on the shape of laser_shift
            laser = roll_fft(laser, self.laser_shift)

        # Same shape as interferograms[0]
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

        mean_laser = self.laser.mean(axis=0)

        # Find the forward and backward phases
        # rough cut on the the mean mirror position : find the large peaks of the derivatives
        diff_laser = savgol_filter(mean_laser, 15, 2, deriv=1)
        turnovers, _ = find_peaks(-np.abs(diff_laser), prominence=1e-4, height=np.min(diff_laser) / 2)

        # Retains only the higest two
        turnovers = turnovers[np.argsort(_["prominences"])[-2:]]

        # Mainly for cosine simulations which actually start on a turnover
        if len(turnovers) == 1:
            turnovers = np.append([0], turnovers)

        assert len(turnovers) == 2

        turnovers.sort()

        laser_directions = np.zeros(self.nptint, dtype=np.int8)
        laser_directions[:] = (
            LaserDirection.BACKWARD.value
            if np.mean(diff_laser[slice(*turnovers)]) > 0
            else LaserDirection.FORWARD.value
        )
        laser_directions[slice(*turnovers)] = (
            LaserDirection.FORWARD.value
            if np.mean(diff_laser[slice(*turnovers)]) > 0
            else LaserDirection.BACKWARD.value
        )

        return laser_directions

    def find_lasershifts_brute(
        self, ikid=None, start=-10, stop=10, num=21, roll_func="numpy.roll", plot=False, mode="single", sos=None
    ):
        """Find potential shift between mirror position and interferograms timeline.

        Brute force approach by computing min Chi2 value between rolled forward and backward interferograms

        Parameters
        ----------
        ikid : tuple (optional)
            The list of kid index in self.list_detector to use (default: all)
        start, stop : float or int
            the minimum and maximum shift to consider
        num : int
            the number of rolls between those two values
        roll_func : str ('numpy.roll'|'kidsdata.utils.roll_fft')
            the rolling function to be used 'numpy.roll' for integer rolls and 'kidsdata.utils.roll_fft' for floating values
        plot : bool
            display so debugging plots
        mode : str (single|per_det|per_int|per_det_int)
            return value mode (see Notes), (default: single)
        sos : array_like, optionnal
            A second-order sections representation of an IIR filter (see `scipy.signal`), default None

        Returns
        -------
        lasershifts : array_like
            the lasershift which shape depends on the selected mode
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # Apply a high pass filter at 100 Hz
        # This DO NOT work for ralenti > 1
        # self.__log.info("Applying a high pass filter at 100 Hz")
        # sos = signal.butter(4, 100, "high", fs=self.param_c["acqfreq"], output="sos")

        interferograms = self.interferograms_pipeline(
            ikid=tuple(ikid), flatfield=None, baseline_opd=3, baseline_time=3, cm_func=None, sos=sos
        ).filled(0)

        lasers = self.laser
        laser_mask = self.laser_directions

        if roll_func == "numpy.roll":
            rolls = np.linspace(start, stop, num, dtype=int)
        elif roll_func == "kidsdata.utils.roll_fft":
            rolls = np.linspace(start, stop, num)
        else:
            raise ValueError("Unknown roll_func : {}".format(roll_func))

        self.__log.info("Brute force rolling of laser position from {} to {} ({})".format(start, stop, num))
        # laser_rolls = []
        # for roll in rolls:
        #     laser_rolls.append(find_shift__roll_chi2(interferograms, laser, laser_mask, roll))
        _this = partial(_pool_find_lasershifts_brute, _roll_func=roll_func)
        with Pool(
            N_CPU,
            initializer=_pool_initializer,
            initargs=(
                interferograms,
                lasers,
                laser_mask == LaserDirection.FORWARD.value,
                laser_mask == LaserDirection.BACKWARD.value,
            ),
        ) as pool:
            laser_rolls = pool.map(_this, np.array_split(rolls, N_CPU))

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
            im = ax.imshow(shifts, aspect="auto", origin="lower")
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
    def opds(self, ikid=None, opd_mode="per_det", laser_bins="sqrt", **kwargs):
        """Retrieve the optical path differences for each detector.

        Parameters
        ----------
         ikid : tuple
            the list of kid index in self.list_detector to use (default: all)
        opd_mode : str ('common' | 'per_det' )
            See notes
        laser_bins : int or sequence of scalars or str, optional
            the binning for the common laser position, by default 'sqrt'

        Returns
        -------
        opds : array_like (ndet, nint, nptint)
            the optical path difference for all the detectors
        zlds : array_like
            the zero laser differences positions

        Notes
        -----
        Two modes are possible :
            * 'common' : one single zld for all the detectors
            * 'per_det' : one zld per detector

        Notes
        -----
        This depends on the `laser_shift` property.
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # Raw interferograms, without pipeline to get residuals peaks
        # interferograms = self.interferograms[ikid]
        interferograms = self.interferograms_pipeline(ikid=tuple(ikid), cm_func=None, flatfield=None, baseline=3)

        # Global Shift MUST be applied before using this...
        laser = self.laser

        # Regrid all interferograms to the same laser grid
        _, bins = np.histogram(laser.flatten(), bins=laser_bins)
        c_bins = np.mean([bins[1:], bins[:-1]], axis=0)

        self.__log.debug("Regriding iterferograms")
        worker = partial(_pool_interferograms_regrid, bins=bins)
        with Pool(N_CPU, initializer=_pool_initializer, initargs=(interferograms.filled(0), laser)) as p:
            output = p.map(worker, range(len(ikid)))

        output = np.array(output).reshape(interferograms.shape[0], interferograms.shape[1], -1)

        # Take the mean per detector
        output_per_det = np.nanmean(output, axis=1)
        self.__log.debug("Computing Zero Laser Differences")
        zlds = c_bins[np.argmax(output_per_det, axis=1)]

        # spec_FF = np.max(output_per_det, axis=1)

        if opd_mode == "common":
            # overall median
            zlds = np.nanmedian(zlds)
            opds = np.broadcast_to(laser - zlds, interferograms.shape)
        elif opd_mode == "per_det":
            # per detector median
            opds = np.broadcast_to(laser, interferograms.shape) - zlds[:, None, None]

        # Optical path differences are actually twice the laser position difference
        opds = 2 * opds

        return opds, zlds

    def interferograms_pipeline(
        self,
        ikid=None,
        coord=None,
        flatfield="amplitude",
        baseline=None,
        baseline_opd=None,
        baseline_time=None,
        cm_func="kidsdata.common_mode.pca_filtering",
        sos=None,
        **kwargs,
    ):
        """Return the interferograms processed by given pipeline.

        Parameters
        ----------
        ikid : tuple
            the list of kid index in self.list_detector to use (default: all)
        coord : str, optional
            coordinate type to retrieve additionnal mask, by default None
        flatfield: str (None|'amplitude'|'interferograms'|'specFF')
            the flatfield applied to the data prior to common mode removal (default: amplitude)
        baseline : int, optionnal
            the polynomial degree (in opd space) of final baselines to be removed
        baseline_opd : int, optionnal
            the polynomial degree (in opd space) of baselines
        baseline_time : int, optionnal
            the polynomial degree (in time space) of baselines
        cm_func : str
            Function to use for the common mode removal, by default 'kidsdata.common_mode.pca_filtering'
        sos : array_like, optionnal
            A second-order sections representation of an IIR filter (see `scipy.signal`), default None
        **kwargs :
            Additionnal keyword argument passed to cm_func

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

        self.__log.debug("Copy interferograms")
        # KIDs selection : this copy the data
        interferograms = self.interferograms[ikid]

        # Add the mask from the positions
        # interferograms.mask |= self.mask_tel[None, :, None]
        if coord is not None:
            _, _, mask_tel = self.get_telescope_positions(coord=coord, undersampled=False)
            interferograms.mask[:, mask_tel] = True

        if self.optical_flip:
            self.__log.info("Flipping {}".format(self.optical_flip))
            p = re.compile(self.optical_flip)
            vmatch = np.vectorize(lambda det: bool(p.match(det)))
            to_flip = vmatch(self.list_detector[ikid])

            interferograms[to_flip] *= -1

        if self.mask_glitches:
            self.__log.info("Masking glitches")

            # Tests -- Rough Deglitching
            # Remove the mean value per interferogram index -> left with variations only
            # BUT this do not work as there are jumps
            # interferograms_norm = kd.interferograms - kd.interferograms.mean(axis=1)[:, None, :]
            # Use a gaussian filter with a width of 3 interferograms to get smooth variations along the interferograms indexes
            # from scipy.ndimage import gaussian_filter1d, gaussian_laplace
            # might flag real source
            # # interferograms  -= gaussian_filter1d(kd.interferograms, 3, axis=1)

            # Try something else on the modulation flagged interferograms
            abs_interferograms = np.abs(interferograms)
            max_abs_interferogram = abs_interferograms.max(axis=2)

            # Threshold for gaussian statistic, along the interferogram axis only :
            sigma = erfcinv(self.glitches_threshold / np.product(interferograms.shape[1])) * np.sqrt(2)

            cutoffs = max_abs_interferogram.mean(axis=1) + sigma * max_abs_interferogram.std(axis=1)
            glitches_mask = abs_interferograms > cutoffs[:, None, None]
            del abs_interferograms

            self.__log.warning("Masking {:3.1f}% of the data from glitches".format(np.mean(glitches_mask) * 100))

            interferograms.mask |= glitches_mask

        if sos is not None:
            self.__log.debug("Applying IIR filter")
            dummy = signal.sosfiltfilt(sos, interferograms.filled(0))
            interferograms = np.ma.array(dummy, mask=interferograms.mask)

        self.__log.debug("Applying flatfield : {}".format(flatfield))
        # FlatField normalization
        if flatfield in ["amplitude", "interferogram", "specFF"] and flatfield in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            flatfield = _kidpar[flatfield].data
        elif flatfield is not None:
            raise ValueError("Can not use this flat field : {}".format(flatfield))

        if isinstance(flatfield, MaskedColumn):
            flatfield = flatfield.filled(np.nan)

        if flatfield is not None:
            interferograms /= flatfield[:, np.newaxis, np.newaxis]

        shape = interferograms.shape

        if cm_func is not None:
            self.__log.debug("Common mode removal ; {}, {}".format(cm_func, kwargs))

            # ugly hack for now :
            if cm_func == "kidsdata.common_mode.common_itg" and "laser" not in kwargs:
                kwargs["laser"] = self.laser

            cm_func = _import_from(cm_func)
            # There is a copy here (with the .filled(0))
            output = cm_func(interferograms.reshape(shape[0], -1).filled(0), **kwargs).reshape(shape)
            # Put back the original mask
            interferograms = np.ma.array(output, mask=interferograms.mask)

        if baseline is not None:
            warnings.warn("baseline is deprecated, use baseline_opd and/or baseline_time instead", DeprecationWarning)
            if baseline_opd is None:
                baseline_opd = baseline
            if baseline_time is None:
                baseline_time = baseline

        if baseline_opd is not None and baseline_time is not None:
            idx = (np.tile(np.arange(self.nptint), self.nint).reshape(self.nint, self.nptint), self.laser)

            self.__log.info(
                "Polynomial baseline per block on laser position ({}) and time position ({})".format(
                    baseline_opd, baseline_time
                )
            )

            interferograms, _ = remove_polynomial(interferograms, idx, (baseline_time, baseline_opd))

        elif baseline_opd is not None:
            idx = self.laser
            self.__log.info("Polynomial baseline per block on laser position ({})".format(baseline_opd))

            interferograms, _ = remove_polynomial(interferograms, idx, baseline_opd)

        elif baseline_time is not None:
            idx = np.tile(np.arange(self.nptint), self.nint).reshape(self.nint, self.nptint)
            self.__log.info("Polynomial baseline per block on time position ({})".format(baseline_time))

            interferograms, _ = remove_polynomial(interferograms, idx, baseline_time)

        return interferograms

    def _build_3d_wcs(
        self, ikid=None, wcs=None, coord="diff", cdelt=(0.1, 0.1, 0.2), cunit=("deg", "deg", "mm"), **kwargs
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

        cdelt : tuple of 2 or 3 floats,
            either (spthe size of the pixels and delta opd width (see Notes) in ...
        cunit : tupe of 2 or 3 str
            ... the units of the above cdelt

        Returns
        -------
        ~astropy.wcs.WCS, tuple
            the projection wcs and shape of the resulting cube

        Notes
        -----
        When prodiving a tuple of length 2 for cdelt and cunit, the pixels are assumed spatially square,
        ie (dx, ds) is equivalent to (dx, dx, ds)
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        if len(cdelt) == 2:
            cdelt = (cdelt[0], cdelt[0], cdelt[1])
        if len(cunit) == 2:
            cunit = (cunit[0], cunit[0], cunit[1])

        # TODO: Undersampled for the moment... swith to fully sampled ??
        az, el, mask_tel = self.get_telescope_positions(coord)
        good_tel = ~mask_tel
        az, el = az[good_tel], el[good_tel]

        opds = self.opds(ikid=tuple(ikid), **kwargs)[0][:, good_tel, :]

        _kidpar = self.kidpar.loc[self.list_detector[ikid]]

        # Need to include the extreme kidspar offsets
        kidspar_margin_x = (_kidpar["x0"].max() - _kidpar["x0"].min()) / cdelt[0]
        kidspar_margin_y = (_kidpar["y0"].max() - _kidpar["y0"].min()) / cdelt[1]

        if wcs is None:
            # Project only the telescope position
            wcs, _, _ = build_celestial_wcs(
                az,
                el,
                crval=(0, 0),
                ctype=("OLON-SFL", "OLAT-SFL"),
                cdelt=cdelt[0:2],
                cunit=cunit[0:2],
            )

            # Add marging from the kidpar offsets
            wcs.wcs.crpix[0:2] += (kidspar_margin_x / 2, kidspar_margin_y / 2)

        if wcs.is_celestial:
            # extend the wcs for a third axis :
            wcs = extend_wcs(wcs, opds.flatten(), crval=0, ctype="OPD", cdelt=cdelt[2], cunit=cunit[2])
            # Round the crpix3
            crpix3 = wcs.wcs.crpix[2]
            wcs.wcs.crpix[2] = np.round(crpix3)

        # az_all = (az[:, np.newaxis] + _kidpar["x0"]).T
        # el_all = (el[:, np.newaxis] + _kidpar["y0"]).T
        # Or use the maxima, which will be too big
        az_all = (np.array([az.min(), az.max()])[:, np.newaxis] + _kidpar["x0"]).T
        el_all = (np.array([el.min(), el.max()])[:, np.newaxis] + _kidpar["y0"]).T

        # Recompute the full projected coordinates
        x, y = wcs.celestial.all_world2pix(az_all, el_all, 0)
        z = wcs.sub([3]).all_world2pix([opds.min(), opds.max()], 0)[0]

        shape = (
            np.round(z.max()).astype(np.int) + 1,
            np.round(y.max()).astype(np.int) + 1,
            np.round(x.max()).astype(np.int) + 1,
        )

        return wcs, shape

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
        kid_weights ; array_like
            the corresponding kid_weights
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))
        else:
            ikid = np.asarray(ikid)

        # KIDs selection
        interferograms = self.interferograms_pipeline(ikid=tuple(ikid), flatfield=flatfield, cm_func=None)

        if isinstance(interferograms, np.ma.MaskedArray):
            interferograms = interferograms.data

        mod_mask = self.mod_mask

        A_high = mod_mask_to_flag(mod_mask, ModulationValue.high)
        A_low = mod_mask_to_flag(mod_mask, ModulationValue.low)

        # Compute median standard deviation in the modulation points
        mad_stds = []
        for itgs, low, high in zip(interferograms.swapaxes(0, 1), A_low, A_high):
            itgs[:, low] -= np.median(itgs[:, low])
            itgs[:, high] -= np.median(itgs[:, high])

            mad_stds.append(np.median(np.abs(itgs[:, high | low]), axis=1))

        return np.asarray(mad_stds).T

    def interferogram_cube(
        self,
        ikid=None,
        wcs=None,
        shape=None,
        coord="diff",
        cdelt=(0.1, 0.1, 0.2),
        cunit=("deg", "deg", "mm"),
        kid_weights=None,
        opd_trim=None,
        **kwargs,
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
        cdelt : tuple of 2 or 3 float
            the size of the pixels and delta opd width in ...
        cunit : tupe of 2 or 3 str
            ... the units of the above cdelt
        kid_weights : str (None|'std'|'continuum_std'| 'continuum_mad' | 'modulation_std' | key)
            the inter kid weight to use (see Note)
        opd_trim : float
            Fraction of spaxels in the 3d dimension to be kept

        Returns
        -------
        output : FTSData
            cube of projected interferograms

        Notes
        -----
        Any keyword arguments from `KissSpectroscopy.opds` or `KissSpectroscopy.interferograms_pipeline` can be used

        The kid weights are used to combine different kids together :
        - None : do not apply weights
        - `std` : standard deviation of each timeline (actually 1 / std**2) (!! Bad idea in interferometry)
        - `continuum_std` : standard deviation of continuum timelines (actually 1 / std**2)
        - `continuum_mad` : median absolute deviation of ontinuum timelines (actually 1 / mad**2)
        - `modulation_std` : standard deviation during modulation time (actually 1 / std**2)
        - key : any key from the kidpar table
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        # TODO: Undersampled for the moment... swith to fully sampled !!
        az, el, mask_tel = self.get_telescope_positions(coord)

        # opds and #interferograms should be part of the object
        self.__log.info("Interferograms pipeline with {}".format(kwargs))
        data = self.interferograms_pipeline(tuple(ikid), coord=coord, **kwargs)

        self.__log.info("Computing OPDs")
        opds = self.opds(ikid=tuple(ikid), **kwargs)[0]

        ## We do not have the telescope position at 4kHz, but we NEED it !
        self.__log.info("Computing projected quantities")
        wcs, _shape = self._build_3d_wcs(ikid=ikid, wcs=wcs, coord=coord, cdelt=cdelt, cunit=cunit, **kwargs)

        if shape is None:
            shape = _shape

        self.__log.info("Projecting data")
        # At this stage we have maps per kids
        offsets = self.kidpar.loc[self.list_detector[ikid]]["x0", "y0"]
        outputs, weights, hits = sky_to_cube(data, opds, az, el, offsets, wcs, shape)

        # At this satge data, weights, hits are (ndet, nitg, nx, ny)

        self.__log.info("Computing kid weights")
        if kid_weights is None:
            kid_weights = np.ones(outputs.shape[0])
        elif kid_weights == "std":
            self.__log.warning("Using std kid_weights in spectroscopy is probably a bad idea")
            with np.errstate(divide="ignore"):
                # Compute the weight per kid as the std of the median per interferogram
                # Probably NOT a good idea !!!!
                kid_weights = 1 / outputs.std(axis=(1, 2)) ** 2
        elif kid_weights == "continuum_std":
            with np.errstate(divide="ignore"):
                kid_weights = 1 / self.continuum_pipeline(tuple(ikid), **kwargs).std(axis=1) ** 2
        elif kid_weights == "continuum_mad":
            with np.errstate(divide="ignore"):
                kid_weights = 1 / mad_std(self.continuum_pipeline(tuple(ikid), **kwargs), axis=1) ** 2
        elif kid_weights == "modulation_std":
            with np.errstate(divide="ignore"):
                kid_weights = 1 / self._modulation_std(ikid=ikid, flatfield=kwargs.get("flatfield", None)) ** 2
        elif kid_weights in self.kidpar.keys():
            _kidpar = self.kidpar.loc[self.list_detector[ikid]]
            kid_weights = _kidpar[kid_weights].data
        else:
            raise ValueError("Unknown kid weights : {}".format(kid_weights))

        bad_weight = np.isnan(kid_weights) | np.isinf(kid_weights)
        if np.any(bad_weight):
            kid_weights[bad_weight] = 0

        if isinstance(outputs, np.ma.MaskedArray):
            outputs = outputs.filled(0)

        if isinstance(kid_weights, np.ma.MaskedArray):
            kid_weights = kid_weights.filled(0)

        # Combine all kids, including inter kid weights
        weight = np.nansum(weights * kid_weights[:, None, None, None], axis=0)
        output = np.nansum(outputs * weights * kid_weights[:, None, None, None], axis=0) / weight
        hits = np.nansum(hits, axis=0)

        # Add standard keyword to header
        self._update_meta()
        meta = self.meta
        meta["N_KIDS"] = len(ikid)

        # TODO: CUT hits/data/weight and wcs here
        if opd_trim is not None and isinstance(opd_trim, (int, np.int, float, np.float)):
            mostly_good_opd = (~np.isnan(output)).sum(axis=(1, 2))
            mostly_good_opd = np.abs(mostly_good_opd / np.median(mostly_good_opd) - 1) < opd_trim
            _slice = slice(*np.nonzero(mostly_good_opd)[0][[0, -1]])
            hits = hits[_slice]
            output = output[_slice]
            weight = weight[_slice]
            wcs.wcs.crpix[2] -= _slice.start

        return FTSData(
            output, wcs=wcs, meta=meta, mask=np.isnan(output), uncertainty=InverseVariance(weight), hits=hits
        )

    def interferogram_beamcubes(
        self,
        ikid=None,
        wcs=None,
        shape=None,
        coord="diff",
        cdelt=(0.1, 0.1, 0.2),
        cunit=("deg", "deg", "mm"),
        opd_trim=None,
        **kwargs,
    ):
        """Project the interferograms into individual 3D cubes.

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
        cdelt : tuple of 2 or 3 float
            the size of the pixels and delta opd width in ...
        cunit : tupe of 2 or 3 str
            ... the units of the above cdelt

        opd_trim : float
            Fraction of spaxels in the 3d dimension to be kept

        Returns
        -------
        output : FTSData
            cube of projected interferograms

        Notes
        -----
        Any keyword arguments from `KissSpectroscopy.opds` or `KissSpectroscopy.interferograms_pipeline` can be used
        """
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        # expand cdelt and cunit if present
        if len(cdelt) == 2:
            cdelt = cdelt[0], cdelt[0], cdelt[1]
        if len(cunit) == 2:
            cunit = cunit[0], cunit[0], cunit[1]

        # TODO: Undersampled for the moment... swith to fully sampled !!
        az, el, mask_tel = self.get_telescope_positions(coord)

        # opds and #interferograms should be part of the object
        self.__log.info("Interferograms pipeline")
        kwargs["flatfield"] = None
        data = self.interferograms_pipeline(tuple(ikid), coord=coord, **kwargs)

        self.__log.info("Computing opds")
        opds = self.opds(ikid=tuple(ikid), **kwargs)[0]

        ## We do not have the telescope position at 4kHz, but we NEED it !
        self.__log.info("Computing WCS")
        if wcs is None:
            # Move cdelt back to celestial only
            wcs, x, y = build_celestial_wcs(
                az, el, ctype=("OLON-SFL", "OLAT-SFL"), crval=(0, 0), cdelt=cdelt[0:2], cunit=cunit[0:2]
            )
        else:
            x, y = wcs.all_world2pix(az, el, 0)

        if wcs.is_celestial:
            # extend the wcs for a third axis :
            wcs = extend_wcs(wcs, opds.flatten(), crval=0, ctype="OPD", cdelt=cdelt[2], cunit=cunit[2])
            # Round the crpix3
            wcs.wcs.crpix[2] = np.round(wcs.wcs.crpix[2])

        z = wcs.sub([3]).all_world2pix([opds.min(), opds.max()], 0)[0]

        shape = (
            np.round(z.max()).astype(np.int) + 1,
            np.round(y.max()).astype(np.int) + 1,
            np.round(x.max()).astype(np.int) + 1,
        )

        self.__log.info("Projecting data")

        # Null offsets for a beammap
        offsets = Table([Column(np.zeros_like(ikid), name="x0"), Column(np.zeros_like(ikid), name="y0")])
        outputs, weights, hits = sky_to_cube(data, opds, az, el, offsets, wcs, shape)

        # At this stage outputs shape is (nkids, nitg, nx, ny)

        # TODO: Check this
        # +: CUT hits/data/weight and wcs here
        if opd_trim is not None and isinstance(opd_trim, (int, np.int, float, np.float)):
            mostly_good_opd = (~np.isnan(outputs)).sum(axis=(0, 2, 3))
            mostly_good_opd = np.abs(mostly_good_opd / np.median(mostly_good_opd) - 1) < opd_trim
            _slice = slice(*np.nonzero(mostly_good_opd)[0][[0, -1]])
            hits = hits[:, _slice]
            outputs = outputs[:, _slice]
            weights = weights[:, _slice]
            wcs.wcs.crpix[2] -= _slice.start

        return (outputs, weights, hits), wcs

    def interferograms_psds(self, ikid=None, rebin=1, interpolation=False, **kwargs):

        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        datas = self.interferograms_pipeline(ikid=tuple(ikid), **kwargs)

        if interpolation is False:
            datas = datas.reshape(datas.shape[0], -1).filled(0)
        else:
            mask = datas.mask.reshape(datas.mask.shape[0], -1)
            datas = np.array(datas)
            datas = datas.reshape(datas.shape[0], -1)

            for ndet in np.arange(np.shape(datas)[0]):
                x_all = np.arange(np.shape(datas)[-1])
                x_fit = x_all[mask[ndet] == 0]

                interp = 0.01
                b, a = signal.butter(3, interp)

                data_original = datas[ndet, mask[ndet] == 0]
                data_fit = signal.filtfilt(b, a, data_original, method="gust")
                f = interpolate.interp1d(x_fit, data_fit, fill_value="extrapolate")

                datas[ndet, mask[ndet] > 0] = f(x_all[mask[ndet] > 0])

        Fs = self.param_c["acqfreq"]

        freq, psds = psd_cal(datas, Fs, rebin)

        return freq, psds

    def plot_interferograms_psds(self, ikid=None, rebin=1, **kwargs):
        if ikid is None:
            ikid = np.arange(len(self.list_detector))

        freq, psds = self.interferograms_psds(ikid=ikid, rebin=rebin, **kwargs)

        return (
            kids_plots.plot_psd(psds, freq, self.list_detector[ikid], **kwargs),
            freq,
            psds,
        )
