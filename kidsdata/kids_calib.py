import warnings
from enum import Enum
import numpy as np
import dask.array as da
from functools import partial
from itertools import zip_longest

from multiprocessing import Pool
from scipy.ndimage.morphology import binary_erosion, binary_opening
from scipy.ndimage import uniform_filter1d

from astropy.stats import mad_std

from .utils import cpu_count, grouper, interp1d_nan
from .utils import fit_circle_3pts, fit_circle_algebraic, fit_circle_leastsq, radii

import logging


"""
Module for KIDs calibration a la KISS
"""
N_CPU = cpu_count()
logging.debug("KidsCalib Working on {} core".format(N_CPU))


def continuum(R0, P0, calfact):
    """Background based on calibration factors."""
    # In order to catch the potential RuntimeWarning which happens when some data can not be calibrated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bgrd = np.unwrap(R0 - P0, axis=1) * calfact
    return bgrd


def angle0(phi):
    return np.mod((phi + np.pi), (2 * np.pi)) - np.pi


def get_calfact(dataI, dataQ, A_masq, fmod=1, mod_factor=0.5, wsample=[], docalib=True):
    """
    Compute calibration to convert into frequency shift in Hz
    We fit a circle to the available data (2 modulation points + data)


    Parameters:
    -----------
    dataI, dataQ : array_like, shape (ndet, nint, nptint)
        the raw KIDS I & Q data
    A_masq : array_like, shape (nint, nptint)
        the corresponding modulation flags
    fmod : float
        the modulation frequency in Hz
    mod_factor : float
        factor to account for the difference between the registered modulation and the true one

    Ouput:
    -----

    """
    ndet, nint, nptint = dataI.shape

    calfact = np.zeros((ndet, nint), np.float32)
    Icc, Qcc = np.zeros((ndet, nint), np.float32), np.zeros((ndet, nint), np.float32)
    P0 = np.zeros((ndet, nint), np.float32)
    R0 = np.zeros((ndet, nint), np.float32)
    kidfreq = np.zeros_like(dataI)

    for iint in range(nint):  # single interferogram

        Icurrent = dataI[:, iint, :]
        Qcurrent = dataQ[:, iint, :]
        A_masqcurrent = A_masq[iint, :]

        l1 = A_masqcurrent == 3  # A_masq is the flag for calibration, values:-> 3: lower data
        l2 = A_masqcurrent == 1  # 1: higher data
        l3 = A_masqcurrent == 0  # 0: normal data

        # Make sure we have no issues with l1 and l2
        l1 = binary_erosion(l1, iterations=2)
        l2 = binary_erosion(l2, iterations=2)

        # remove first point in l3
        l3[:6] = False

        # Check for cases with missing data in one of the modulation (all flagged)
        if np.all(~l1) or np.all(~l2) or np.all(~l3):
            warnings.warn("Interferogram {} could not be calibrated".format(iint))
            continue

        x1 = np.median(Icurrent[:, l1], axis=1)
        y1 = np.median(Qcurrent[:, l1], axis=1)
        x2 = np.median(Icurrent[:, l2], axis=1)
        y2 = np.median(Qcurrent[:, l2], axis=1)
        x3 = np.median(Icurrent[:, l3], axis=1)
        y3 = np.median(Qcurrent[:, l3], axis=1)

        # Fit circle
        den = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        Ic = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)
        Qc = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ic /= den
            Qc /= den

        # Filter
        nfilt = 9
        if iint < nfilt:
            epsi = np.zeros(ndet) + 1.0 / np.double(iint + 1)
        else:
            epsi = np.zeros(ndet) + np.double(1.0 / nfilt)
        # epsi=1.0
        valIQ = (Ic * Ic) + (Qc * Qc)
        # CHECK : This will take the last element for the first interferogram
        dist = (Icc[:, iint - 1] - Ic) * (Icc[:, iint - 1] - Ic) + (Qcc[:, iint - 1] - Qc) * (Qcc[:, iint - 1] - Qc)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epsi[dist > 0.05 * valIQ] = 1.0

        # TODO: This could be vectorized
        if iint > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Ic = Ic * epsi + (1 - epsi) * Icc[:, iint - 1]
                Qc = Qc * epsi + (1 - epsi) * Qcc[:, iint - 1]

        Icc[:, iint] = Ic
        Qcc[:, iint] = Qc

        # Comupute circle radius and zero angle
        # TODO: Not used ??
        # rc = np.sqrt((x3 - Ic) * (x3 - Ic) + (y3 - Qc) * (y3 - Qc))
        P0[:, iint] = np.arctan2(Ic, Qc)

        # compute angle difference between two modulation points
        r0 = np.arctan2(Ic - x3, Qc - y3)
        R0[:, iint] = r0

        r1 = np.arctan2(Ic - x1, Qc - y1)
        r2 = np.arctan2(Ic - x2, Qc - y2)
        diffangle = angle0(r2 - r1)

        # Get calibration factor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diffangle[np.abs(diffangle) < 0.001] = 1.0

        calcoeff = 2 / diffangle
        calfact[:, iint] = calcoeff * fmod * mod_factor

        #        r = np.arctan2(Icc[:,iint]-Icurrent,np.transpose(Qcc[:,iint])-Qcurrent)
        r = np.arctan2(Icc[:, iint][:, np.newaxis] - Icurrent, Qcc[:, iint][:, np.newaxis] - Qcurrent)

        #        r = angleto0(np.arctan2((Icc[:,iint]-Icurrent.transpose()),\
        #                                (Qcc[:,iint]-Qcurrent.transpose())) - r0).transpose()
        ra = angle0(r - r0[:, np.newaxis])

        if docalib:
            kidfreq[:, iint, :] = calfact[:, iint][:, np.newaxis] * ra
        else:
            kidfreq[:, iint, :] = ra

    output = {
        "Icc": Icc,
        "Qcc": Qcc,
        "P0": P0,
        "R0": R0,
        "calfact": calfact,
        "continuum": continuum(R0, P0, calfact),
        "kidfreq": kidfreq,
    }

    return output


# Help function for multiprocessing, must be a module level
_pool_global = None


def _pool_reducs_initializer(*args):
    global _pool_global
    _pool_global = args


def _pool_reducs(iint, _reduc=np.median):
    # To be used with initialized pool
    global _pool_global
    _data, _low, _high, _normal = _pool_global

    # Remove empy iint indexes
    iint = [_idx for _idx in iint if _idx is not None]

    # When A_masq is strongly mask (in particular for A_low and A_high), they could be empty thus one need to catch error here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.array(
            [
                [_reduc(__data[:, low], axis=1), _reduc(__data[:, high], axis=1), _reduc(__data[:, normal], axis=1)]
                for __data, low, high, normal in zip(_data.swapaxes(0, 1)[iint], _low[iint], _high[iint], _normal[iint])
            ]
        )

    return result


class ModulationValue(Enum):
    normal = 0
    low = 3
    high = 1


def mod_masq_to_flag(A_masq, modulation):
    """Transform A_masq to clean flag value depending on the modulation value

    Parameters
    ----------
    A_masq : array_like
        the A_masq array
    modulation : ModulationValue
        the modulation for which to retrieve the flag
    """

    _masq = A_masq == modulation.value

    # Make sure we have no issues with l1 and l2
    structure = np.zeros((3, 3), np.bool)
    structure[1] = True

    # A_masq has problems when != (0,1,3), binary_closing opening,
    # up to 6 iterations (see scan 800 iint=7)
    _masq = binary_opening(_masq * 4, structure, output=_masq, iterations=4)

    # Remove 2 samples at the edges of _low and _high to insure proper values
    if modulation is not modulation.normal:
        _masq = binary_erosion(_masq * 2, structure, output=_masq, iterations=2)
    else:
        # remove the first 6 points of the normal mask
        _masq[:, :6] = False

    return _masq


def get_calfact_3pts(
    dataI,
    dataQ,
    mod_masq,
    fmod=1,
    mod_factor=0.5,
    method=("per", "3tps"),
    nfilt=9,
    sigma=None,
    _reduc=np.median,
    do_calib=True,
    fix_masq=True,
):
    """Compute calibration to converto into frequency shift in Hz.

    We fit circles per detector using the 3 modulations points

    Parameters
    ----------
    dataI, dataQ : array_like, shape (ndet, nint, nptint)
        the raw KIDS I & Q data
    mod_masq : array_like, shape (nint, nptint)
        the corresponding modulation flags
    fmod : float
        the modulation frequency in Hz
    mod_factor : float
        factor to account for the difference between the registered modulation and the true one
    method : (data_method, fit_method) tuple
        the calibration method to be used (per|all, 3pts|algebraic|leastsq) (default ("per", "3pts"), see Notes)
    nfilt : int
        the length of the top hat filter to apply to the circles positions (default: 9, None to disable)
    sigma : float, optional
        The number of standard deviations to use for clipping limit to flag the data (default: None, do not flag)
    _reduc : numpy ufunc
        the dimensionnality reduction function to be used on the 3 modulations (default : `numpyp.median`)
    do_calib : bool
        apply calibration to the kids frequency (default: True)

    Returns
    -------
    calib : dict
        calibrated data with keys
            * 'calfact'
            * 'Icc', 'Qcc' : array_like, shape (ndet, nint) or (ndet, )
            * 'R0', 'P0' : array_like, shape (ndet, nint) or (ndet, )
            * 'kidfreq' : array_like, shape (ndet, nint, nptint)
                the calibrated interferograms
            * 'continuum' : array_like, shape(ndet, nint)
                the calibrated continuum

    Notes
    -----
    if sigma is set, then a last element is returned
        * flag : boolean array_like
            the flag from the circle distances {ndet, nint, nptint}

    They are several methods which can be applied, from two classes :
        * data_method : str (all|per)
            fit the circle center on the all interferograms or per interferogram
        * fit_method : str (3pts|algebraic|leastsq)
            fit the above circles using fast algebraic 3 points method, generalized algebraic method, or least squares.
        * the combination ('all'|'3pts') can not be used

    """
    ndet, nint, nptint = dataI.shape

    # shape = dataI.shape
    # ndet, nint, nptint = shape

    A_low = mod_masq_to_flag(mod_masq, ModulationValue.low) if fix_masq else mod_masq == ModulationValue.low.value
    A_high = mod_masq_to_flag(mod_masq, ModulationValue.high) if fix_masq else mod_masq == ModulationValue.high.value
    A_normal = (
        mod_masq_to_flag(mod_masq, ModulationValue.normal) if fix_masq else mod_masq == ModulationValue.normal.value
    )

    # Selection of the good data for each mask and reduction of the data to one point with `_reduc`

    # WARNING : np.ma.median is VERY slow....
    # %timeit x1 = np.asarray([np.median(_data[:, mask], axis=1) for _data, mask in zip(dataI.swapaxes(0,1), A_low)]).swapaxes(0,1)
    # 1.79 s ± 4.19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # A_low = np.repeat(A_low[None, :, :], ndet, axis=0).reshape(shape)
    # A_high = np.repeat(A_high[None, :, :], ndet, axis=0).reshape(shape)
    # A_normal = np.repeat(A_normal[None, :, :], ndet, axis=0).reshape(shape)
    # dataI = np.ma.array(dataI)
    # dataI.mask = ~A_low
    # %timeit np.ma.median(dataI, axis=2)
    # 48.6 s ± 58.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # %timeit np.nanmedian(dataI.filled(np.nan), axis=2)
    # 15.4 s ± 70.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    # Loop over the interferogram axis for all 3 masks values
    # Slightly faster to have only one list comprehension

    # x = np.asarray(
    #     [
    #         [_reduc(_data[:, low], axis=1), _reduc(_data[:, high], axis=1), _reduc(_data[:, normal], axis=1)]
    #         for _data, low, high, normal in zip(dataI.swapaxes(0, 1), A_low, A_high, A_normal)
    #     ]
    # ).T.swapaxes(0, 1)
    # y = np.asarray(
    #     [
    #         [_reduc(_data[:, low], axis=1), _reduc(_data[:, high], axis=1), _reduc(_data[:, normal], axis=1)]
    #         for _data, low, high, normal in zip(dataQ.swapaxes(0, 1), A_low, A_high, A_normal)
    #     ]
    # ).T.swapaxes(0, 1)

    # Switch to multiprocessing
    _reducs = partial(_pool_reducs, _reduc=_reduc)
    with Pool(N_CPU, _pool_reducs_initializer, (dataI, A_low, A_high, A_normal)) as pool:
        x = np.vstack(pool.map(_reducs, grouper(range(nint), nint // N_CPU))).T.swapaxes(0, 1)

    with Pool(N_CPU, _pool_reducs_initializer, (dataQ, A_low, A_high, A_normal)) as pool:
        y = np.vstack(pool.map(_reducs, grouper(range(nint), nint // N_CPU))).T.swapaxes(0, 1)

    # Transform to dask array for later use
    dataI = da.from_array(dataI, name=False)
    dataQ = da.from_array(dataQ, name=False)

    data_method, fit_method = method
    if data_method.lower() == "all":
        # Fit circles on the full dataset
        x_fit = x.swapaxes(0, 1).reshape(ndet, -1)
        y_fit = y.swapaxes(0, 1).reshape(ndet, -1)

        # Remove potential nans when A_masq remove some of the points
        bad_data = np.isnan(x_fit) | np.isnan(y_fit)
        x_fit = x_fit[~bad_data].reshape(ndet, -1)
        y_fit = y_fit[~bad_data].reshape(ndet, -1)

        if fit_method.lower() == "algebraic":
            Ic, Qc = fit_circle_algebraic(x_fit.T, y_fit.T)
        elif fit_method.lower() == "leastsq":
            Ic, Qc = fit_circle_leastsq(x_fit, y_fit)
        else:
            raise ValueError("Unknown method {} (algebraic|leastsq) for {}".format(fit_method, data_method))

        Icc, Qcc = Ic[:, np.newaxis].astype(np.float32), Qc[:, np.newaxis].astype(np.float32)

        if sigma is not None:
            # R will be shaped (ndet, nint*nptint)
            R = radii((dataI.reshape(ndet, -1), dataQ.reshape(ndet, -1)), (Icc, Qcc))

    elif data_method.lower() == "per":
        # Fit circles per interferograms
        if fit_method.lower() == "3pts":
            Icc, Qcc = fit_circle_3pts(x, y)
        elif fit_method.lower() == "algebraic":
            Icc, Qcc = fit_circle_algebraic(x, y)
        elif fit_method.lower() == "leastsq":
            Icc, Qcc = fit_circle_leastsq(x.T.reshape(-1, 3), y.T.reshape(-1, 3)).reshape(2, nint, ndet).swapaxes(1, 2)
        else:
            raise ValueError("Unknown method {} (3pts|algebraic|leastsq) for {}".format(method, data_method))

        bad_interferograms = np.isnan(Icc) | np.isnan(Qcc)
        if np.any(bad_interferograms):
            warnings.warn("Interferogram {} could not be calibrated".format(np.unique(np.where(bad_interferograms)[1])))

        # filtering
        if nfilt is not None:
            # uniform_filter1d is sensitive to nan, thus if one fit fails, it will crash for the rest of the scan...
            if np.any(np.isnan(Icc)):
                Icc = np.apply_along_axis(interp1d_nan, 1, Icc)
            if np.any(np.isnan(Qcc)):
                Qcc = np.apply_along_axis(interp1d_nan, 1, Qcc)
            Icc = uniform_filter1d(Icc, nfilt, axis=1)
            Qcc = uniform_filter1d(Qcc, nfilt, axis=1)

        Icc, Qcc = Icc.astype(np.float32), Qcc.astype(np.float32)

        if sigma is not None:
            # R will be shaped (ndet, nint*nptint)
            R = radii((dataI, dataQ), (Icc[..., np.newaxis], Qcc[..., np.newaxis])).reshape(ndet, -1)

    P0 = np.arctan2(Icc, Qcc)

    x1, x2, x3 = x
    y1, y2, y3 = y

    R0 = np.arctan2(Icc - x3, Qcc - y3)
    r1 = np.arctan2(Icc - x1, Qcc - y1)
    r2 = np.arctan2(Icc - x2, Qcc - y2)
    diffangle = angle0(r2 - r1)

    # TODO: is it really needed....
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diffangle[np.abs(diffangle) < 0.001] = 1

    # Get calibration factor
    calfact = 2 / diffangle * fmod * mod_factor

    # %timeit r = angle0(da.arctan2(Icc[..., np.newaxis] - da.from_array(dataI), Qcc[..., np.newaxis] - da.from_array(dataQ)) - da.from_array(R0)[..., np.newaxis]).compute()
    # 19.8 s ± 173 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # or... if dataI and dataQ are already dask arrays
    # 6.6 s ± 74.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    # %timeit r = angle0(np.arctan2(Icc[..., np.newaxis] - dataI, Qcc[..., np.newaxis] - dataQ) - R0[..., np.newaxis])
    # 42 s ± 410 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    r = da.arctan2(Icc[..., np.newaxis] - dataI, Qcc[..., np.newaxis] - dataQ)
    kidfreq = np.asarray(angle0(r - R0[..., np.newaxis]))

    if do_calib:
        kidfreq *= calfact[..., np.newaxis]

    output = {
        "Icc": Icc,
        "Qcc": Qcc,
        "P0": P0,
        "R0": R0,
        "calfact": calfact,
        "continuum": continuum(R0, P0, calfact),
        "kidfreq": kidfreq,
    }

    if sigma is not None:
        residual = R - da.median(R, axis=1)[:, np.newaxis]
        # TODO: Should be rewritten for dask array
        std = mad_std(residual, axis=1)
        flag = np.abs(residual) > (sigma * std[:, np.newaxis])
        flag = flag.reshape(ndet, nint, nptint)

        output["calib_flag"] = flag

    return output
