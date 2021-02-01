import numpy as np

from functools import partial
from multiprocessing import Pool

from scipy import ndimage
from scipy.signal import medfilt
from scipy.interpolate import interp1d

from .utils import correlated_median_removal, pca
from .utils import interferograms_regrid
from .utils import cpu_count

# Helper functions to pass large arrays in multiprocessing.Pool
_pool_global = None

N_CPU = cpu_count()


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def basic_continuum(bgrd, diff_mask=False, medfilt_size=None, **kwargs):
    """Time wise medium filtering."""
    # Only a rough Baseline for now...

    if diff_mask:
        # Try to flag the saturated part : where the signal change too much
        diff_bgrd = np.gradient(bgrd, axis=1) / bgrd.std(axis=1)[:, np.newaxis]
        diff_mask = np.abs(diff_bgrd) > 3 * diff_bgrd.std()
        diff_mask = ndimage.binary_dilation(diff_mask, [[True, True, True]])
        bgrd = np.ma.array(bgrd, mask=diff_mask)

    if medfilt_size is not None:
        bgrd -= np.array([medfilt(_bgrd, medfilt_size) for _bgrd in bgrd])

    bgrd -= np.nanmedian(bgrd, axis=1)[:, np.newaxis]

    return bgrd


def median_filtering(bgrd, ikid_ref=0, offset=True, flat=True, **kwargs):
    """Array wise median filtering."""

    # Note that bgrd content is lost here
    bgrd_cleanned, med_flatfield, med_offset, _ = correlated_median_removal(
        bgrd, iref=ikid_ref, offset=offset, flat=flat
    )

    if offset:
        bgrd_cleanned -= med_offset[:, np.newaxis]
    if flat:
        bgrd_cleanned /= med_flatfield[:, np.newaxis]

    return bgrd_cleanned


def pca_filtering(bgrd, ncomp=1, **kwargs):
    """PCA filtering."""
    # PCA needs zero centered values,
    bgrd = bgrd.T
    bgrd -= bgrd.mean(axis=0)

    # TODO: Check potential nan in timelines

    bgrd_PCA, _, eigen_vectors = pca(bgrd)

    bgrd_only = np.dot(bgrd_PCA[:, :ncomp], eigen_vectors[:, :ncomp].T).astype(bgrd.dtype)

    return (bgrd - bgrd_only).T


def _pool_interferograms_regrid(i_kid, bins=None):
    """Regrid interferograms to a common grid."""
    global _pool_global

    interferograms, laser = _pool_global

    return interferograms_regrid(interferograms[i_kid], laser, bins=bins, flatten=True)[0]


def common_itg(bgrd, laser=None, deg=None, bins="sqrt", flat=True, ncomp=None, **kwargs):
    """Common the temporal mean interferogram per kid and propagate it back"""
    # Retrieve the masked elements
    mask = bgrd == 0

    _, bins = np.histogram(laser.flatten(), bins=bins)
    c_bins = np.mean([bins[1:], bins[:-1]], axis=0)

    worker = partial(_pool_interferograms_regrid, bins=bins)
    with Pool(N_CPU, initializer=_pool_initializer, initargs=(bgrd, laser.flatten())) as p:
        common_mode_itg = p.map(worker, range(bgrd.shape[0]))

    common_mode_itg = np.array(common_mode_itg)

    if deg is not None:
        # Remove extra baseline on the common mode interferograms
        p = np.polynomial.polynomial.polyfit(c_bins, common_mode_itg.T, deg=deg)
        baselines = np.polynomial.polynomial.polyval(c_bins, p)
        common_mode_itg -= baselines

    # interpolate back into the interferaogram space
    common_mode = interp1d(c_bins, common_mode_itg, fill_value="extrapolate")(laser.flatten()).reshape(bgrd.shape)

    output = bgrd - common_mode

    output[mask] = 0

    # TODO: One could also derive flatfield and weights from this....
    if flat:
        flat = np.max(common_mode_itg, axis=1) - np.min(common_mode_itg, axis=1)
        flat = np.median(flat) / flat
        output *= flat[:, np.newaxis]

    if ncomp is not None:
        output = pca_filtering(output, **kwargs)

    return output
