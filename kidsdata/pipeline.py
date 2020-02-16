import numpy as np
from scipy import ndimage
from scipy.signal import medfilt
from .utils import correlated_median_removal, pca


def basic_continuum(self, ikid, diff_mask=False, medfilt_size=None, **kwargs):

    # Only a rough Baseline for now...
    bgrd = self.continuum[ikid]
    if diff_mask:
        # Try to flag the saturated part : where the signal change too much
        diff_bgrd = np.gradient(bgrd, axis=1) / bgrd.std(axis=1)[:, np.newaxis]
        diff_mask = np.abs(diff_bgrd) > 3 * diff_bgrd.std()
        diff_mask = ndimage.binary_dilation(diff_mask, [[True, True, True]])
        bgrd = np.ma.array(bgrd, mask=diff_mask)

    if medfilt_size is not None:
        bgrd -= np.array([medfilt(_bgrd, medfilt_size) for _bgrd in bgrd])

    bgrd -= np.median(bgrd, axis=1)[:, np.newaxis]

    return bgrd


def median_continuum(self, ikid, ikid_ref=0, **kwargs):

    bgrd = self.continuum[ikid]

    bgrd_cleanned, flat, offset, _ = correlated_median_removal(bgrd, iref=ikid_ref)

    return (bgrd_cleanned-offset[:, np.newaxis])/flat[:, np.newaxis]

def pca_continuum(self, ikid, ncomp=1, **kwargs):

    bgrd = self.continuum[ikid]

    # PCA needs zero centered values
    bgrd = bgrd.T - bgrd.T.mean(axis=0)

    bgrd_PCA, _, eigen_vectors = pca(bgrd)

    bgrd_only = np.dot(bgrd_PCA[:, :ncomp], eigen_vectors[:, :ncomp].T)

    return (bgrd - bgrd_only).T