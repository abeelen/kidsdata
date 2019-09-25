import numpy as np
from scipy import ndimage
from scipy.signal import medfilt


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
