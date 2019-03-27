import numpy as np
from scipy.ndimage.morphology import binary_erosion

"""
Module for kiss calibration
"""


def angle0(phi):
    return np.mod((phi + np.pi), (2 * np.pi)) - np.pi


def get_calfact(kids, Modfactor=0.5, wsample=[], docalib=True):
    """
    Compute calibration to convert into frequency shift in Hz
    We fit a circle to the available data (2 modulation points + data)


    Parameters:
    -----------
    - data : (object KID data)
      data object from KID data class containing I, Q and A_masq at least

    - Modfact: double (optional)
      Factor to account for the difference between the registered modulation and
      the true one

    Ouput:
    -----
    - calfact: (np.array)
       calibration factor for all detectors

    - Icc, Qcc:  (np.array, np.array)
      center of the cirle for all detectors

    - P0: (np.array)
      Angle with respect to (0,0)

    """

    ndet = kids.ndet
    nptint, nint = kids.nptint, kids.nint
    fmod = kids.param_c['1-modulFreq']  # value [Hz] for the calibration

    calfact = np.zeros((ndet, nint), np.float32)
    Icc, Qcc = np.zeros((ndet, nint), np.float32), np.zeros((ndet, nint), np.float32)
    P0 = np.zeros((ndet, nint), np.float32)
    R0 = np.zeros((ndet, nint), np.float32)

    amask = kids.A_masq.reshape(nint, nptint)
    dataI = kids.I.reshape(ndet, nint, nptint)
    dataQ = kids.Q.reshape(ndet, nint, nptint)

    kidfreq = np.zeros_like(dataI)

    for iint in range(nint):    # single interferogram

        Icurrent = dataI[:, iint, :]
        Qcurrent = dataQ[:, iint, :]
        A_masqcurrent = amask[iint, :]

        l1 = A_masqcurrent == 3  # A_masq is the flag for calibration, values:-> 3: lower data
        l2 = A_masqcurrent == 1  # 1: higher data
        l3 = A_masqcurrent == 0  # 0: normal data

        # Make sure we have no issues with l1 and l2
        l1 = binary_erosion(l1, iterations=2)
        l2 = binary_erosion(l2, iterations=2)

        # remove first point in l3
        l3[:6] = False

        x1 = np.median(Icurrent[:, l1], axis=1)
        y1 = np.median(Qcurrent[:, l1], axis=1)
        x2 = np.median(Icurrent[:, l2], axis=1)
        y2 = np.median(Qcurrent[:, l2], axis=1)
        x3 = np.median(Icurrent[:, l3], axis=1)
        y3 = np.median(Qcurrent[:, l3], axis=1)

        # Fit circle
        den = 2.0 * (x1 * (y2 - y3) +
                     x2 * (y3 - y1) +
                     x3 * (y1 - y2))

        Ic = (x1 * x1 + y1 * y1) * (y2 - y3) + \
            (x2 * x2 + y2 * y2) * (y3 - y1) + \
            (x3 * x3 + y3 * y3) * (y1 - y2)
        Ic /= den

        Qc = (x1 * x1 + y1 * y1) * (x3 - x2) + \
            (x2 * x2 + y2 * y2) * (x1 - x3) + \
            (x3 * x3 + y3 * y3) * (x2 - x1)
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
        epsi[dist > 0.05 * valIQ] = 1.0

        # TODO: This could be vectorized
        if iint > 0:
            Ic = Ic * epsi + (1 - epsi) * Icc[:, iint - 1]
            Qc = Qc * epsi + (1 - epsi) * Qcc[:, iint - 1]

        Icc[:, iint] = Ic
        Qcc[:, iint] = Qc

        # Comupute circle radius and zero angle
        # TODO: Not used ??
        # rc = np.sqrt((x3 - Ic) * (x3 - Ic) + (y3 - Qc) * (y3 - Qc))
        P0[:, iint] = np.arctan2(Ic, Qc)

        # compute angle difference between two modulation points
        R0[:, iint] = np.arctan2(Ic - x3, Qc - y3)

        r1 = np.arctan2(Ic - x1, Qc - y1)
        r2 = np.arctan2(Ic - x2, Qc - y2)
        diffangle = angle0(r2 - r1)

        # Get calibration factor
        diffangle[np.abs(diffangle) < 0.001] = 1.0

        calcoeff = 2 / diffangle
        calfact[:, iint] = calcoeff * fmod * Modfactor


#        r = np.arctan2(Icc[:,iint]-Icurrent,np.transpose(Qcc[:,iint])-Qcurrent)
        r = np.arctan2((Icc[:, iint] - Icurrent.transpose()).transpose(),
                       (Qcc[:, iint] - Qcurrent.transpose()).transpose())

#        r = angleto0(np.arctan2((Icc[:,iint]-Icurrent.transpose()),\
#                                (Qcc[:,iint]-Qcurrent.transpose())) - r0).transpose()
        ra = angle0((r.transpose() - r0).transpose())

        if (docalib):
            kidfreq[:, iint, :] = (calfact[:, iint] * ra.transpose()).transpose()
        else:
            kidfreq[:, iint, :] = ra

    return calfact, Icc, Qcc, P0, R0, kidfreq
