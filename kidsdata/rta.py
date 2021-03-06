import os
import numpy as np
from itertools import chain
from pathlib import Path
from functools import wraps, partial
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.optimize import curve_fit

from astropy.stats import mad_std
from astropy.wcs import WCS
from astropy.table import Table
from astropy.io import fits

from .database.api import get_scan
from .kiss_data import KissData
from .kids_plots import show_contmap

from kidsdata.settings import CALIB_DIR

plt.ion()

__all__ = ["beammap", "contmap", "contmap_coadd", "check_pointing", "skydip"]


def read_scan(scan, array=None, extra_data=None):
    """Read all common and some additionnal extra

    Parameters
    ----------
    scan : int
        the scan number
    array : str (A|B|None)
        read only one array (default: all)
    extra_data : list of str
        list of extra data to read

    Returns
    -------
    kd : KissRawData
        the corresponding object
    """
    kd = KissData(get_scan(scan))

    list_data = kd.names.DataSc + kd.names.DataUc

    # Do not use F_sky_* from file....
    remove_Uc = ["F_sky_Az", "F_sky_El"]
    remove_Ud = ["F_tel_Az", "F_tel_El"]

    for item in chain(remove_Uc, remove_Ud):
        if item in list_data:
            list_data.remove(item)

    # Add extra data at read time directly... otherwise there is a core dump...
    if extra_data is not None:
        list_data = list_data + extra_data
    list_detector = kd.get_list_detector(array, flag=0, typedet=1)

    # Read data
    kd.read_data(list_data=list_data, list_detector=list_detector, silent=True)

    return kd


# Following David Baezly generic pattern
# https://rednafi.github.io/digressions/python/2020/05/13/python-decorators.html
def kd_or_scan(func=None, array=None, extra_data=None):
    """Decorator to allow functions to be call with a scan number or kd object """

    if func is None:
        return partial(kd_or_scan, array=array, extra_data=extra_data)

    @wraps(func)
    def wrapper(scan, *args, **kwargs):
        # If scan number given, read the scan into the object and pass it to function
        if isinstance(scan, (int, np.int, np.int64)):
            scan = read_scan(scan, array=array, extra_data=extra_data)

        return func(scan, *args, **kwargs)

    return wrapper


@kd_or_scan(array=None, extra_data=["I", "Q"])
def beammap(kd: KissData):
    """Display a beammap.

    Parameters
    ----------
    kd : `kissdata.KissRawData` or int
        the KissRawData object to check or scan number to read

    Returns
    -------
    kd, (fig_beammap, fig_geometry)
        return the read  `kissdata.KissRawData`, as well as the beammap and geometry figures

    """
    kd._KissRawData__check_attributes(["R0", "P0", "calfact", "F_sky_Az", "F_sky_El", "A_hours", "A_time_pps"])
    # Compute & plot beammap
    fig_beammap, (_, _, kidpar) = kd.plot_beammap(coord="pdiff", flatfield=None)

    # Apply kidpar to dataset
    kd._extended_kidpar = kidpar

    # plot geometry
    fig_geometry, fwhm = kd.plot_kidpar()

    # Plot the combined map
    # with default KIDs selection
    kid_mask = kd._kids_selection()
    ikid = np.where(kid_mask)[0]

    if len(ikid) > 5:
        fig_coadd, _ = kd.plot_contmap(coord="pdiff", ikid=ikid, cdelt=0.05)
    else:
        fig_coadd = None

    return kd, (fig_beammap, fig_geometry, fig_coadd)


@kd_or_scan(array=None, extra_data=["I", "Q"])
def contmap(kd, e_kidpar="e_kidpar_median.fits", cm_func="kidsdata.common_mode.pca_filtering", **kwargs):
    """Display a continuum map.

    Parameters
    ----------
    kd : `kissdata.KissRawData` or int
        the KissRawData object to check or scan number to read

    Returns
    -------
    kd, (fig_beammap, fig_geometry)
        return the read  `kissdata.KissRawData`, as well as the beammap and geometry figures

    """
    kd._KissRawData__check_attributes(
        ["R0", "P0", "calfact", "mask_tel", "F_sky_Az", "F_sky_El", "A_hours", "A_time_pps"]
    )
    kd._extended_kidpar = Table.read(Path(CALIB_DIR) / e_kidpar)

    # kids selection
    kid_mask = kd._kids_selection(std_dev=0.3)
    ikid_KA = np.where(kid_mask & np.char.startswith(kd.list_detector, "KA"))[0]
    ikid_KB = np.where(kid_mask & np.char.startswith(kd.list_detector, "KB"))[0]
    ikid_KAB = np.concatenate([ikid_KA, ikid_KB])

    # Compute & plot continuum map
    fig, _ = kd.plot_contmap(
        ikid=[ikid_KA, ikid_KB, ikid_KAB], coord="pdiff", flatfield="amplitude", cm_func=cm_func, **kwargs
    )

    return kd, fig


def contmap_coadd(scans, e_kidpar="e_kidpar_median.fits", cm_func="kidsdata.common_mode.pca_filtering", **kwargs):
    """Continuum coaddition of several scans.

    Parameters
    ----------
    scans : list of int
        the list of scans to be coadd
    e_kidpar: str
        the extended kidpar filename to be used
    cm_func : str
        the continuum pipeline function to be used...
    **kwargs:
        ... and its additionnal keyword

    Returns
    -------
    fake_kd : namedtuple
       a fake KissRawData to be used with kids_plot.show_contmap
    fig : matplotlib.figure
       the displayed figure
    combined_map, combined_weights : astropy.io.fits.ImageHDU
       the combined data and weights

    Notes
    -----

    the resulting arguments can be displayed with show_contmap
    >>> from kidsdata.kids_plots import show_contmap
    >>> show_contmap(fake_kd, [combined_map], [combined_weights], None)

    """

    # Define a common wcs/shape
    cdelt = kwargs.get("cdelt", 0.01)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("OLON-SFL", "OLAT-SFL")
    wcs.wcs.cdelt = (cdelt, cdelt)
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = (100, 100)
    shape = (200, 200)

    results = []
    for scan in scans:
        kd = read_scan(scan, extra_data=["I", "Q"])
        kd._KissRawData__check_attributes(["R0", "P0", "calfact", "F_sky_Az", "F_sky_El", "A_hours", "A_time_pps"])
        kd._extended_kidpar = Table.read(Path(CALIB_DIR) / e_kidpar)

        # kids selection
        kid_mask = kd._kids_selection(std_dev=0.3)
        ikid_KA = np.where(kid_mask & np.char.startswith(kd.list_detector, "KA"))[0]
        ikid_KB = np.where(kid_mask & np.char.startswith(kd.list_detector, "KB"))[0]
        ikid_KAB = np.concatenate([ikid_KA, ikid_KB])

        data, weight, hit = kd.continuum_map(
            ikid=ikid_KAB, coord="pdiff", wcs=wcs, shape=shape, flatfield="amplitude", cm_func=cm_func, **kwargs
        )
        del kd
        results.append((data, weight))

    data = [np.ma.array(result[0].data, mask=np.isnan(result[0].data)) for result in results]
    weights = [np.ma.array(result[1].data) for result in results]

    combined_map, combined_weights = np.ma.average(data, axis=0, weights=weights, returned=True)

    header = wcs.to_header()
    header["SCANS"] = str(scans)

    combined_map = fits.ImageHDU(combined_map.filled(np.nan), header, name="data")
    combined_weights = fits.ImageHDU(combined_weights, header, name="weight")

    FakeKissRawData = namedtuple("FakeKissRawData", ["source", "filename"])
    fake_kd = FakeKissRawData(source="dummy", filename="Coadd {}".format(scans))

    fig = show_contmap([combined_map], [combined_weights], None)
    fig.suptitle(fake_kd.filename)

    return fake_kd, fig, (combined_map, combined_weights)


@kd_or_scan
def check_pointing(kd):
    """Check the pointing against the source position.

    Parameters
    ----------
    kd : `kissdata.KissRawData` or int
        the KissRawData object to check or scan number to read

    Returns
    -------
    kd, (fig_pointing)
        return the read  `kissdata.KissRawData`, as well as the pointing figures
    """
    # from .kiss_pointing_model import KISSPmodel

    kd._KissRawData__check_attributes(["F_sky_Az", "F_sky_El", "F_tl_Az", "F_tl_El", "A_hours", "A_time_pps"])

    fig_pointing, ax = plt.subplots()
    mask = kd.telescope_position.mask
    ax.plot(kd.F_sky_Az[~mask], kd.F_sky_El[~mask], label="F_sky")
    ax.plot(kd.F_tl_Az[~mask], kd.F_tl_El[~mask], label="F_tl")
    # ax.plot(*KISSPmodel().telescope2sky(kd.F_tl_Az[mask], kd.F_tl_El[mask]), label="F_sky computed")
    # ax.plot(*KISSPmodel(model="Q1").telescope2sky(kd.F_tl_Az[mask], kd.F_tl_El[mask]), label="F_sky Q1 computed")

    obstime = kd.obstime[mask]
    interp_az, interp_el = kd.get_object_altaz(npoints=100)
    ax.plot(interp_az(obstime.mjd), interp_el(obstime.mjd), label=kd.source)
    ax.legend(loc="best")
    ax.set_xlabel("Az [deg]")
    ax.set_ylabel("El [deg]")
    fig_pointing.suptitle(kd.filename)

    return kd, fig_pointing


def skydip(scans):
    """Fit a powerlaw to skydip scans.

    Parameters
    ----------
    scans : list of string
        the list of scans to use (see Notes)

    Returns
    -------
    fig_skydip_fit, sig_skydip_stat
        the two output figures

    Notes
    -----
    One has to give a list of filename, which can be obtained with `get_manual`
    >>> from datetime import datetime
    >>> from kidsdata import get_manual
    >>> scans = get_manual(start=datetime(2019, 5, 1, 19, 14, 24),
                          end=datetime(2019, 5, 1, 19, 52, 53))

    >>> skydip(scans)
    will produce a skydip fit with all the scans from 2019-05-01T19:14:24 to 2019-05-01T19:52:53
    """
    title = Path(scans[0]).name + " ".join([Path(scan).name.split("_")[4] for scan in scans[1:]])

    signal = []
    std = []
    elevation = []

    for scan in scans:
        kd = KissData(scan)
        kd.read_data(list_data=["A_masq", "I", "Q", "F_tone", "F_tl_Az", "F_tl_El"])

        # TODO: Why do we need copy here, seems that numpy strides are making
        # funny things here !

        F_tone = 1e3 * kd.F_tone.copy().mean(1)[:, np.newaxis] + kd.continuum
        signal.append(F_tone.mean(1))
        std.append(F_tone.std(1))
        elevation.append(kd.F_tl_El.mean())

    signal = np.array(signal)
    std = np.array(std)
    elevation = np.array(elevation)
    detectors = kd.list_detector

    # rearrange signal to be coherent with the fit ?
    signal_new = 2 * signal[:, 0][:, np.newaxis] - signal

    air_mass = 1.0 / np.sin(np.radians(elevation))

    def T(
        airm, const, fact, tau_f
    ):  # signal definition for skydip model: there is -1 before B to take into account the increasing resonance to lower optical load
        return const + 270.0 * fact * (1.0 - np.exp(-tau_f * airm))

    popts = []
    pcovs = []
    for _sig, _std in zip(signal_new.T, std.T):
        P0 = (4e8, 1e8, 1.0)
        popt, pcov = curve_fit(T, air_mass, _sig, sigma=_sig, p0=P0, maxfev=100000)

        popts.append(popt)
        pcovs.append(pcovs)

    popts = np.array(popts)

    ndet = popts.shape[0]
    fig_skydip_fit, axes = plt.subplots(
        np.int(np.sqrt(ndet)), np.int(ndet / np.sqrt(ndet)), sharex=True
    )  # , sharey=True)
    for _sig, _std, popt, detector, ax in zip(signal_new.T, std.T, popts, detectors, axes.flatten()):
        ax.errorbar(air_mass, _sig, _std)
        ax.plot(air_mass, T(air_mass, *popt))
        ax.set_title(detector, pad=-15)
        ax.label_outer()

    fig_skydip_fit.suptitle(title)
    fig_skydip_fit.tight_layout()
    fig_skydip_fit.subplots_adjust(wspace=0, hspace=0)

    Ao, Bo, tau = popts.T

    fig_skydip_stat, axes = plt.subplots(1, 3)
    for (item, value), ax in zip({r"$A_0$": Ao, r"$B_0$": Bo, "tau": tau}.items(), axes):
        mean_value = np.nanmedian(value)
        std_value = mad_std(value, ignore_nan=True)
        range_value = np.array([-3, 3]) * std_value + mean_value
        ax.hist(value, range=range_value)
        ax.set_xlabel(item)
    fig_skydip_stat.suptitle(title)

    return fig_skydip_fit, fig_skydip_stat
