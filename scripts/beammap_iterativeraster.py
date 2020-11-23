#!/bin/env python
# SBATCH -J beammap_iterativeraster
# SBATCH -N 1
# SBATCH -c 8
# SBATCH --mem=60GB
# SBATCH -o slurm-%A_%a.out
# SBATCH -e slurm-%A_%a.err
"""
Process all moon iterativeraster scan to derive beammaps

"""


import os
import sys
import warnings
import datetime
import functools
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt

# We should be in non interactive mode...
if int(os.getenv("SLURM_ARRAY_TASK_COUNT", 0)) > 0:
    logging.info("Within sbatch...")
    mpl.use("Agg")
    # To avoid download concurrency,
    # but need to be sure that the cache is updated
    from astropy.utils.data import conf

    conf.remote_timeout = 120
    conf.download_cache_lock_attempts = 120

    # from astropy.utils import iers
    # iers.conf.auto_download = False
    # iers.conf.auto_max_age = None

    # Do not work
    # from astropy.config import set_temp_cache
    # set_temp_cache('/tmp')
else:
    plt.ion()

# For some reason, this has to be done BEFORE importing anything from kidsdata, otherwise, it does not work...
logging.basicConfig(
    level=logging.DEBUG,  # " stream=sys.stdout,
    # format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Prevent verbose output from matplotlib
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import numpy as np
from pathlib import Path
from multiprocessing import Pool
from itertools import zip_longest, islice

from kidsdata import *
from kidsdata.kiss_continuum import KissContinuum

from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt


plt.ion()


# https://docs.python.org/fr/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def process_scan(scan):
    try:
        kd, (fig_beammap, figs_geometries, fig_coadd) = beammap(scan)

        fig_beammap.savefig(f"beammap_{kd.scan}.png")
        plt.close(fig_beammap)

        for i, fig in enumerate(figs_geometries):
            fig.savefig(f"geometry_{kd.scan}_{i}.png")
            plt.close(fig)

        if fig_coadd is not None:
            fig_coadd.savefig(f"coadd_{kd.scan}.png")
            plt.close(fig_coadd)

        e_kidpar = kd._extended_kidpar
        e_kidpar.write(f"e_kidpar_{kd.scan}.fits", overwrite=True)
        return True
    except (TypeError, ValueError, IndexError) as e:
        return False


from autologging import logged


@logged
def process_scan_new_calib(scan, outdir=None):
    try:
        process_scan_new_calib._log.info("Reading")
        kd = KissData(get_scan(scan))
        kd.read_data(
            list_data=[
                "indice",
                "A_masq",
                "I",
                "Q",
                "F_tl_Az",
                "F_tl_El",
                "F_state",
                "F_subsc",
                "F_nbsubsc",
                "E_X",
                "E_status",
                "u_itfamp",
                "C_laser1_pos",
                "C_laser2_pos",
                "A_time_ntp",
                "A_time_pps",
                "A_time",
                "A_hours",
                "B_time_ntp",
                "B_time_pps",
                "B_time",
                "B_hours",
            ]
        )
        # # BEFORE
        # process_scan_new_calib._log.info('before')
        # suffix = "calib_raw"
        # process_scan_new_calib._log.info('calib_raw')
        # kd.calib_raw()
        # process_scan_new_calib._log.info('beammap()')
        # kd, (fig_beammap, figs_geometries, fig_coadd) = beammap(kd)

        # process_scan_new_calib._log.info('save_figs')
        # fig_beammap.savefig(f"beammap_{kd.scan}_{suffix}.png")
        # plt.close(fig_beammap)

        # for i, fig in enumerate(figs_geometries):
        #     fig.set_size_inches(15,10)
        #     fig.savefig(f"geometry_{kd.scan}_{i}_{suffix}.png")
        #     plt.close(fig)

        # if fig_coadd is not None:
        #     fig_coadd.savefig(f"coadd_{kd.scan}_{suffix}.png")
        #     plt.close(fig_coadd)

        # e_kidpar = kd._extended_kidpar
        # e_kidpar.write(f"e_kidpar_{kd.scan}_{suffix}.fits", overwrite=True)

        # AFTER
        # process_scan_new_calib._log.info("after")

        suffix = "all_leastsq"
        process_scan_new_calib._log.info("get_calfact_3pts")
        kd.calib_raw(calib_func="kidsdata.kids_calib.get_calfact_3pts", method=("all", "leastsq"), nfilt=None)
        # KissRawData.continuum.fget.cache_clear()
        KissContinuum.continuum_pipeline.cache_clear()

        process_scan_new_calib._log.info("beammap")
        kd, (fig_beammap, figs_geometries, fig_coadd) = beammap(kd)

        process_scan_new_calib._log.info("save_figs")
        fig_beammap.savefig(outdir / f"beammap_{kd.scan}_{suffix}.png")
        plt.close(fig_beammap)

        for i, fig in enumerate(figs_geometries):
            fig.set_size_inches(15, 10)
            fig.savefig(outdir / f"geometry_{kd.scan}_{i}_{suffix}.png")
            plt.close(fig)

        if fig_coadd is not None:
            fig_coadd.savefig(outdir / f"coadd_{kd.scan}_{suffix}.png")
            plt.close(fig_coadd)

        e_kidpar = kd._extended_kidpar
        e_kidpar.write(outdir / f"e_kidpar_{kd.scan}_{suffix}.fits", overwrite=True)

        return True
    except (TypeError, ValueError, IndexError) as e:
        return False


def combine_kidpars(indir=Path("."), suffix="", kid_threshold=0.9, inside_threshold=0.9):
    filenames = [
        filename for filename in indir.glob("e_kidpar_*{}.fits".format(suffix)) if "median" not in filename.name
    ]
    logging.info("Starting with : {} kidpar".format(len(filenames)))

    kidpars = []
    frac_kids = []
    frac_insides = []
    # Selection of "godd" kidpar
    for filename in filenames:
        _kidpar = Table.read(filename)
        nofit_kid = np.isnan(_kidpar["amplitude"]) | np.isnan(_kidpar["x0"]) | np.isnan(_kidpar["y0"])
        # fraction of fitted kids
        frac_kid = np.mean(~nofit_kid)
        # Fraction within 1 degree
        frac_inside = np.mean(np.sqrt(_kidpar["x0"][~nofit_kid] ** 2 + _kidpar["y0"][~nofit_kid] ** 2) < 1)
        frac_kids.append(frac_kid)
        frac_insides.append(frac_inside)
        if frac_kid > kid_threshold and frac_inside > inside_threshold:
            kidpars.append(_kidpar)

    logging.info("Selecting {} kidpars".format(len(kidpars)))

    # Lookup the most present/valid KIDs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_kidpar = vstack(kidpars).group_by("namedet")

    idx_mostpresent = np.argmax([np.sum(~np.isnan(_kidpar["amplitude"])) for _kidpar in all_kidpar.groups])
    mostpresent = all_kidpar.groups.keys[idx_mostpresent]["namedet"]
    logging.info("Most present kid {}".format(mostpresent))

    freq_mostpresent = [np.mean(~np.isnan(_kidpar["amplitude"])) for _kidpar in all_kidpar.groups]
    mask_mostpresent = np.array(freq_mostpresent) == np.max(freq_mostpresent)
    mostpresent = all_kidpar.groups.keys[mask_mostpresent]["namedet"]
    logging.info("Most present kids {}".format(mostpresent))

    # Normalize amplitude and position to the most presents KID
    for kidpar in kidpars:
        kidpar.add_index(["namedet"])
        kidpar["amplitude"] /= np.nanmedian(kidpar.loc[mostpresent]["amplitude"])
        kidpar["x0"] -= np.nanmedian(kidpar.loc[mostpresent]["x0"])
        kidpar["y0"] -= np.nanmedian(kidpar.loc[mostpresent]["y0"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_kidpar = vstack(kidpars).group_by("namedet")

    # Median aggregate
    median_kidpar = all_kidpar.groups.aggregate(np.nanmedian)
    median_kidpar.add_index(["namedet"])

    # # Check distribution of kidpars
    # # Construct 2D histograms per kid of all positions
    # hists = []
    # for _kid_par in all_kidpar.groups:
    #     hists.append(np.histogram2d(_kid_par['y0'], _kid_par['x0'], range=[[-0.5, 0.5], [-0.5, 0.5]], bins=100)[0])
    # # Plot them with overplot of median position
    # from mpl_toolkits.axes_grid1 import ImageGrid
    # fig = plt.figure()
    # n_kids = len(hists)
    # nx = np.ceil(np.sqrt(n_kids)).astype(np.int)
    # ny = np.ceil(n_kids / nx).astype(np.int)
    # grid = ImageGrid(fig, 111, nrows_ncols=(ny, nx), axes_pad=0, ngrids=n_kids, share_all=True)
    # for ax, im, _kidpar in zip(grid, hists, median_kidpar):
    #     ax.imshow(im, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])
    #     ax.scatter(_kidpar['x0'], _kidpar['y0'], c='r', alpha=0.5)
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)

    # Centralize positions
    median_kidpar["x0"] -= np.nanmedian(median_kidpar["x0"])
    median_kidpar["y0"] -= np.nanmedian(median_kidpar["y0"])

    # Normalize amplitude
    median_kidpar["amplitude"] /= np.nanmedian(median_kidpar["amplitude"])

    filename = "e_kidpar_median_{}_{}.fits".format(suffix, datetime.datetime.now().isoformat(timespec="hours"))

    median_kidpar.meta["scan"] = "median of {} scans".format(len(kidpars))
    median_kidpar.meta["filename"] = filename
    median_kidpar.meta["created"] = datetime.datetime.now().isoformat()
    median_kidpar.add_index(["namedet"])
    median_kidpar.write(filename, overwrite=True)


# filenames = ['kidpos_median.fits', 'e_kidpar_median.fits', 'e_kidpar_median_calib_raw.fits', 'e_kidpar_median_all_leastsq.fits']
def compare_kidpars(filenames=["kidpos_median.fits"]):

    ## Plotting and comparing
    kidpars = []
    for filename in filenames:
        kidpar = Table.read(filename)
        if filename == "kidpos_median.fits":  # Transform to our names
            kidpar = Table.read(filename)
            kidpar.rename_column("name", "namedet")
            kidpar.rename_column("off_az", "x0")
            kidpar.rename_column("off_el", "y0")
            kidpar["x0"] -= np.nanmedian(kidpar["x0"])
            kidpar["y0"] -= np.nanmedian(kidpar["y0"])
            kidpar.meta["FILENAME"] = filename
        kidpar.add_index(["namedet"])
        kidpars.append(kidpar)

    # Register positions with the first pixel, should be present in all kidpars
    _kid = kidpars[0]["namedet"][0]
    for kidpar in kidpars:
        kidpar["x0"] -= kidpar.loc[_kid]["x0"] - kidpars[0].loc[_kid]["x0"]
        kidpar["y0"] -= kidpar.loc[_kid]["y0"] - kidpars[0].loc[_kid]["y0"]

    fig, axes = plt.subplots(nrows=2, ncols=len(kidpars), sharex=True, sharey=True)
    for _axes, kidpar in zip(axes.T, kidpars):
        _axes[0].set_title(kidpar.meta["FILENAME"])
        KA = [name.startswith("KA") for name in kidpar["namedet"]]
        KB = [name.startswith("KB") for name in kidpar["namedet"]]
        for ax, _kidpar in zip(_axes, [kidpar[KA], kidpar[KB]]):
            ax.scatter(_kidpar["x0"], _kidpar["y0"])
            ax.set_aspect("equal")
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.6, 0.6)
            ax.set_xlabel("lon offset [deg]")
            ax.set_ylabel("lat offset [deg]")

    fig.set_size_inches(5 * len(kidpars), 10)
    fig.tight_layout()
    return fig


def main():

    kwargs = {}

    # All Moon scans
    db = list_scan(output=True, source="Moon", obsmode="ITERATIVERASTER")
    scans = [_["scan"] for _ in db]
    kwargs["outdir"] = Path("Moon")

    # Test on new calibration
    process_scan = process_scan_new_calib

    # Single thread
    # for scan in scans:
    #     process(scan)

    # Multiprocessing threads
    # process = functools.partial(process_scan, **kwargs)
    # with Pool(10) as p:
    #     print(p.map(process, scans))

    # Slurm...
    i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    n = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    _scans = nth(zip(*grouper(scans, n)), i)

    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(message)s",
    #     handlers=[logging.FileHandler(f"beammap_iterativeraster_{i}.log"), logging.StreamHandler()],
    # )

    logging.info("{} will do {}".format(i, _scans))

    if _scans is None:
        return None

    for scan in _scans:
        if scan is None:
            continue
        logging.info("{} : {}".format(i, scan))
        process_scan(scan, **kwargs)


if __name__ == "__main__":
    main()
