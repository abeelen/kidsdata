#!/bin/env python
#SBATCH --job-name=shift_flip
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=100GB
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0

# Launch with :
# > sbatch template.py *args
# Or
# > sbatch --array=0-19 template.py *args
# To parallelize on 20 task (with 3 CPUs and 50 GB each)
# See help for more options
"""
Compute Shifts between mirror position and interferogram maximum for all scans
"""

import os
import logging
import argparse
import numpy as np
import matplotlib as mpl
from pathlib import Path

from scipy.ndimage.morphology import binary_dilation, binary_opening
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# For some reason, this has to be done BEFORE importing anything from kidsdata, otherwise, it does not work...
logging.basicConfig(
    level=logging.DEBUG,  # " stream=sys.stdout,
    # format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Prevent verbose output from matplotlib
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

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

import warnings
from autologging import logged, TRACE
from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table, Column, vstack

from kidsdata import KissData, get_scan, list_scan, list_extra
from kidsdata.utils import interferograms_regrid

CALIB_PATH = Path("/data/KISS/Calib")

# Helper functions to pass large arrays in multiprocessing.Pool
_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _pool_interferograms_regrid(i_kid, bins=None):
    """Regrid interferograms to a common grid."""
    global _pool_global

    interferograms, laser = _pool_global

    return interferograms_regrid(interferograms[i_kid], laser, bins=bins, flatten=True)[0]


@logged
def process_scan(filename, output_dir=Path("."), laser_shift=None, **kwargs):

    try:
        label = Path(filename).name
        process_scan._log.info("Reading Data")
        kd = KissData(filename)
        kd.read_data(
            list_data=[
                "C_motor1_pos",
                "C_laser1_pos",
                "C_laser2_pos",
                "I",
                "Q",
                "A_masq",
            ]
        )
        process_scan._log.info("Calibrating Data")

        calib_kwargs = {
            "calib_func": "kidsdata.kids_calib.get_calfact_3pts",
            "method": ("all", "leastsq"),
            "nfilt": None,
        }
        kd.calib_raw(**calib_kwargs)

        process_scan._log.info("Find laser shift")
        if laser_shift is None:
            laser_shift = kd.find_lasershifts_brute(start=-2, stop=2, num=5, mode="single")
            laser_shifts = kd.find_lasershifts_brute(
                start=laser_shift - 1,
                stop=laser_shift + 1,
                num=201,
                roll_func="kidsdata.utils.roll_fft",
                mode="per_det",
            )
            laser_shift = np.median(laser_shifts)

            out_filename = output_dir / "{}_laser_shift.fits".format(label)
            process_scan._log.info("Saving laser shifts")

            meta = {"filename": kd.filename, "scan": kd.scan, "laser_shift": kd.laser_shift}
            result = Table(
                [
                    Column(kd.list_detector, name="namedet"),
                    Column(laser_shifts, name="laser_shift"),
                ],
                meta=meta,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                result.write(out_filename, overwrite=True)

        kd.laser_shift = laser_shift

        # Check for sign of the interferograms :
        # - flat kidfreq to remove modulation signal
        # - regris each kids onto a common laser position
        # - find the symmetry of the resulting interferograms
        # - get the value at the point (will be the flatfield for dome or pure atmosphere observations)
        # - get the sign of that value

        process_scan._log.info("Search signs of interferograms")

        # Check for flip.... (from kiss_spectroscopy.interferograms)
        A_masq = kd.A_masq

        # Make sure we have no issues with A_masq
        structure = np.zeros((3, 3), np.bool)
        structure[1] = True

        # A_masq has problems when != (0,1,3), binary_closing opening,
        # up to 6 iterations (see scan 800 iint=7)
        A_masq = binary_opening(A_masq * 4, structure, iterations=4)

        # Remove a bit more from A_masq, will also remove some good data : TBC
        A_masq = binary_dilation(A_masq, structure, iterations=2)

        process_scan._log.info("Mask interferograms")

        # Make kidfreq into a masked array (copy data just in case here, should not be needed)
        # TODO: This copy the data...
        interferograms = np.ma.array(
            kd.kidfreq, mask=np.tile(A_masq, kd.ndet).reshape(kd.kidfreq.shape), fill_value=0, copy=True
        )

        # Interferogram have nans sometimes.... CAN NOT proceed with that !!!
        nan_itg = np.isnan(interferograms)
        if np.any(nan_itg):
            interferograms.mask = interferograms.mask | nan_itg

        interferograms = interferograms.reshape(kd.ndet, -1).filled(0)
        laser = kd.laser

        process_scan._log.info("Rebin interferograms")
        bins = "sqrt"
        _, bins = np.histogram(laser.flatten(), bins=bins)
        c_bins = np.mean([bins[1:], bins[:-1]], axis=0)

        worker = partial(_pool_interferograms_regrid, bins=bins)
        with Pool(cpu_count(), initializer=_pool_initializer, initargs=(interferograms, laser.flatten())) as p:
            common_mode_itg = p.map(worker, range(kd.ndet))

        common_mode_itg = np.array(common_mode_itg)

        process_scan._log.info("Polynomial baseline removal of interferograms")
        common_mode_itg[np.isnan(common_mode_itg)] = 0
        deg = 3
        # Remove extra baseline on the common mode interferograms
        p = np.polynomial.polynomial.polyfit(c_bins, common_mode_itg.T, deg=deg)
        baselines = np.polynomial.polynomial.polyval(c_bins, p)
        common_mode_itg -= baselines

        process_scan._log.info("Search symmetry turning point index")
        # Find the center of symmetry for all interferogram by looking for minimum at folded chi2 distribution around the center
        mid_idx = c_bins.shape[0] // 2
        len_idx = c_bins.shape[0] // 4
        idx_range = range(mid_idx - len_idx, mid_idx + len_idx)
        chi2s = [
            np.sum(
                (
                    common_mode_itg[:, central_idx : central_idx - len_idx : -1]
                    - common_mode_itg[:, central_idx : central_idx + len_idx : 1]
                )
                ** 2,
                axis=1,
            )
            for central_idx in idx_range
        ]
        chi2s = np.array(chi2s)

        min_chi2_idx = np.argmin(chi2s, axis=0) + len_idx

        # NO NORMALIZATION !!!
        flat_field_zld = [common_mode_itg[i, idx] for i, idx in enumerate(min_chi2_idx)]

        out_filename = output_dir / "{}_flatfield_sign.fits".format(label)
        process_scan._log.info("Saving zlpd, flat_field, signs")

        meta = {"filename": kd.filename, "scan": kd.scan, "laser_shift": kd.laser_shift}
        result = Table(
            [
                Column(kd.list_detector, name="namedet"),
                Column(c_bins[min_chi2_idx], name="zlp"),
                Column(np.abs(flat_field_zld), name="specFF"),
                Column(np.sign(flat_field_zld), name="sign"),
            ],
            meta=meta,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            result.write(out_filename, overwrite=True)

            return 0
    except (TypeError, ValueError, IndexError, MemoryError, AssertionError) as e:
        process_scan._log.error("An exception occured : {}".format(e))
        return 1


def med_mad(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return med, mad


def process_output(indir=Path("shift_flip"), pattern="*flatfield_sign.fits", sign_threshold=0.9):
    filenames = list(indir.glob(pattern))

    sign_fractions = []
    correct_filenames = []
    bad_filenames = []
    for filename in filenames:
        tab = Table.read(filename)
        tab.convert_bytestring_to_unicode()
        KA = np.char.startswith(tab["namedet"], "KA")
        KB = np.char.startswith(tab["namedet"], "KB")
        sign_fraction = (np.sum(tab[KA]["sign"] == 1) + np.sum(tab[KB]["sign"] == -1)) / len(tab)
        sign_fractions.append(sign_fraction)
        if sign_fraction > sign_threshold:
            correct_filenames.append(filename)
        else:
            bad_filenames.append(filename)

    fig, ax = plt.subplots()
    ax.hist(sign_fractions, bins=100)
    ax.axvline(sign_threshold, linestyle="--", c="r")
    ax.set_xlabel("KA+KB mean interferogram peak sign fraction")
    ax.set_ylabel("# of scan")
    fig.savefig("Itg_signs_hist.png")

    # Fake kidpar for latter plots
    e_kidpar = "e_kidpar_median_all_leastsq.fits"
    e_kidpar = "e_kidpar_median_all_leastsq_2020-10-19T16.fits"
    e_kidpar = Table.read(CALIB_PATH / e_kidpar)
    e_kidpar.convert_bytestring_to_unicode()
    e_kidpar.add_index("namedet")
    from dataclasses import make_dataclass

    fake_kd = make_dataclass("fake_kd", ["kidpar", "list_detector"])
    kd = fake_kd(e_kidpar, e_kidpar["namedet"])
    from kidsdata.kids_plots import plot_geometry

    # Check laser shift PER KIDS
    tabs = []
    for filename in correct_filenames:
        filename = filename.parent / filename.name.replace("flatfield_sign", "laser_shift")
        if filename.exists():
            tab = Table.read(filename)
            tab.convert_bytestring_to_unicode()
            tabs.append(tab)

    if len(tabs) > 0:
        tab_shift = vstack(tabs)
        tab_shift = tab_shift.group_by("namedet")

        laser_shifts = []
        for group, key in zip(tab_shift.groups, tab_shift.groups.keys):
            laser_shifts.append((key[0], *med_mad(group["laser_shift"])))
        laser_shifts = Table(list(zip(*laser_shifts)), names=["namedet", "median_laser_shift", "mad_laser_shift"])

        range_value = np.mean(laser_shifts["median_laser_shift"]) + np.array([-1, 1]) * 2 * np.mean(
            laser_shifts["mad_laser_shift"]
        )

        fig, axes = plt.subplots(ncols=2)
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))

        for array, ax in zip(["KA", "KB"], axes):
            mask = np.chararray.startswith(kd.list_detector, array)
            ikid = np.arange(kd.list_detector.shape[0])
            scatter = plot_geometry(kd, ikid[mask], value=laser_shifts["median_laser_shift"][mask], ax=ax, norm=norm)
            ax.set_title(array)
            cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal")
            cbar.set_label("median laser_shift of all scans")

        fig.savefig("kidpar_lasershift.png")

        laser_shift, mad_laser_shift = med_mad(laser_shifts["median_laser_shift"])
        print(laser_shift, mad_laser_shift)

        fig, ax = plt.subplots()
        ax.hist(laser_shifts["median_laser_shift"], bins=20)
        ax.set_xlabel("laser_shift per kid median over all scans")
        ax.set_ylabel("# of kid")
        ax.axvline(laser_shift, linestyle="--", c="r")
        ax.axvline(laser_shift + mad_laser_shift, linestyle="dotted", c="r")
        ax.axvline(laser_shift - mad_laser_shift, linestyle="dotted", c="r")
        fig.savefig("laser_shifts_per_kid_hist.png")

    # Check ZPDs
    tabs = []
    for filename in correct_filenames:
        tab = Table.read(filename)
        tab.convert_bytestring_to_unicode()
        # Normalize the spectral flat field for each scan
        tab["specFF"] /= np.nanmedian(tab["specFF"])
        tabs.append(tab)

    laser_shifts = [tab.meta["laser_shift"] for tab in tabs]

    if len(np.unique(laser_shifts)) > 1:

        laser_shift, mad_laser_shift = med_mad(laser_shifts)
        print(laser_shift, mad_laser_shift)

        fig, ax = plt.subplots()
        ax.hist(laser_shifts, bins=30)
        ax.set_xlabel("laser_shift per scan median over all kids ")
        ax.set_ylabel("# of scans")
        ax.axvline(laser_shift, linestyle="--", c="r")
        ax.axvline(laser_shift + mad_laser_shift, linestyle="dotted", c="r")
        ax.axvline(laser_shift - mad_laser_shift, linestyle="dotted", c="r")
        fig.savefig("laser_shifts_hist.png")

    # This could have been done with different laser_shift... dangerous !
    assert len(np.unique(laser_shifts)) == 1

    stack_tabs = vstack(tabs)
    stack_tabs = stack_tabs.group_by("namedet")

    zlps = []
    for group, key in zip(stack_tabs.groups, stack_tabs.groups.keys):
        zlps.append((key[0], *med_mad(group["zlp"])))
    zlps = Table(list(zip(*zlps)), names=["namedet", "median_zlp", "mad_zlp"])

    range_value = np.mean(zlps["median_zlp"]) + np.array([-1, 1]) * 2 * np.mean(zlps["mad_zlp"])

    fig, axes = plt.subplots(ncols=2)
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))

    for array, ax in zip(["KA", "KB"], axes):
        mask = np.chararray.startswith(kd.list_detector, array)
        ikid = np.arange(kd.list_detector.shape[0])
        scatter = plot_geometry(kd, ikid[mask], value=zlps["median_zlp"][mask], ax=ax, norm=norm)
        ax.set_title(array)
        cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal")
    cbar.set_label("median ZPD over all scans")

    fig.savefig("kidpar_zpds.png")

    specFFs = []
    for group, key in zip(stack_tabs.groups, stack_tabs.groups.keys):
        specFFs.append((key[0], *med_mad(group["specFF"])))
    specFFs = Table(list(zip(*specFFs)), names=["namedet", "median_specFF", "mad_specFF"])

    range_value = np.mean(specFFs["median_specFF"]) + np.array([-1, 1]) * 5 * np.mean(specFFs["mad_specFF"])

    fig, axes = plt.subplots(ncols=2)
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))

    for array, ax in zip(["KA", "KB"], axes):
        mask = np.chararray.startswith(kd.list_detector, array)
        ikid = np.arange(kd.list_detector.shape[0])
        scatter = plot_geometry(kd, ikid[mask], value=specFFs["median_specFF"][mask], ax=ax, norm=norm)
        ax.set_title(array)
        cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal")
    cbar.set_label("median spectral flatfield over all scans")

    fig.savefig("kidpar_specFF.png")


def check_outfilename(filename, output_dir, template=None):
    label = Path(filename).name
    out_filename = output_dir / template.format(label)
    return out_filename.exists()


def main(args):

    # Slurm magyc...
    i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    n = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    if int(i) == 0:
        logging.info(f"Output directory {args.output_dir} created")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}

    dbs = [list_scan(output=True, obsmode="ITERATIVERASTER", size__gt=100), list_extra(output=True, size__gt=100)]
    filenames = [_["filename"] for db in dbs for _ in db]

    # Remove filename that exists already
    filenames = [
        filename
        for filename in filenames
        if not check_outfilename(filename, args.output_dir, template="{}_flatfield_sign.fits")
    ]

    # Randomize filenames order
    np.random.seed(int(os.getenv("SLURM_JOBID", "42")))
    np.random.shuffle(filenames)

    # Split the work
    _filenames = np.array_split(filenames, n)[i]

    logging.info("{} will do {}".format(i, _filenames))

    if _filenames is None:
        return None

    for filename in _filenames:
        logging.info("{} : {}".format(i, filename))
        process_scan(filename, output_dir=args.output_dir, laser_shift=args.laser_shift, **kwargs)


if __name__ == "__main__":

    # i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

    parser = argparse.ArgumentParser(description="Process all scans.")
    parser.add_argument("--output_dir", type=str, action="store", help="output directory (default: `source`)")
    parser.add_argument(
        "--laser_shift", type=float, action="store", help="given laser shift (default: `None`)", default=None
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir if args.output_dir else "shift_flip")

    main(args)
