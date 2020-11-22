#!/bin/env python
#SBATCH --job-name=process_iterativeraster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=63GB
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0

# Launch with :
# > sbatch process_iterativeraster.py source_name
# Or
# > sbatch --array=0-19 process_iterativeraster.py source_name
# To parallelize on 20 task (with 3 CPUs and 50 GB each)
# See help for more options
"""
Process interativeraster scans, produce fits maps of photometry
"""

import os
import matplotlib as mpl
from astropy.utils.data import conf

# We should be in non interactive mode...
if int(os.getenv("SLURM_ARRAY_TASK_COUNT", 0)) > 0:
    print("SBATCH ARRAY MODE !!!")
    mpl.use("Agg")
    # To avoid download concurrency,
    # but need to be sure that the cache is updated
    conf.remote_timeout = 120
    conf.download_cache_lock_attempts = 120


# from multiprocessing import Pool

# from astropy.utils import iers
# iers.conf.auto_download = False
# iers.conf.auto_max_age = None

# Do not work
# from astropy.config import set_temp_cache
# set_temp_cache('/tmp')

import os
from itertools import zip_longest, islice, chain
from pathlib import Path
import argparse
import functools
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

# For some reason, this has to be done BEFORE importing anything from kidsdata, otherwise, it does not work...
import logging
from autologging import logged, TRACE

logging.basicConfig(
    level=logging.DEBUG,  # " stream=sys.stdout,
    # format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Prevent verbose output from matplotlib
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


from kidsdata import KissData, get_scan, list_scan
from kidsdata.utils import fit_gaussian


CALIB_PATH = Path("/data/KISS/Calib")

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


@logged
def process_scan(scan, e_kidpar="e_kidpar_median_all_leastsq.fits", calib_kwargs={}, output_dir=Path("."), **kwargs):

    try:
        kd = KissData(get_scan(scan))
        kd.read_data(
            list_data=[
                "A_masq",
                "I",
                "Q",
                "F_tl_Az",
                "F_tl_El",
                "F_state",
                "F_subsc",
                "F_nbsubsc",
                "A_time_ntp",
                "A_time_pps",
                "A_time",
                "A_hours",
                "B_time_ntp",
                "B_time_pps",
                "B_time",
                "B_hours",
                "C_laser1_pos",
                "C_laser2_pos",
            ],
            cache=True,
            array=np.array,
        )
        kd.calib_raw(**calib_kwargs)
        kd._extended_kidpar = Table.read(CALIB_PATH / e_kidpar)

        # Write the file to cache for further use
        # if not kd._cache_filename.exists():
        if True:
            kd._write_data(mode="w")

        # kids selection
        kid_mask = kd._kids_selection(std_dev=0.3)
        ikid_KA = np.where(kid_mask & np.char.startswith(kd.list_detector, "KA"))[0]
        ikid_KB = np.where(kid_mask & np.char.startswith(kd.list_detector, "KB"))[0]
        ikid_KAB = np.concatenate([ikid_KA, ikid_KB])

        ikids = [ikid_KA, ikid_KB, ikid_KAB]
        labels = ["KA", "KB", "KAB"]

        # Combined maps
        fig_pca, results = kd.plot_contmap(ikid=ikids, label=labels, **kwargs)

        ncomp = kwargs.get("ncomp", None)
        calib_method = calib_kwargs.get("calib_func", "kidsdata.kids_calib.get_calfact").split(".")[-1]
        kidpar_id = "_".join(Path(e_kidpar).with_suffix("").name.split("_")[2:])
        file_id = f"coadd_PCA{ncomp}_{kd.source}_{kd.scan}_{calib_method}_{kidpar_id}"

        fig_pca.savefig(output_dir / f"{file_id}.png")
        plt.close(fig_pca)

        # Saving combined maps as fits
        hdus = [fits.PrimaryHDU(None, header=fits.Header(results[0].header))]
        for result, label in zip(results, labels):
            hdus += result.to_hdu(
                hdu_data=f"data_{label}",
                hdu_uncertainty=f"weight_{label}",
                hdu_mask=f"mask_{label}",
                hdu_hits=f"hits_{label}",
            )

        hdus = fits.HDUList(hdus)
        hdus.writeto(output_dir / f"{file_id}.fits", overwrite=True)

        # Laser shift....

        logging.info("Find laser shift")
        laser_shift = kd.find_lasershifts_brute(start=-2, stop=2, num=5, mode="single")
        # Finer grid
        logging.info("Find laser shift with finer grid")
        from kidsdata.utils import roll_fft

        laser_shifts = kd.find_lasershifts_brute(
            start=laser_shift - 1,
            stop=laser_shift + 1,
            num=201,
            roll_func="kidsdata.utils.roll_fft",
            mode="per_det_int",
        )
        with open(output_dir / f"{file_id}_laser_shifts.npy", "wb") as f:
            np.save(f, laser_shifts)

        logging.info(f"{file_id} median laser shift : {np.median(laser_shifts)}")

        from kidsdata.kids_plots import plot_geometry

        shifts_per_kid = laser_shifts.mean(axis=1)
        shifts_per_kid -= np.median(shifts_per_kid)
        KA = np.char.startswith(kd.list_detector, "KA")
        KB = np.char.startswith(kd.list_detector, "KB")
        range_value = np.array([-3, 3]) * np.std(shifts_per_kid) + np.mean(shifts_per_kid)
        norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))
        fig_shifts, axes = plt.subplots(ncols=2)
        for ax, label in zip(axes, ["KA", "KB"]):
            ax.set_title(label)
        for mask, ax in zip([KA, KB], axes):
            scatter = plot_geometry(kd, np.argwhere(mask)[:, 0], ax, value=shifts_per_kid[mask], norm=norm)
            cbar = fig_shifts.colorbar(scatter, ax=ax, orientation="horizontal")
        fig_shifts.suptitle(str(kd.filename) + f"\n median laser shift : {np.median(laser_shifts)}")
        fig_shifts.savefig(output_dir / f"lasershift_{file_id}.png")
        plt.close(fig_shifts)

        return 0
    except (TypeError, ValueError, IndexError) as e:
        print(e)
        return 1


@logged
def fit_gaussian_central_third(data, weight):
    snr = data * np.sqrt(weight)
    shape = snr.shape
    islice = slice(shape[0] // 3, shape[0] * 2 // 3)
    jslice = slice(shape[1] // 3, shape[1] * 2 // 3)
    try:
        popt = fit_gaussian(snr[islice, jslice])
        popt = np.array(popt)
        popt[1:3] += [islice.start, jslice.start]
    except ValueError:
        popt = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    return np.array(popt)


def combine(prefix, output_dir=None, suffix=None):
    if suffix is None:
        suffix = ""
    filenames = Path(output_dir).glob("coadd_{}_*{}.fits".format(prefix, suffix))

    data = []
    weights = []
    exptimes = []
    files = []
    popts = []
    for filename in filenames:
        with fits.open(filename) as hdul:
            header = hdul["data_KAB"].header
            wcs = WCS(header)
            _data, _weight = hdul["data_KAB"].data, hdul["weight_KAB"].data
            # Try to fit on individual maps to select them
            popts.append(fit_gaussian_central_third(_data, _weight))
            data.append(np.ma.array(_data, mask=np.isnan(_data)))
            weights.append(np.ma.array(_weight))
            exptimes.append(header["EXPTIME"])
            files.append(Path(header["FILENAME"]).name)

    # Mask scans where position is off more that 4 sigma
    popts = np.asarray(popts)
    popts -= np.nanmedian(popts, axis=0)
    mad_std = 1.482602218505602 * np.nanmedian(np.abs(popts), axis=0)
    popts_mask = np.abs(popts) > 4 * mad_std
    popts_mask[np.isnan(popts)] = True
    mask = popts_mask[:, 1] | popts_mask[:, 2]

    logging.info("Selecting {} scans out of {}".format(sum(~mask), len(mask)))

    data = [data[index] for index in np.where(~mask)[0]]
    weights = [weights[index] for index in np.where(~mask)[0]]
    exptimes = [exptimes[index] for index in np.where(~mask)[0]]
    files = [files[index] for index in np.where(~mask)[0]]

    source = header["OBJECT"]

    # scan selection
    popts = []
    for _data, _weigth in zip(data, weights):
        popts.append(fit_gaussian(_data * np.sqrt(_weight)))

    popts = np.asarray(popts)
    popts -= np.median(popts, axis=0)
    from astropy.stats import mad_std

    stds = mad_std(popts, axis=0)
    # mask on position offsets
    mask = np.any(np.abs(popts[:, [1, 2]]) > 3 * stds[[1, 2]], axis=1)

    data = [_data for _data, _mask in zip(data, mask) if ~_mask]
    weights = [_weight for _weight, _mask in zip(weights, mask) if ~_mask]

    combined_map, combined_weights = np.ma.average(data, axis=0, weights=weights, returned=True)

    popts = fit_gaussian_central_third(combined_map, combined_weights)

    fwhm_arcmin = popts[3:5] * wcs.wcs.cdelt * 60
    print("FWHM : {:4.2f} x {:4.2f} arcmin".format(*fwhm_arcmin))

    header = wcs.to_header()
    header["OBJECT"] = source
    header["EXPTIME"] = np.sum(exptimes)
    header["UNIT"] = "Uncalibrated Hz"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(combined_map.filled(np.nan), header, name="data"),
            fits.ImageHDU(combined_weights, header, name="weight"),
        ]
    )
    hdul.writeto(Path(output_dir) / "combined_{}_{}.fits".format(prefix, suffix), overwrite=True)

    fig = display_hdu(hdul)
    fig.suptitle("Combined {} on {} scans\nFWHM: {:4.2f}x{:4.2f} arcmin".format(prefix, len(data), *fwhm_arcmin))
    fig.savefig(Path(output_dir) / "combined_{}_{}.png".format(prefix, suffix))


def display_hdu(hdul, plot_data=True, plot_snr=True):

    data = hdul["data"].data
    header = hdul["data"].header
    weight = hdul["weight"].data

    snr = data * np.sqrt(weight)
    # normalization
    snr /= np.nanstd(snr)

    shape = data.shape
    _islice = slice(shape[0] // 3, shape[0] * 2 // 3)
    _jslice = slice(shape[1] // 3, shape[1] * 2 // 3)
    popts = fit_gaussian(snr[_islice, _jslice])
    popts[1:3] += [_islice.start, _jslice.start]

    ncols = int(plot_data + plot_snr)
    to_plots = [_this for _this, to_plot in zip([("Signal", data), ("SNR", snr)], [plot_data, plot_snr]) if to_plot]

    fig, axes = plt.subplots(
        nrows=1, ncols=ncols, squeeze=False, subplot_kw={"projection": WCS(header)}, sharex=True, sharey=True
    )
    for (
        ax,
        (title, to_plot),
    ) in zip(axes[0], to_plots):
        lon = ax.coords[0]
        lon.set_ticks(spacing=1 * u.deg)
        lon.set_ticklabel(exclude_overlapping=True)
        lon.set_coord_type("longitude", 180)

        norm = Normalize(vmin=np.nanmin(to_plot[_islice, _jslice]), vmax=np.nanmax(to_plot[_islice, _jslice]))
        im = ax.imshow(to_plot, origin="lower", norm=norm)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.add_patch(
            Ellipse(
                xy=[popts[1], popts[2]],
                width=popts[3],
                height=popts[4],
                angle=np.degrees(popts[5]),
                edgecolor="r",
                fc="None",
                lw=2,
            )
        )

    return fig


def main(source, cdelt, output_dir):

    # Slurm magyc...
    i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    n = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    if int(i) == 0:
        logging.info(f"Output directory {output_dir} created")
        output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "cm_func": "kidsdata.common_mode.pca_filtering",
        "ncomp": 5,
        "wcs": None,
        "shape": None,
        "coord": "pdiff",
        "cdelt": cdelt,
        "snr": True,
    }

    calib_kwargs = {"calib_func": "kidsdata.kids_calib.get_calfact_3pts", "method": ("all", "leastsq"), "nfilt": None}

    # Produce maps on the same large enough grid
    cdelt = kwargs.get("cdelt")
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("OLON-SFL", "OLAT-SFL")
    wcs.wcs.cdelt = (cdelt, cdelt)
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = (100, 100)
    shape = (200, 200)

    kwargs["wcs"] = wcs
    kwargs["shape"] = shape

    db = list_scan(output=True, source=source, obsmode="ITERATIVERASTER")
    scans = [_["scan"] for _ in db]

    # process = functools.partial(process_scan, **kwargs)
    # with Pool(10) as p:
    #     print(p.map(process, scans))

    _scans = nth(zip(*grouper(scans, n)), i)

    logging.info("{} will do {}".format(i, _scans))

    if _scans is None:
        return None

    for scan in _scans:
        if scan is None:
            continue
        print("{} : {}".format(i, scan))
        #        for e_kidpar in ['e_kidpar_median.fits', 'e_kidpar_median_all_leastsq.fits', 'e_kidpar_median_calib_raw.fits']:
        #            process_scan(scan, e_kidpar=e_kidpar, calib_kwargs=calib_kwargs, output_dir=output_dir, **kwargs)
        process_scan(
            scan,
            # e_kidpar="e_kidpar_median_all_leastsq.fits",
            e_kidpar="e_kidpar_median_all_leastsq_2020-10-19T16.fits",
            calib_kwargs=calib_kwargs,
            output_dir=output_dir,
            **kwargs,
        )


if __name__ == "__main__":

    i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

    parser = argparse.ArgumentParser(description="Process some source.")
    parser.add_argument("source", type=str, action="store", help="name of the source")
    parser.add_argument("--cdelt", type=float, action="store", default=0.02, help="pixel size in deg")
    parser.add_argument("--output_dir", type=str, action="store", help="output directory (default: `source`)")
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir if args.output_dir else args.source)

    main(args.source, args.cdelt, args.output_dir)
