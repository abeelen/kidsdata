import logging
import numpy as np
from pathlib import Path


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

from astropy.table import Table, Column

from kidsdata import KissData, list_scan, get_scan
from kidsdata import common_mode as cm
from kidsdata.kids_calib import get_calfact_3pts

from multiprocessing import Pool, cpu_count
from kidsdata.utils import roll_fft
from kidsdata.kids_plots import plot_geometry
from kidsdata.ftsdata import forman


mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

CALIB_PATH = Path("/data/KISS/Calib")

plt.ion()
plot = False


def load_external_kidpar(self, filename):
    """load kidpar produced by the Grenoble pipeline

    Parameters
    ----------
    filename : str
        filename of the kidpar
    """
    extended_kidpar = Table.read(filename)

    # Here the extended_kidpar has for definition :
    # popts = Table(np.array(popts), names=["amplitude", "x0", "y0", "fwhm_x", "fwhm_y", "theta", "offset"])
    # popts.add_column(namedet, 0)
    extended_kidpar.rename_column("name", "namedet")
    extended_kidpar.rename_column("off_az", "x0")
    extended_kidpar.rename_column("off_el", "y0")
    extended_kidpar.rename_column("acqbox", "acqbox_dummy")
    extended_kidpar.add_column(Column(name="fwhm_x", data=np.ones(len(extended_kidpar))))
    extended_kidpar.add_column(Column(name="fwhm_y", data=np.ones(len(extended_kidpar))))
    extended_kidpar.add_column(Column(name="amplitude", data=np.ones(len(extended_kidpar))))

    extended_kidpar.meta["ORIGIN"] = Path(filename).name

    self._extended_kidpar = extended_kidpar


# Load the scan header
logging.info("Reading data")
kd = KissData(get_scan(768))  # Good moon scan
# kd = KissData(get_scan(800)) # Good Jupiter scan

# Read the needed data
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

# 768 : 22 GB -> 12 GB


logging.info("Calibrating data")
# Calibrate from I/Q to TOI in Hz (continuum and interferograms)
kd.calib_raw(calib_func=get_calfact_3pts, method=("all", "leastsq"))

# 768 -> 37 GB

# Load external calibration file : the kid parameter file
# load_external_kidpar(kd, '/data/KISS/Processing/KidsData/kidpos_median.fits')
# OR
kd._extended_kidpar = Table.read(CALIB_PATH / "e_kidpar_median_all_leastsq.fits")
kd._extended_kidpar.meta["ORIGIN"] = "e_kidpar_median_all_leastsq.fits"

###############################################################################
###############################################################################
#### Continuum Checks

kid_mask = kd._kids_selection(std_dev=0.3)
ikid_KA = np.where(kid_mask & np.char.startswith(kd.list_detector, "KA"))[0]
ikid_KB = np.where(kid_mask & np.char.startswith(kd.list_detector, "KB"))[0]
ikid_KAB = np.concatenate([ikid_KA, ikid_KB])

# print(np.take(kd.list_detector, ikid_KA))
# print(np.take(kd.list_detector, ikid_KB))

if plot:
    fig_default, _ = kd.plot_contmap(ikid=[ikid_KA, ikid_KB, ikid_KAB], coord="pdiff", cdelt=0.02)
    fig_median, _ = kd.plot_contmap(
        ikid=[ikid_KA, ikid_KB, ikid_KAB], coord="pdiff", cdelt=0.02, cm_func=cm.median_filtering
    )
    fig_pca1, _ = kd.plot_contmap(
        ikid=[ikid_KA, ikid_KB, ikid_KAB], coord="pdiff", cdelt=0.02, cm_func=cm.pca_filtering, ncomp=1
    )
    fig_pca5, (data, weight, hit) = kd.plot_contmap(
        ikid=[ikid_KA, ikid_KB, ikid_KAB], coord="pdiff", cdelt=0.02, cm_func=cm.pca_filtering, ncomp=5
    )

    for fig in [fig_default, fig_median, fig_pca1, fig_pca5]:
        fig.set_size_inches(10, 5)

    fig_default.savefig("combined_map_default_{}_S{}.png".format(kd._extended_kidpar.meta["ORIGIN"], kd.scan))
    fig_default.savefig("combined_map_median_{}_S{}.png".format(kd._extended_kidpar.meta["ORIGIN"], kd.scan))
    fig_default.savefig("combined_map_pca1_{}_S{}.png".format(kd._extended_kidpar.meta["ORIGIN"], kd.scan))
    fig_default.savefig("combined_map_pca5_{}_S{}.png".format(kd._extended_kidpar.meta["ORIGIN"], kd.scan))


###############################################################################
###############################################################################
#### Spectro

###############################################################################
# Find delay between mirror timeline and interferograms

# Large brute force approach
logging.info("Find laser shift")
laser_shift = kd.find_lasershifts_brute(min_roll=-2, max_roll=2, n_roll=5, mode="single")
# Finer grid
logging.debug("Find laser shift with finer grid")

laser_shifts = kd.find_lasershifts_brute(
    min_roll=laser_shift - 1, max_roll=laser_shift + 1, n_roll=201, roll_func=roll_fft, mode="per_det_int"
)


shifts_per_kid = laser_shifts.mean(axis=1)
shifts_per_kid -= np.median(shifts_per_kid)
KA = np.char.startswith(kd.list_detector, "KA")
KB = np.char.startswith(kd.list_detector, "KB")
range_value = np.array([-3, 3]) * np.std(shifts_per_kid) + np.mean(shifts_per_kid)
norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))
fig, axes = plt.subplots(ncols=2)
for mask, ax in zip([KA, KB], axes):
    scatter = plot_geometry(kd, np.argwhere(mask)[:, 0], ax, value=shifts_per_kid[mask], norm=norm)
    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal")
fig.suptitle(kd.filename)


# For now only single shift are supported
kd.laser_shift = np.median(laser_shifts)

itg_cube_KA = kd.interferograms_cube(ikid_KA, ncomp=1, coord="pdiff", cdelt=(0.02, 0.1))
spec_cube_KA = itg_cube_KA.to_spectra(pcf_apodization=forman, doublesided_apodization=np.hanning)

itg_cube_KB = kd.interferograms_cube(ikid_KB, ncomp=1, coord="pdiff", cdelt=(0.02, 0.1))
spec_cube_KB = itg_cube_KB.to_spectra(pcf_apodization=forman, doublesided_apodization=np.hanning)


# TODO: Plotting routines for cubes ?
# TODO: Slicing ??
