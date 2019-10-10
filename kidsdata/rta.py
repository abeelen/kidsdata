import logging

logging.basicConfig(level=logging.INFO)

from pathlib import Path
import matplotlib.pyplot as plt
from . kiss_data import KissRawData
from  . db import list_scan
import numpy as np
import warnings

from scipy import optimize
from matplotlib.patches import Ellipse

import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import join

from scipy.interpolate import interp1d

plt.ion()


def beammap(scan=None, kd=None, array="B"):

    # Open file
    if kd is None:
        kd = KissRawData(get_scan(scan))
        list_data = " ".join(kd.names.ComputedDataSc + kd.names.ComputedDataUc + ["I", "Q"])
        list_detector = kd.get_list_detector(array, flag=0)

        # Read data
        kd.read_data(list_data=list_data, list_detector=list_detector, silent=True)

    # Compute & plot beammap
    fig_beammap, datas, wcs, popts = kd.plot_beammap(coord="pdiff")

    # Update kidpar
    for key in ["x0", "y0"]:
        popts[key] -= np.nanmedian(popts[key])
    kd._extended_kidpar = popts

    # plot geometry
    fig_geometry, fwhm = kd.plot_kidpar()

    # select good detector, ie within 60 arcmin of the center and fwhm 25 +- 10
    pos = np.array([kd.kidpar[kd.list_detector]["x0"], kd.kidpar[kd.list_detector]["y0"]]) * 60  # arcmin
    fwhm = np.array(np.abs(kd.kidpar[kd.list_detector]["fwhm_x"]) + np.abs(kd.kidpar[kd.list_detector]["fwhm_y"])) / 2 * 60
    ikid = np.where((np.sqrt(pos[0] ** 2 + pos[1] ** 2) < 60) & (np.abs(fwhm - 25) < 10))[0]

    data, weight, hits = kd.continuum_map(coord="pdiff", ikid=ikid, cdelt=0.05)

    return kd, (fig_beammap, fig_geometry)


def check_pointing(scan=None, kd=None):

    if kd is None:
        # Open file
        kd = KissRawData(get_scan(scan))
        list_data = " ".join(kd.names.ComputedDataSc + kd.names.ComputedDataUc)

        # Read daa
        kd.read_data(list_data=list_data, silent=True)

    fig_pointing, ax = plt.subplots()
    ax.plot(kd.F_sky_Az, kd.F_sky_El)
    obstime = kd.obstime
    interp_az, interp_el = kd.get_object_altaz(npoints=100)
    ax.plot(interp_az(obstime.mjd), interp_el(obstime.mjd), label=kd.source)
    ax.legend(loc="best")
    fig_pointing.suptitle(kd.filename)

    return kd, fig_pointing
