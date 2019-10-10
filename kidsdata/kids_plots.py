import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter1d as smooth
from astropy.stats import mad_std


def calibPlot(self, ikid=0):
    """ Plot Icc, Qcc, calfact, and kidfreq distributions for ikid detector;
        show median calfact for all detectors in the last panel.
    """
    fig = plt.figure(figsize=(5 * 3, 4 * 2))

    ax = plt.subplot(2, 3, 1)
    ax.plot(self.Icc[ikid, :], label="Original")
    ax.plot(smooth(self.Icc[ikid, :], 21), label="Smoothed")
    ax.grid()
    ax.set_ylabel("I circle center [arbitrary units]")
    ax.set_xlabel("Sample Number")
    ax.legend()

    ax = plt.subplot(2, 3, 2)
    ax.plot(self.Qcc[ikid, :], label="Original")
    ax.plot(smooth(self.Qcc[ikid, :], 21), label="Smoothed")
    ax.grid()
    ax.set_ylabel("Q circle center [arbitrary units]")
    ax.set_xlabel("Sample Number")
    ax.legend()

    ax = plt.subplot(2, 3, 3)
    ax.plot(self.calfact[ikid, :], label="Original")
    ax.plot(smooth(self.calfact[ikid, :], 21), label="Smoothed")
    ax.grid()
    ax.set_ylabel("Calibration Factor [Hz/rad]")
    ax.set_xlabel("Sample Number")
    ax.legend()

    ax = plt.subplot(2, 3, 4)
    ax.plot(self.kidfreq[ikid, 4:12].ravel(), label="Detector:" + self.kidpar["namedet"][ikid])
    ax.grid()
    ax.set_ylabel("Signal [Hz]")
    ax.set_xlabel("Sample Number")
    ax.legend()

    ax = plt.subplot(2, 3, 5)
    ax.plot(np.median(self.calfact, axis=1), label="Original")
    ax.plot(medfilt(np.median(self.calfact, axis=1), 5), label="Fitted")
    ax.grid()
    ax.set_ylabel("Median Calibration Factor [Hz/rad]")
    ax.set_xlabel("Detector Number")
    ax.legend()

    fig.suptitle(self.filename)
    fig.tight_layout()

    return fig


def checkPointing(self):
    """ Plot:
        1. Azimuth distribution of samples.
        2. Elevation distribuiton of samples.
        3. 2D distribution of (Elevation, Azimuth) for samples.
        4. Medians of poitnings for each interferogram,
        compared with pointing models.
    """
    fig = plt.figure(figsize=(5 * 2 + 1, 4 * 2))
    fig.suptitle(self.filename)

    mask_pointing = self.mask_pointing
    az_tel, el_tel, mask_tel = self.F_tl_Az, self.F_tel_El, self.mask_tel
    az_sky, el_sky = self.F_sky_Az, self.F_sky_El
    az_skyQ1, el_skyQ1 = self.F_skyQ1_Az, self.F_skyQ1_El

    if hasattr(self, "F_azimuth") & hasattr(self, "F_elevation"):
        azimuth, elevation = self.F_azimuth, self.F_Elevation
        ax = plt.subplot(2, 2, 1)
        ax.plot(azimuth[mask_pointing])
        ax.set_ylabel("Azimuth [deg]")
        ax.set_xlabel("Sample number [dummy units]")
        ax.grid()

        ax = plt.subplot(2, 2, 2)
        ax.plot(elevation[mask_pointing])
        ax.set_xlabel("Sample number [dummy units]")
        ax.set_ylabel("Elevation [deg]")
        ax.grid()

        ax = plt.subplot(2, 2, 3)
        ax.plot(azimuth[mask_pointing], elevation[mask_pointing], ".")
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Elevation [deg]")
        ax.set_title("Pointing")
        ax.grid()

    ax = plt.subplot(2, 2, 4)
    plt.plot(az_tel[mask_tel], el_tel[mask_tel], "+", ms=12, label="Telescope")
    ax.plot(az_sky[mask_tel], el_sky[mask_tel], "+", ms=12, label="Sky")
    ax.plot(az_skyQ1[mask_tel], el_skyQ1[mask_tel], "+", ms=12, label="Sky Q1")
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.grid()
    ax.legend()

    fig.tight_layout()

    return fig


#%%
def photometry(self):
    fig = plt.figure(figsize=(5 * 2 + 1, 4 * 2 + 0.5))
    fig.suptitle(self.filename)

    bgrd = self.continuum
    meds = np.median(bgrd, axis=1)
    stds = np.std(bgrd, axis=1)

    ax = plt.subplot(2, 2, 1)
    ax.semilogy(meds)
    ax.set_xlabel("Detector Number")
    ax.set_ylabel("Median of Photometry")

    ax = plt.subplot(2, 2, 2)
    ax.semilogy(stds)
    ax.set_xlabel("Detector Number")
    ax.set_ylabel("STD of Photometry")

    fig.tight_layout()

    return fig


#%%


def show_maps(self, ikid=0):
    nrow = 1
    ncol = 1
    fig = plt.figure(figsize=(5 * ncol + 1, 4 * nrow + 0.5))

    subtitle = str(testikid)
    fig.suptitle(self.filename + subtitle)

    ax = plt.subplot(ncol, nrow, 1)
    ax.imshow(self.beammap)

    #    wcs = self.beamwcs
    #    ax = plt.subplot(ncol, nrow, 1, projection=wcs)
    #    bgrs = self.bgrs[153, self.mask_tel]
    #    ax.imshow(bgrs)

    ax.grid(color="white", ls="solid")
    #    ax.set_xlabel('Az [deg]')
    #    ax.set_ylabel('El [deg]')

    return fig


def show_beammaps(self, datas, wcs, popts):
    # Plot all det
    nx = np.ceil(np.sqrt(self.ndet)).astype(np.int)
    ny = np.ceil(self.ndet / nx).astype(np.int)
    fig_beammap, axes = plt.subplots(nx, ny, sharex=True, sharey=True, subplot_kw={"projection": wcs})
    fig_beammap.set_size_inches(10, 11)
    fig_beammap.subplots_adjust(hspace=0, wspace=0)
    for _data, popt, ax in zip(datas, popts, axes.flatten()):
        ax.imshow(_data, origin="lower")
        ax.set_aspect("equal")
        if popt is not None:
            ax.add_patch(
                Ellipse(
                    xy=[popt["x0"] / wcs.wcs.cdelt[0], popt["y0"] / wcs.wcs.cdelt[1]],
                    width=popt["fwhm_x"] / wcs.wcs.cdelt[0],
                    height=popt["fwhm_y"] / wcs.wcs.cdelt[1],
                    angle=np.degrees(popt["theta"]),
                    edgecolor="r",
                    fc="None",
                    lw=2,
                )
            )

    for ax in axes.ravel():
        if ax.images == []:
            ax.set_axis_off()
            if getattr(ax, "coords"):
                ax.coords[0].set_ticklabel_visible(False)
                ax.coords[0].set_ticks_visible(False)
                ax.coords[0].set_axislabel("")
                ax.coords[1].set_ticklabel_visible(False)
                ax.coords[1].set_ticks_visible(False)
                ax.coords[1].set_axislabel("")
                ax.set_visible(False)

    for ax in axes[0:-1, 1:].ravel():
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

    for ax in axes[:-1, 0:].ravel():
        ax.coords[0].set_ticklabel_visible(False)

    for ax in axes[-1, 1:].ravel():
        ax.coords[1].set_ticklabel_visible(False)

    fig_beammap.suptitle(self.filename)

    return fig_beammap


def show_kidpar(self, show_beam=True):
    # Geometry
    popt = self.kidpar.loc[self.list_detector]
    pos = np.array([popt["x0"], popt["y0"]]).T * 60  # arcmin
    sizes = np.array([popt["fwhm_x"], popt["fwhm_y"]]).T * 60

    values = {
        "fwhms [arcmin]": np.max(np.abs(sizes), axis=1),
        "ellipticities": (np.max(np.abs(sizes), axis=1) - np.min(np.abs(sizes), axis=1))
        / np.max(np.abs(sizes), axis=1),
        "amplitudes [rel. abu]": np.array(popt["amplitude"]) / np.nanmedian(popt["amplitude"]),
    }

    fig, axes = plt.subplots(2, 3)
    for (item, value), ax_top, ax_bottom in zip(values.items(), axes[0], axes[1]):
        mean_value = np.nanmedian(value)
        std_value = mad_std(value, ignore_nan=True)
        range_value = np.array([-3, 3]) * std_value + mean_value
        ax_bottom.hist(value, range=range_value)

        scatter = ax_top.scatter(pos[:, 0], pos[:, 1], c=np.clip(value, *range_value))
        cbar = fig.colorbar(scatter, ax=ax_top, orientation="horizontal")
        cbar.set_label(item)
        ax_top.set_xlim(-0.62 * 60, 0.62 * 60)
        ax_top.set_ylim(-0.62 * 60, 0.62 * 60)
        ax_top.set_aspect("equal")
        ax_top.set_xlabel("lon offset [arcmin]")
        ax_top.set_ylabel("lat offset [arcmin]")
    fig.suptitle(self.filename)
    fig.tight_layout()

    return fig


def show_kidpar_fwhm(self):

    sizes = (
        np.array([self.kidpar.loc[self.list_detector]["fwhm_x"], self.kidpar.loc[self.list_detector]["fwhm_y"]]).T * 60
    )  # arcmin
    fig, ax = plt.subplots()
    for _sizes, label in zip(sizes.T, ["major", "minor"]):
        ax.hist(np.abs(_sizes), label=label, alpha=0.5, range=(0, 40), bins=50)
    ax.legend()
    ax.set_xlabel("FWHM [arcmin]")
    fig.suptitle(self.filename)
    return fig
