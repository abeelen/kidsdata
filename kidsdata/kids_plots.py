import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter1d as smooth

from astropy.stats import mad_std
from astropy.wcs import WCS

from matplotlib.colors import Normalize


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
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.grid()
    ax.legend()

    fig.tight_layout()

    return fig


# %%
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


# %%


def show_maps(self, ikid=0):
    nrow = 1
    ncol = 1
    fig = plt.figure(figsize=(5 * ncol + 1, 4 * nrow + 0.5))

    subtitle = str(ikid)
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


def show_beammaps(self, datas, wcs, kidpar, pointing_offsets=(0, 0)):
    # Plot all det
    n_kid = len(datas)
    nx = np.ceil(np.sqrt(n_kid)).astype(np.int)
    ny = np.ceil(n_kid / nx).astype(np.int)
    fig_beammap, axes = plt.subplots(nx, ny, sharex=True, sharey=True, subplot_kw={"projection": wcs})
    fig_beammap.set_size_inches(10, 11)
    fig_beammap.subplots_adjust(hspace=0, wspace=0)

    # Plot images
    for _data, ax in zip(datas, axes.flatten()):
        if not np.all(np.isnan(_data)):
            ax.imshow(_data, origin="lower")
        ax.set_aspect("equal")

    # Overplot fit results in pixels coordinates
    x, y = wcs.all_world2pix(pointing_offsets[0] - kidpar["x0"], pointing_offsets[1] - kidpar["y0"], 0)
    fwhm_x, fwhm_y = kidpar["fwhm_x"] / wcs.wcs.cdelt[0], kidpar["fwhm_y"] / wcs.wcs.cdelt[1]

    for pos_x, pos_y, fwhm_x, fwhm_y, theta, ax in zip(x, y, fwhm_x, fwhm_y, kidpar["theta"], axes.flatten()):
        if pos_x is not np.nan:
            ax.add_patch(
                Ellipse(
                    xy=[pos_x, pos_y],
                    width=fwhm_x,
                    height=fwhm_y,
                    angle=np.degrees(theta),
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


def show_contmap(self, data, weights, hits, label=None, snr=False):

    # TODO: assert same header....
    fig, axes = plt.subplots(
        ncols=len(data),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": WCS(data[0].header)},
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    if isinstance(axes, plt.Axes):
        axes = [axes]

    if snr:
        datas = [_data.data * np.sqrt(_weight.data) for _data, _weight in zip(data, weights)]
        datas = [_snr / np.nanstd(_snr) for _snr in datas]  # Proper weight normalization
    else:
        datas = [_data.data for _data in data]

    norm = Normalize(vmin=np.nanmean(datas) - 3 * np.nanstd(datas), vmax=np.nanmean(datas) + 3 * np.nanstd(datas))

    cdelt = data[0].header["CDELT1"], data[0].header["CDELT2"]

    for ax, _data in zip(axes, datas):
        im = ax.imshow(_data, origin="lower", norm=norm)
        ax.set_aspect("equal")

        if self.source == "Moon":
            ax.add_patch(
                Ellipse(
                    xy=(0, 0),
                    width=31 / 60 / cdelt[0],
                    height=31 / 60 / cdelt[1],
                    angle=0,
                    edgecolor="r",
                    fc="None",
                    lw=2,
                    alpha=0.5,
                )
            )

    for ax in axes[1:]:
        lon = ax.coords[1]
        lon.set_ticklabel_visible(False)
        lon.set_ticks_visible(False)
        lon.set_axislabel("")

    if label is not None:
        for ax, _label in zip(axes, label):
            ax.set_title(_label)

    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(self.filename)
    return fig


def plot_geometry(self, ikid, ax, value=None, **kwargs):
    x0, y0 = self.kidpar.loc[self.list_detector[ikid]]["x0"], self.kidpar.loc[self.list_detector[ikid]]["y0"]
    scatter = ax.scatter(x0, y0, c=value, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("lon offset [arcmin]")
    ax.set_ylabel("lat offset [arcmin]")

    return scatter


def show_kidpar(self, ikid=None, show_beam=True, ranges=[None, None, None], bins=[30, 30, 30], **kwargs):
    # Geometry

    if ikid is None:
        ikid = np.arange(len(self.list_detector))
    else:
        ikid = np.asarray(ikid)

    popt = self.kidpar.loc[self.list_detector[ikid]]

    acqboxes = np.unique(popt["acqbox"])
    figs = []

    # Loop over acquisition box
    for acqbox in acqboxes:
        mask_box = popt["acqbox"] == acqbox

        _ikid = ikid[mask_box]
        _sizes = np.array([popt["fwhm_x"][mask_box], popt["fwhm_y"][mask_box]]).T * 60
        _amplitudes = np.array(popt["amplitude"][mask_box])

        values = {
            "fwhms [arcmin]": np.max(np.abs(_sizes), axis=1),  # Largest fwhms
            "ellipticities": (np.max(np.abs(_sizes), axis=1) - np.min(np.abs(_sizes), axis=1))
            / np.max(np.abs(_sizes), axis=1),
            "amplitudes [rel. abu]": _amplitudes,
        }

        fig, axes = plt.subplots(2, 3, **kwargs)

        for (item, value), ax_top, ax_bottom, range_value, bins_value in zip(
            values.items(), axes[0], axes[1], ranges, bins
        ):

            if range_value is None:
                mean_value = np.nanmedian(value)
                std_value = mad_std(value, ignore_nan=True)
                range_value = np.array([-3, 3]) * std_value + mean_value

            ax_bottom.hist(value[~np.isnan(value)], range=range_value, bins=bins_value)
            norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))
            scatter = plot_geometry(self, _ikid, ax_top, value=value, norm=norm)
            cbar = fig.colorbar(scatter, ax=ax_top, orientation="horizontal")
            cbar.set_label(item)
            ax_top.set_xlim(-0.62 * 60, 0.62 * 60)
            ax_top.set_ylim(-0.62 * 60, 0.62 * 60)
        fig.suptitle("{} / aqbox: {}".format(self.filename, acqbox))
        fig.tight_layout()

        figs.append(fig)

    return figs


def show_kidpar_fwhm(self):

    sizes = (
        np.array([self.kidpar.loc[self.list_detector]["fwhm_x"], self.kidpar.loc[self.list_detector]["fwhm_y"]]).T * 60
    )  # arcmin
    fig, ax = plt.subplots()
    for _sizes, label in zip(sizes.T, ["major", "minor"]):
        ax.hist(np.abs(_sizes[~np.isnan(_sizes)]), label=label, alpha=0.5, range=(0, 40), bins=50)
    ax.legend()
    ax.set_xlabel("FWHM [arcmin]")
    fig.suptitle(self.filename)
    return fig
