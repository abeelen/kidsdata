import warnings
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter1d as smooth

import astropy.units as u
from astropy.stats import mad_std
from astropy.wcs import WCS

from matplotlib.colors import Normalize
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty


def calibPlot(self, ikid=0):
    """Plot Icc, Qcc, calfact, and interferograms distributions for ikid detector;
    show median calfact for all detectors in the last panel.
    """
    warnings.warn("This needs to be updated", DeprecationWarning)
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
    ax.plot(self.interferograms[ikid, 4:12].ravel(), label="Detector:" + self.kidpar["namedet"][ikid])
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
    """Plot:
    1. Azimuth distribution of samples.
    2. Elevation distribuiton of samples.
    3. 2D distribution of (Elevation, Azimuth) for samples.
    4. Medians of poitnings for each interferogram,
    compared with pointing models.
    """
    warnings.warn("This needs to be updated", DeprecationWarning)
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
    plt.plot(az_tel[~mask_tel], el_tel[~mask_tel], "+", ms=12, label="Telescope")
    ax.plot(az_sky[~mask_tel], el_sky[~mask_tel], "+", ms=12, label="Sky")
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.grid()
    ax.legend()

    fig.tight_layout()

    return fig


# %%
def photometry(self):
    warnings.warn("This needs to be updated", DeprecationWarning)

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


def show_beammaps(self, datas, wcs, kidpar, pointing_offsets=(0, 0), show_fit=True):
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

    if show_fit:
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


def quickshow_beammaps(self, ikids, datas, wcs, kidpar, pointing_offsets=(0, 0), show_fit=True):
    """WIP..."""

    marging = 1
    npix_y, npix_x = datas[0].shape
    n_pages = 1

    _data, (ncols, nrows) = multi_im(datas, aspect_ratio=16 / 9, marging=marging, n_pages=n_pages, norm=None)

    i_page = 0

    # Plot all det
    # n_kid = len(datas)
    fig_beammap, ax = plt.subplots()
    ax.imshow(_data)
    ax.set_aspect("equal")

    # Overplot fit results in pixels coordinates
    x, y = wcs.all_world2pix(pointing_offsets[0] - kidpar["x0"], pointing_offsets[1] - kidpar["y0"], 0)
    fwhm_x, fwhm_y = kidpar["fwhm_x"] / wcs.wcs.cdelt[0], kidpar["fwhm_y"] / wcs.wcs.cdelt[1]

    page_slice = slice(i_page * (nrows * ncols), (i_page + 1) * (nrows * ncols))

    for i, label in enumerate(self.list_detector[ikids][page_slice]):
        offset_x = i % ncols
        offset_y = i // ncols
        ax.annotate(
            label, (offset_x * (npix_x + marging) + npix_x / 4, offset_y * (npix_y + marging) + npix_y / 4), c="r"
        )

    if show_fit:
        for i, (pos_x, pos_y, fwhm_x, fwhm_y, theta) in enumerate(
            zip(x[page_slice], y[page_slice], fwhm_x[page_slice], fwhm_y[page_slice], kidpar["theta"][page_slice])
        ):
            offset_x = i % ncols
            offset_y = i // ncols
            if pos_x is not np.nan:
                ax.add_patch(
                    Ellipse(
                        xy=[pos_x + offset_x * (npix_x + marging), pos_y + offset_y * (npix_y + marging)],
                        width=fwhm_x,
                        height=fwhm_y,
                        angle=np.degrees(theta),
                        edgecolor="r",
                        fc="None",
                        lw=2,
                    )
                )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(ncols) * (npix_x + marging))
    ax.set_yticks(np.arange(nrows) * (npix_y + marging))

    fig_beammap.suptitle(self.filename)

    return fig_beammap


def show_contmap(data, label=None, snr=False, norm=None):

    if not isinstance(data, list):
        data = [data]

    # TODO: assert same header....
    fig, axes = plt.subplots(
        ncols=len(data),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": data[0].wcs},
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    if isinstance(axes, plt.Axes):
        axes = [axes]

    if snr:
        datas_to_plot = [
            _data.data * np.sqrt(_data.uncertainty.array)
            if isinstance(_data.uncertainty, InverseVariance)
            else _data.data / _data.uncertainty.array
            if isinstance(_data.uncertainty, StdDevUncertainty)
            else _data.data / np.sqrt(_data.uncertainty.array)
            if isinstance(_data.uncertainty, VarianceUncertainty)
            else None
            for _data in data
        ]
        datas_to_plot = [_snr / np.nanstd(_snr) for _snr in datas_to_plot]  # Proper weight normalization
    else:
        datas_to_plot = [_data.data for _data in data]

    if norm is None:
        norm = Normalize(
            vmin=np.nanmean(datas_to_plot) - 3 * np.nanstd(datas_to_plot),
            vmax=np.nanmean(datas_to_plot) + 3 * np.nanstd(datas_to_plot),
        )

    for ax, _data in zip(axes, datas_to_plot):
        im = ax.imshow(_data, origin="lower", norm=norm)
        ax.set_aspect("equal")

        """
        if self.source == "Moon":
            cdelt = np.diag(_data.wcs.pixel_scale_matrix)
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
        """

    for ax in axes[1:]:
        lat = ax.coords[1]
        lat.set_ticklabel_visible(False)
        lat.set_ticks_visible(False)
        lat.set_axislabel("")

    for ax in axes:
        lon = ax.coords[0]
        lon.set_ticklabel(exclude_overlapping=True)
        lon.set_coord_type("longitude", 180)

    if label is not None:
        for ax, _label in zip(axes, label):
            ax.set_title(_label)

    fig.colorbar(im, ax=axes, shrink=0.6)

    return fig


def plot_geometry(self, ikid, ax, value=None, **kwargs):
    x0, y0 = [self.kidpar.loc[self.list_detector[ikid]][item].to(u.arcmin).value for item in ["x0", "y0"]]
    scatter = ax.scatter(x0, y0, c=value, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("lon offset [arcmin]")
    ax.set_ylabel("lat offset [arcmin]")

    return scatter


def default_range(value, threshold_limits=[-3, 3]):
    mean_value = np.nanmedian(value)
    std_value = mad_std(value, ignore_nan=True)
    return np.array(threshold_limits) * std_value + mean_value  # Default to [-3 3] sigma


def show_kidpar(
    self,
    ikid=None,
    to_plot=["fwhms", "eccentricities", "amplitudes"],
    plot_hist=True,
    ranges=None,
    bins=None,
    limits=[-0.62, 0.62],
    group_key="acqbox",
    group_list=None,
    namedet=False,
    **kwargs
):
    """Display a kidpar geometry.

    Parameters
    ----------

    self : ~kidsdata
        a kidsdata object to retrieve the `kidpar` and `list_detector` properties
    ikid : array_like, optionnal
        to plot a subsample of the `list_detector` list
    to_plot : list of str
        list of item to plot within 'fwhms', 'major_axis, 'minor_axis', 'mean_fwhms', eccentricities', 'amplitudes', None
    plot_hist: bool
        do we plot the histograms, default True
    ranges : dict, optionnal
        display ranges for the histograms as {item: [min, max]} with item within the to_plot list
    bins : dict, optionnal
        bins of the histogram as {item: bins} with item within the to_plot list
    limits : array_like
        limits in x & y in degree for the geometry,
    group_key : str
        the kidpar key to use to identify arrays, default 'acqbox',
    group_list : dict, optionnal
        to regroup the group_key as {label: [group_key_values]}, by default each group_key will be plotted separately
    namedet : bool
        to overplot the name of the kids

    Returns
    -------
    figs : list of :matplotlib.Figure:
        the resulting figures
    """
    if ikid is None:
        ikid = np.arange(len(self.list_detector))
    else:
        ikid = np.asarray(ikid)

    if ranges is None:
        ranges = {}
    if bins is None:
        bins = {}

    popt = self.kidpar.loc[self.list_detector[ikid]]

    if group_list is None:
        group_list = {"{} {}".format(group_key, group): [group] for group in np.unique(popt[group_key])}

    fwhms = np.abs([popt["fwhm_x"], popt["fwhm_y"]]).T * 60  # fwhm in arcmin

    plotting_values = {
        "fwhms": {"value": np.nanmax(fwhms, axis=1), "label": "fwhms [arcmin]"},  # fwhm in arcmin
        "major_axis": {"value": np.nanmax(fwhms, axis=1), "label": "major axis [arcmin]"},  # fwhm in arcmin
        "minor_axis": {"value": np.nanmin(fwhms, axis=1), "label": "minor axis [arcmin]"},  # fwhm in arcmin
        "mean_fwhms": {"value": np.nanmean(fwhms, axis=1), "label": "mean fwhms [arcmin]"},  # fwhm in arcmin
        "eccentricities": {
            "value": np.sqrt(1 - np.min(fwhms, axis=1) ** 2 / np.max(fwhms, axis=1) ** 2),
            "label": "eccentricities",
        },
        "amplitudes": {"value": np.array(popt["amplitude"]), "label": "amplitudes [rel.abu]"},
        None: {"value": np.ones(len(popt)), "label": ""},
    }

    for key in plotting_values:
        if key is None:
            plotting_values[key]["range"] = [1, 1]
            plotting_values[key]["bins"] = 1
            continue
        plotting_values[key]["range"] = ranges.get(key) or default_range(plotting_values[key]["value"])
        plotting_values[key]["bins"] = bins.get(key) or 30

    figs = []

    # Loop over grouping
    for group_label, group in group_list.items():
        mask_box = [popt[group_key] == item for item in group]
        mask_box = np.bitwise_or.reduce(mask_box, axis=0)

        if np.all(~mask_box):
            # No kid in the group
            continue

        _ikid = ikid[mask_box]

        if plot_hist:
            gs_kw = {"width_ratios": np.ones(len(to_plot)), "height_ratios": [1, 0.05, 0.3]}
        else:
            gs_kw = {"width_ratios": np.ones(len(to_plot)), "height_ratios": [1, 0.05]}

        fig, axes = plt.subplots(
            len(gs_kw["height_ratios"]),
            len(gs_kw["width_ratios"]),
            squeeze=False,
            constrained_layout=True,
            gridspec_kw=gs_kw,
            **kwargs
        )

        # Share x&y for the top row:
        target = axes[0, 0]
        for ax in axes[0, 1:]:
            ax._shared_x_axes.join(target, ax)
            ax._shared_y_axes.join(target, ax)

        for key_plot, ax_top, ax_color in zip(to_plot, axes[0], axes[1]):
            values = plotting_values[key_plot]["value"][mask_box]
            range_value = plotting_values[key_plot]["range"]
            bins_value = plotting_values[key_plot]["bins"]
            label = plotting_values[key_plot]["label"]

            norm = Normalize(vmin=np.min(range_value), vmax=np.max(range_value))
            scatter = plot_geometry(self, _ikid, ax_top, value=values, norm=norm)

            if namedet:
                x0, y0 = [self.kidpar.loc[self.list_detector[_ikid]][item].to(u.arcmin).value for item in ["x0", "y0"]]
                names = self.kidpar.loc[self.list_detector[_ikid]]["namedet"]
                for x, y, name, value in zip(x0, y0, names, values):
                    ax_top.text(x, y, name, clip_on=True, fontsize="xx-small", c=scatter.cmap(norm(value)))

            if key_plot is not None:
                cbar = fig.colorbar(scatter, ax=ax_top, cax=ax_color, orientation="horizontal")
                cbar.set_label(label)

            ax_top.set_xlim(np.array(limits) * 60)
            ax_top.set_ylim(np.array(limits) * 60)

        if plot_hist:
            for key_plot, ax_bottom in zip(to_plot, axes[2]):
                if key_plot is None:
                    ax_bottom.remove()
                else:
                    values = plotting_values[key_plot]["value"][mask_box]
                    range_value = plotting_values[key_plot]["range"]
                    bins_value = plotting_values[key_plot]["bins"]
                    ax_bottom.hist(values[~np.isnan(values)], range=range_value, bins=bins_value)

        fig.suptitle("{} / {}".format(self.filename.name, group_label))
        fig.tight_layout()

        figs.append(fig)

    return figs


def show_kidpar_fwhm(self):

    sizes = np.array([self.kidpar.loc[self.list_detector][item].to(u.arcmin) for item in ["fwhm_x", "fwhm_y"]])
    fig, ax = plt.subplots()
    for _sizes, label in zip(sizes, ["major", "minor"]):
        ax.hist(np.abs(_sizes[~np.isnan(_sizes)]), label=label, alpha=0.5, range=(0, 40), bins=50)
    ax.legend()
    ax.set_xlabel("FWHM [arcmin]")
    fig.suptitle(self.filename)
    return fig


def multi_im(xs, aspect_ratio=1, marging=1, n_pages=1, norm=None):
    """Display list of images as an combined image.

    Parameters
    ----------
    X : list of 2D array_like or 3D array-like
        The images data :

        - list of (M,N) images
        - (n, M, N): 3D images
    aspect_ratio : float
        the overall image aspect ratio, by default 1.
    marging: int
        number of pixel to left blank between images, by default 1.
    n_pages: int
        number of page to produce, by default 1.
    norm : `~matplotlib.colors.Normalize`, optional
        The `.Normalize` instance used to scale scalar data to the [0, 1]
        range before mapping to colors using *cmap*. By default, a linear
        scaling mapping the lowest value to 0 and the highest to 1 is used.
        This parameter is ignored for RGB(A) data.

    Returns
    -------
    images : 2D or 3D array_like
        bigger mosaic of images, first index is the page number
    (ncols, rows) : ints
        the number of columns and rows per page

    Notes
    -----
    if the norm keyword is set, if will be used globally, by default, each images is normed individually
    """

    xs = np.asarray(xs)
    n, M, N = xs.shape

    ncols = np.ceil(np.sqrt(n / n_pages * aspect_ratio)).astype(int)
    nrows = np.ceil(n / n_pages / ncols).astype(int)

    image_width = ncols * N + (ncols - 1) * marging
    image_height = nrows * M + (nrows - 1) * marging

    pixels = np.full((n_pages, image_height, image_width), np.nan)
    for page in range(n_pages):
        for _sub, (j, i) in zip(
            xs[page * ncols * nrows : (page + 1) * ncols * nrows], product(range(nrows), range(ncols))
        ):
            if norm is None:
                _sub = Normalize(vmin=np.nanmin(_sub), vmax=np.nanmax(_sub))(_sub)
            else:
                _sub = norm(_sub)
            pixels[page, j * (M + marging) : j * (M + marging) + M, i * (N + marging) : i * (N + marging) + N] = _sub

    return np.squeeze(pixels), (ncols, nrows)


def plot_psd(psds, freq, list_detector, square=True, xmin=None, xmax=None, ymax=None, ymin=None, **kwargs):

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([0.04, 0.15, 0.4, 0.79])
    ax2 = fig.add_axes([0.59, 0.15, 0.4, 0.79])
    cax = fig.add_axes([0.45, 0.15, 0.006, 0.79])

    if square is not True:
        psds = np.sqrt(psds)
        ax2.set_ylabel(r"PSD [Hz/ $\sqrt{Hz}$]", labelpad=-2)
    else:
        ax2.set_ylabel(r"PSD [$Hz^{2}/Hz$]", labelpad=-2)

    color_sequence = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]

    im = ax1.imshow(np.log10(psds), aspect="auto", extent=(0.2, freq.max(), psds.shape[0], 0), vmin=ymin, vmax=ymax)
    boxes = set(namedet[0:2] for namedet in list_detector)
    boxes = np.sort(list(boxes))
    ticks = []
    ticklabels = []
    for box in boxes:
        mask_box = np.char.startswith(list_detector, box)
        _min, _max = np.where(mask_box)[0].min(), np.where(mask_box)[0].max()
        ticks.append((_min + _max) / 2)
        ticklabels.append(box)

    ax1.yaxis.set_ticks(ticks)
    ax1.yaxis.set_ticklabels(ticklabels)
    ax1.set_xlabel("Freq [Hz]")
    cb = plt.colorbar(im, ax=ax1, cax=cax)
    cb.set_label("log10(PSD)")
    ax1.set_xscale("log")

    boxes = set(namedet[0:2] for namedet in list_detector)
    for box in boxes:
        mask_box = np.char.startswith(list_detector, box)
        ax2.plot(freq, np.median(psds[mask_box], axis=0), color=color_sequence[list(boxes).index(box)], label=box)

    if ymin is None and ymax is None:
        pass
    else:
        im.set_norm(Normalize(np.log10(ymin), np.log10(ymax)))
        ax2.set_ylim(ymin, ymax)

    if xmin is None and xmax is None:
        pass
    else:
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

    ax2.set_xlabel("Freq [Hz]")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend(ncol=3)

    return fig
