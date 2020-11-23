import numpy as np
import pytest

from astropy.table import Table

from kidsdata.utils import project


def otf(xlen=200, xstep=9, ylen=200, ystep=9, zigzag=True):
    # 3.2.6.8 from APEX-MPI-MAN-0011-R3_2.pdf

    nX = xlen // xstep
    nY = ylen // ystep
    x = np.arange(0, xlen, xstep) - xlen // 2
    x = np.tile(x, nY).reshape(nY, nX)

    if zigzag:
        x[np.mod(np.arange(nY), 2) == 1] *= -1

    y = np.arange(0, ylen, ystep) - ylen // 2
    y = np.repeat(y, nX).reshape(nY, nX)

    return x, y


def gen_kidpar(nkid_x=5, nkid_y=5, dkid_x=5, dkid_y=5):
    return Table(
        {
            "x0": np.repeat((np.arange(nkid_x) - nkid_x // 2) * dkid_x, nkid_y),
            "y0": np.tile((np.arange(nkid_y) - nkid_y // 2) * dkid_y, nkid_x),
        }
    )


def gen_source_uniform(shape, n_sources, f=None, marging=None):

    if marging is None:
        x_ = np.random.uniform(shape[1], size=n_sources)
        y_ = np.random.uniform(shape[0], size=n_sources)
    else:
        x_ = np.random.uniform(marging[1], shape[1] - marging[1], size=n_sources)
        y_ = np.random.uniform(marging[0], shape[0] - marging[0], size=n_sources)

    if f is not None:
        f_ = np.random.uniform(*f, size=n_sources)
    else:
        f_ = np.ones_like(x_)

    return x_, y_, f_


def gen_source_timeline(x, y, kidpar, x_sources, y_sources, f_sources):
    source_timelines = np.zeros((len(kidpar), len(x)))

    for x_source, y_source, f_source in zip(x_sources, y_sources, f_sources):
        for i, (x0, y0) in enumerate(zip(kidpar["x0"], kidpar["y0"])):
            source_timelines[i] += f_source * np.exp(
                -((x + x0 - x_source) ** 2 + (y + y0 - y_source) ** 2) / (2 * sigma ** 2)
            )
    return source_timelines


def gen_colored_noise(length, res=1, knee=1e-4, alpha=-1.5):
    freq = np.abs(np.fft.fftfreq(length, d=res))
    psd = 2 * (freq / knee) ** alpha
    psd[0] = 0
    pha = np.random.uniform(low=-np.pi, high=np.pi, size=length)
    fft_ = np.sqrt(psd) * (np.cos(pha) + 1j * np.sin(pha))
    noise_timeline = np.real(np.fft.ifft(fft_)) * length / res ** 2
    return noise_timeline


n_pix = 128
sigma = 2
n_sources = 10

kidpar = gen_kidpar()
x_sources, y_sources, f_sources = gen_source_uniform((n_pix, n_pix), n_sources, (1, 5), (10, 10))

x, y = otf(xlen=n_pix, ylen=n_pix, xstep=1, ystep=1)
x = x.flatten() + n_pix // 2
y = y.flatten() + n_pix // 2

source_timelines = gen_source_timeline(x, y, kidpar, x_sources, y_sources, f_sources)

correlated_noise_timeline = gen_colored_noise(len(x), knee=1e-4)
noise_timelines = np.random.normal(0, 2, source_timelines.shape)

# Timelines, no gain
timelines = source_timelines + noise_timelines + correlated_noise_timeline

## True map
xx, yy = np.meshgrid(np.arange(n_pix), np.arange(n_pix))
source_map = np.zeros((n_pix, n_pix))
for x_source, y_source, f_source in zip(x_sources, y_sources, f_sources):
    source_map += f_source * np.exp(-((xx - x_source) ** 2 + (yy - y_source) ** 2) / (2 * sigma ** 2))

source_mask = source_map > 0.1

x_all = (x[:, np.newaxis] + kidpar["x0"]).T
y_all = (y[:, np.newaxis] + kidpar["y0"]).T
raw_data, raw_weight, raw_hits = project(x_all.flatten(), y_all.flatten(), timelines.flatten(), (n_pix, n_pix))

filtered_timelines = median_filtering(timelines.copy(), ikid_ref=0, offset=True, flat=True)
median_data, median_weight, median_hits = project(
    x_all.flatten(), y_all.flatten(), filtered_timelines.flatten(), (n_pix, n_pix)
)

filtered_timelines = pca_filtering(timelines.copy(), ncomp=1)
pca_data, pca_weight, pca_hits = project(x_all.flatten(), y_all.flatten(), filtered_timelines.flatten(), (n_pix, n_pix))

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

norm = Normalize(vmin=np.min(source_map), vmax=np.max(source_map))
to_plot = [source_map, raw_data, median_data - source_map, pca_data - source_map]

fig, axes = plt.subplots(ncols=len(to_plot), sharex=True, sharey=True)
for ax, data in zip(axes, to_plot):
    ax.imshow(data, origin="lower")
    ax.scatter(kidpar["x0"] + n_pix // 2, kidpar["y0"] + n_pix // 2, alpha=0.5)
