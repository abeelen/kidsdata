import logging

from datetime import datetime

from pathlib import Path
import matplotlib.pyplot as plt
from src.kiss_data import KissRawData
from src.db import get_extra
import numpy as np
import warnings

from scipy.optimize import curve_fit
from astropy.stats import mad_std

logging.basicConfig(level=logging.INFO)

plt.ion()

scans = get_extra(start=datetime(2019, 5, 1, 19, 14, 24), end=datetime(2019, 5, 1, 19, 52, 53))

title = Path(scans[0]).name + " ".join([Path(scan).name.split("_")[4] for scan in scans[1:]])

signal = []
std = []
elevation = []

for scan in scans:

    kd = KissRawData(scan)
    kd.read_data(list_data="A_masq I Q F_tone F_tl_Az F_tl_El")

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
fig, axes = plt.subplots(np.int(np.sqrt(ndet)), np.int(ndet / np.sqrt(ndet)), sharex=True)  # , sharey=True)
for _sig, _std, popt, detector, ax in zip(signal_new.T, std.T, popts, detectors, axes.flatten()):
    ax.errorbar(air_mass, _sig, _std)
    ax.plot(air_mass, T(air_mass, *popt))
    ax.set_title(detector, pad=-15)
    ax.label_outer()

fig.suptitle(title)
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

Ao, Bo, tau = popts.T

fig, axes = plt.subplots(1, 3)
for (item, value), ax in zip({r"$A_0$": Ao, r"$B_0$": Bo, "tau": tau}.items(), axes):
    mean_value = np.nanmedian(value)
    std_value = mad_std(value, ignore_nan=True)
    range_value = np.array([-3, 3]) * std_value + mean_value
    ax.hist(value, range=range_value)
    ax.set_xlabel(item)
fig.suptitle(title)
