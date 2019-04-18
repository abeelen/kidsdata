#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:38:32 2019

@author: yixiancao
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter1d as smooth
from astropy.wcs import WCS
#from scipy.ndimage.filters import gaussian_filter1d

#from Labtools_JM_KISS import kiss_map_proj as kmp


def calibPlot(kids, ikid=0):
    """ Plot Icc, Qcc, calfact, and kidfreq distributions for ikid detector;
        show median calfact for all detectors in the last panel.
    """
    fig = plt.figure(figsize=(5 * 3, 4 * 2))

    ax = plt.subplot(2, 3, 1)
    ax.plot(kids.Icc[ikid, :], label='Original')
    ax.plot(smooth(kids.Icc[ikid, :], 21), label='Smoothed')
    ax.grid()
    ax.set_ylabel('I circle center [arbitrary units]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2, 3, 2)
    ax.plot(kids.Qcc[ikid, :], label='Original')
    ax.plot(smooth(kids.Qcc[ikid, :], 21), label='Smoothed')
    ax.grid()
    ax.set_ylabel('Q circle center [arbitrary units]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2, 3, 3)
    ax.plot(kids.calfact[ikid, :], label='Original')
    ax.plot(smooth(kids.calfact[ikid, :], 21), label='Smoothed')
    ax.grid()
    ax.set_ylabel('Calibration Factor [Hz/rad]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2, 3, 4)
    ax.plot(kids.kidfreq[ikid, 4:12].ravel(),
            label='Detector:' + kids.kidpar['namedet'][ikid])
    ax.grid()
    ax.set_ylabel('Signal [Hz]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2, 3, 5)
    ax.plot(np.median(kids.calfact, axis=1), label='Original')
    ax.plot(medfilt(np.median(kids.calfact, axis=1), 5), label='Fitted')
    ax.grid()
    ax.set_ylabel('Median Calibration Factor [Hz/rad]')
    ax.set_xlabel('Detector Number')
    ax.legend()

    fig.suptitle(kids.filename)
    fig.tight_layout()

    return fig


def checkPointing(kids):
    """ Plot:
        1. Azimuth distribution of samples.
        2. Elevation distribuiton of samples.
        3. 2D distribution of (Elevation, Azimuth) for samples.
        4. Medians of poitnings for each interferogram,
        compared with pointing models.
    """
    fig = plt.figure(figsize=(5 * 2 + 1, 4 * 2))
    fig.suptitle(kids.filename)

    azimuth, elevation, mask_pointing = kids.F_azimuth, kids.F_elevation, kids.mask_pointing
    az_tel, el_tel, mask_tel = kids.az_tel, kids.el_tel, kids.mask_tel
    az_sky, el_sky = kids.az_sky, kids.el_sky
    az_skyQ1, el_skyQ1 = kids.az_skyQ1, kids.el_skyQ1

    ax = plt.subplot(2, 2, 1)
    ax.plot(azimuth[mask_pointing])
    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlabel('Sample number [dummy units]')
    ax.grid()

    ax = plt.subplot(2, 2, 2)
    ax.plot(elevation[mask_pointing])
    ax.set_xlabel('Sample number [dummy units]')
    ax.set_ylabel('Elevation [deg]')
    ax.grid()

    ax = plt.subplot(2, 2, 3)
    ax.plot(azimuth[mask_pointing],
            elevation[mask_pointing], '.')
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Elevation [deg]')
    ax.set_title('Pointing')
    ax.grid()

    ax = plt.subplot(2, 2, 4)
    plt.plot(az_tel[mask_tel], el_tel[mask_tel], '+', ms=12, label='Telescope')
    ax.plot(az_sky[mask_tel], el_sky[mask_tel], '+', ms=12, label='Sky')
    ax.plot(az_skyQ1[mask_tel], el_skyQ1[mask_tel], '+', ms=12, label='Sky Q1')
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Elevation [deg]')
    ax.grid()
    ax.legend()

    fig.tight_layout()

    return fig


def photometry(kids):
    fig = plt.figure(figsize=(5 * 2 + 1, 4 * 2 + 0.5))
    fig.suptitle(kids.filename)

    bgrd = kids.background
    meds = np.median(bgrd, axis=1)
    stds = np.std(bgrd, axis=1)

    ax = plt.subplot(2, 2, 1)
    ax.semilogy(meds)
    ax.set_xlabel('Detector Number')
    ax.set_ylabel("Median of Photometry")

    ax = plt.subplot(2, 2, 2)
    ax.semilogy(stds)
    ax.set_xlabel('Detector Number')
    ax.set_ylabel("STD of Photometry")

    fig.tight_layout()

    return fig


def show_maps(kids, ikid=0):
    nrow = 1
    ncol = 1
    fig = plt.figure(figsize=(5 * ncol + 1, 4 * nrow + 0.5))

    subtitle = str(testikid)
    fig.suptitle(kids.filename + subtitle)

    ax = plt.subplot(ncol, nrow, 1)
    ax.imshow(kids.beammap)

#    wcs = kids.beamwcs
#    ax = plt.subplot(ncol, nrow, 1, projection=wcs)
#    bgrs = kids.bgrs[153, kids.mask_tel]
#    ax.imshow(bgrs)

    ax.grid(color='white', ls='solid')
#    ax.set_xlabel('Az [deg]')
#    ax.set_ylabel('El [deg]')

    return fig
