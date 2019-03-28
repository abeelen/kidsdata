#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:38:32 2019

@author: yixiancao
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage.filters import uniform_filter1d as smooth
#from scipy.ndimage.filters import gaussian_filter1d

from Labtools_JM_KISS import kiss_pointing_model as kpm
#from Labtools_JM_KISS import kiss_map_proj as kmp

def calibPlot(kids, ikid = 195):
    fig = plt.figure(figsize=(5*3,4*2))
    
    ax = plt.subplot(2,3,1)
    ax.plot(kids.Icc[ikid,:],label='Original')
    ax.plot(smooth(kids.Icc[ikid,:],21),label='Smoothed')
    ax.grid()
    ax.set_ylabel('I circle center [arbitrary units]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2,3,2)
    ax.plot(kids.Qcc[ikid,:],label='Original')
    ax.plot(smooth(kids.Qcc[ikid,:],21),label='Smoothed')
    ax.grid()
    ax.set_ylabel('Q circle center [arbitrary units]')
    ax.set_xlabel('Sample Number')
    ax.legend()

    ax = plt.subplot(2,3,3)
    ax.plot(kids.calfact[ikid,:],label='Original')
    ax.plot(smooth(kids.calfact[ikid,:],21),label='Smoothed')
    ax.grid()
    ax.set_ylabel('Calibration Factor [Hz/rad]')
    ax.set_xlabel('Sample Number')
    ax.legend()


    ax = plt.subplot(2,3,4)
    ax.plot(np.median(kids.calfact,axis=1), label = 'Original')
    ax.plot(medfilt(np.median(kids.calfact,axis=1),5), label = 'Fitted')
    ax.grid()
    ax.set_ylabel('Median Calibration Factor [Hz/rad]')
    ax.set_xlabel('Detector Number')
    ax.legend()

    ax = plt.subplot(2,3,5)
    ax.plot(kids.kidfreq[ikid,4:12].ravel(), \
            label = 'Detector:' + kids.kidpar['namedet'][ikid])
    ax.grid()
    ax.set_ylabel('Signal [Hz]')
    ax.set_xlabel('Sample Number')
    ax.legend()
    
    fig.suptitle(kids.filename)

    return fig
    
    # plotname = dir_plot + plotname
    # fig.savefig(plotname)
    
    # print ('Figure saved: ' + plotname )
    
    # return plotname

def checkPointing(kids):
    fig = plt.figure(figsize=(5*2+1,4*2))
    fig.suptitle(kids.filename)

    # better to do the unit conversion in reading? 
    for key in ['F_azimuth', 'F_elevation']:
        kids.__dict__[key] = np.rad2deg(kids.__dict__[key]/1000.0)


    okp = np.where((kids.F_elevation >0.0) & (kids.F_azimuth > 0.0))[0]

    ax = plt.subplot(2,2,1)
    ax.plot(kids.F_azimuth[okp]) 
    ax.set_ylabel('Azimuth [deg]')
    ax.set_xlabel('Sample number [dummy units]')
    ax.grid()
    
    ax = plt.subplot(2,2,2)
    ax.plot(kids.F_elevation[okp])
    ax.set_xlabel('Sample number [dummy units]')
    ax.set_ylabel('Elevation [deg]')
    ax.grid()
    
    ax = plt.subplot(2,2,3)
    ax.plot(kids.F_azimuth[okp],
            kids.F_elevation[okp],'.')
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Elevation [deg]')
    ax.set_title('Pointing')
    ax.grid()
    
    az_tel = np.median(kids.F_azimuth.reshape((kids.nint,kids.nptint)),axis=1)
    el_tel = np.median(kids.F_elevation.reshape((kids.nint,kids.nptint)),axis=1)
    
    okp = np.where((el_tel >0.0) & (az_tel > 0.0))[0]
    print (len(okp), az_tel[okp], el_tel[okp])
    KISSpm = kpm.KISSPmodel()
    KISSpmQ1 = kpm.KISSPmodel(model='Q1')
    az_sky, el_sky = KISSpm.telescope2sky(az_tel,el_tel)
    az_skyQ1, el_skyQ1 = KISSpmQ1.telescope2sky(az_tel,el_tel)
    # cache
    
    ax = plt.subplot(2,2,4)
    plt.plot(az_tel[okp],el_tel[okp],'+', ms =12,  label='Telescope')
    ax.plot(az_sky[okp],el_sky[okp],'+', ms = 12, label='Sky')
    ax.plot(az_skyQ1[okp],el_skyQ1[okp],'+', ms = 12, label='Sky Q1')
    ax.set_xlabel('Azimuth [deg]')
    ax.set_ylabel('Elevation [deg]')
    ax.grid()
    ax.legend()
    
    return fig