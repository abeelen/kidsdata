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

def calibPlot(kids, ikid = 195, dir_plot = '../plots/', plotname = 'calib.pdf'):
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

    
    plotname = dir_plot + plotname
    fig.savefig(plotname)
    
    print ('Figure saved: ' + plotname )
    
    return plotname