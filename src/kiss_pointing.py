#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:49:13 2019

"""
import numpy as np
from astropy.time import TimeDelta

from matplotlib import pyplot as plt

from Labtools_JM_KISS import kiss_coordinates as kc
from Labtools_JM_KISS import kiss_pointing_model as kpm

def kiss_obstime(self): # self =  kiss
    """ Get observation time of each interfergram in UTC. """
    time_pps = (self.A_hours + self.A_time)
    time_pps = time_pps.reshape(self.nint, self.nptint)
    time_mint = np.median(time_pps, axis = 1) # Median UT time for each interfergram
    time_mint = TimeDelta(time_mint, format = 'sec')

    obstime = self.obs['date_utc'] + time_mint
    
    self.obstime = obstime
    return obstime

#%%
    

def check_pointing_source(self): # self = kiss
    """ Check pointing with a source. """
    obstime = kiss_obstime(self)
    RDS, AES, tframe = kc.get_sourceAzEl(self.obs['source'],obstime)
    AzEl_S = kc.azel2coord(AES.az.deg,AES.alt.deg,obstime)

    az_tel = np.rad2deg(self.F_tl_Az/1000.0)                                      
    el_tel = np.rad2deg(self.F_tl_El/1000.0)
    az_sky = np.rad2deg(self.F_sky_Az/1000.0)  
    el_sky = np.rad2deg(self.F_sky_El/1000.0)

    
    KISSpm = kpm.KISSPmodel()
    KISSpmQ1 = kpm.KISSPmodel(model='Q1')
    az_sky, el_sky = KISSpm.telescope2sky(az_tel,el_tel)
    az_skyQ1, el_skyQ1 = KISSpmQ1.telescope2sky(az_tel,el_tel)


    AzEl_tel = kc.azel2coord(az_tel,el_tel,obstime) 
    AzEl_sky = kc.azel2coord(az_sky,el_sky,obstime)

    # Compute coordinates offset                                                                                                                                              
    az_diff_tel = AES.az.deg  - az_tel
    el_diff_tel = AES.alt.deg - el_tel

    ra_diff_tel  =  AzEl_S.icrs.ra.deg -AzEl_tel.icrs.ra.deg
    dec_diff_tel =  AzEl_S.icrs.dec.deg -AzEl_tel.icrs.dec.deg

    pflag = (self.F_tl_Az > 0) & (self.F_tl_El > 0)


    plt.figure()
    plt.plot(az_tel[pflag],el_tel[pflag],'.',label='Tel',markersize=0.5)
    plt.plot(az_sky[pflag],el_sky[pflag],'.',label='Sky',markersize=0.5)
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    
    return az_tel[pflag],el_tel[pflag], az_sky[pflag],el_sky[pflag]