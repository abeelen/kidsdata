#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate kids_raw_data, fix/flag the bad if necessary 

Created on Tue May 21 15:00:37 2019
@author: yixiancao
"""

import numpy as np
from scipy.interpolate import interp1d

def kids_validate(kids, **kwargs):
        # Compute time pps_time difference

    return


def kids_time(kiss):
    kids_itp_time(kiss, t_type = 'A_hours')
    kids_itp_time(kiss, t_type='A_time_pps') 
    
# mdate = date[0:4]+'-'+date[4:6]+'-'+date[6:]    
#     mdate = date[0:4]+'-'+date[4:6]+'-'+date[6:]
    
    # Median UT time for each interfergram
    utt = (kiss.A_hours_itp + kiss.A_time_itp)/3600.0
    utt = utt.reshape(kiss.nint, kiss.nptint)
    uti = np.median(utt, axis = 1)
    
    return uti

def kids_itp_time(kiss, t_type = 'A_hours'):
    """ Replace bad time data by interpolated data"""
    
    t_orig = kiss.__dict__[t_type]#.reshape(kiss.nint, kiss.nptint)    

    if t_type == 'A_hours': # flag out bad time data
        t_median = np.median(t_orig)
        tflag = np.abs(t_orig-t_median) < 1.1 
    elif t_type == 'A_time_pps':
        tflag = t_orig > 1.0e-4 # non-zero
    
    # Replace flagged out data by interpolation 
    tidx = np.arange(kiss.nsamples, dtype =np.double)
    interp = interp1d(tidx[tflag],t_orig[tflag])
    
    kiss.__dict__[t_type+'_itp'] = interp(tidx)
    
    return


def kids_time(kiss):
    """ 
    Parameters
    ----------
    kiss: KissRawData
        Kiss Raw data 
    
    """
    
    list_data = 'A_time_ntp A_time_pps A_time A_hours' 
    kiss.read_data(list_data = list_data)
    
    if 'A_time' in dataSc:
        pps = dataSc['A_time']
        other_time = [key for key in dataSc if key.endswith('_time') and key != 'A_time']
        if other_time and 'sample' in dataSc:
            pps_diff = {'A_time-{}'.format(key): (pps - dataSc[key]) * 1e6 for key in other_time}
            pps_diff['pps_diff'] = np.asarray(list(pps_diff.values())).max(axis=0)

            dataSc.update(pps_diff)

        # Fake pps time if necessary
        if correct_pps:
            dummy = np.diff(pps, append=0)
            good = np.abs(dummy - 1 / param_c['acqfreq']) < 0.02
            if any(~good):
                param = np.polyfit(dataSc['sample'][good], pps[good], 1)
                pps[~good] = np.polyval(param, dataSc['sample'][~good])

        dataSc['pps'] = pps

    gc.collect()
    ctypes._reset_cache()

