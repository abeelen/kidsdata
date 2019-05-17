#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:27:34 2019

@author: ycao
"""
import numpy as np
from scipy.interpolate import interp1d
#Define time variables for interolation

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
    
