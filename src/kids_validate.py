#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate kids_raw_data, fix/flag the bad if necessary 

Created on Tue May 21 15:00:37 2019
@author: yixiancao
"""

import numpy as np
from scipy.interpolate import interp1d


def kids_validate(kiss, correct_time = True, 
                  **kwargs):
    
    # Compute time pps_time difference
    if correct_time: kids_time_validate(kiss)

    return

#%%

def kids_time_validate(kiss):
    attrlist = ['A_time_ntp', 'A_time_pps',
                  'A_time', 'A_hours']
    
    check_attr(kiss,attrlist)

    kids_itp_time(kiss, t_type = 'A_hours')
    kids_itp_time(kiss, t_type='A_time_pps') 
    
    return 


#%%

def kids_itp_time(kiss, t_type = 'A_hours'):
    """ Replace bad time data by interpolated data.
    
    Parameters
    ----------
    kiss: :obj: KissRawData
        Kiss Raw Data read from data files. 
    t_type: str
        Time to be interpolated. 
    
    """
    
    t_orig = kiss.__dict__[t_type]#.reshape(kiss.nint, kiss.nptint)    

    if t_type == 'A_hours': # flag out bad time data
        t_median = np.median(t_orig)
        tflag = np.abs(t_orig-t_median) < 1.1 
    elif t_type == 'A_time_pps':
        tflag = t_orig > 1.0e-4 # non-zero
    
    # Replace flagged out data by interpolation 
    tidx = np.arange(kiss.nsamples, dtype =np.double)
    interp = interp1d(tidx[tflag],t_orig[tflag])
    
#    kiss.__dict__[t_type+'_itp'] = interp(tidx)

    kiss.__dict__[t_type] = interp(tidx)
    kiss.logger.history("Replaced bad "+ t_type +" by interpolation." )    
    
    return 

#%%

def kids_pfit_time(kiss):
    """ Replace bad time data by polynomial linear fitting. 
    
    Parameters
    ----------
    kiss: :obj: KissRawData
        Kiss Raw data read from data files.
    
    """
    try:
        pps = kiss.A_time
        other_time = [key for key in kiss.__dict__ if key.endswith('_time') and key != 'A_time']
        if hasattr(kiss, 'sample') and hasattr(kiss, 'sample'):
            pps_diff = {'A_time-{}'.format(key): (pps - kiss.__dict__[key]) * 1e6 for key in other_time}
            pps_diff['pps_diff'] = np.asarray(list(pps_diff.values())).max(axis=0)

            kiss.pps_diff = pps_diff

        # Fake pps time if necessary
        dummy = np.diff(pps, append=0)
        good = np.abs(dummy - 1 / kiss.param_c['acqfreq']) < 0.02
        if any(~good):
            param = np.polyfit(kiss.sample[good], pps[good], 1)
            pps[~good] = np.polyval(param, kiss.sample[~good])

        kiss.pps = pps
        kiss.logger.history("Replaced bad A_time by polynomial linear fitting." )    

    except AttributeError: 
        print ("Time data not read.")


#%%

def kids_pointing_validate(self):
    attrlist = ['F_tl_Az', 'F_tl_El']
    self.check_attr(attrlist)

    fig = self.pointing_plot()

    pflag = (self.F_tl_Az > 0) & (self.F_tl_El > 0)
