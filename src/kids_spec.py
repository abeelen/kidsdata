#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May 24 14:12:42 2019
"""

import numpy as np
from matplotlib import pyplot as plt
from  scipy.ndimage.filters import uniform_filter1d as smooth
from scipy.interpolate import interp1d
from scipy.signal import medfilt

#%%

datadir = '/Users/yixiancao/Work/Concerto/Data/kissRaw/'
pltdir = '/Users/yixiancao/workCodes/kidsdata/plots/'

#filename = 'X_2018_12_13_19h09m31_AA_man' # This one show erros for header reading. 
#filename = 'X_2018_12_14_11h58m14_AA_man'
#filename = 'X_2018_12_14_11h55m15_AA_man'
filename = 'X20190427_0910_S0319_Moon_SCIENCEMAP' 
filename = datadir + filename
kiss = kids_data.KissRawData(filename)
kiss.calib_raw()

#%%
#def kids_spec(self):
    
    # Data needed: C_laser1_pos, C_laser2_pos
    # From spectro_qt
    # Basic steps:


    # Filter out low frequency components in the timeline 
    # Y.C.: move to __init__ for KissRawData
self = kiss
npts_mod = 134 # points for modulation
npts_int = int((self.nptint - npts_mod)/2) # points for integration 


attrlist = ['C_laser1_pos', 'C_laser2_pos'] 
self.check_attr(attrlist)

laser1_pos = self.C_laser1_pos.reshape(self.nint, self.nptint)
laser2_pos = self.C_laser2_pos.reshape(self.nint, self.nptint)

# Separate individual interferograms  
npos = npts_mod
nneg = npts_mod + npts_int

dlaser1_pos = laser1_pos[:, npos:nneg]
dlaser1_neg = laser1_pos[:, nneg:]

dlaser2_pos = laser2_pos[:, npos:nneg]
dlaser2_neg = laser2_pos[:, nneg:]


#%% Compute baseline for each interferogram    
dint_pos = self.kidfreq[:,:,npos:nneg]
dint_neg = self.kidfreq[:,:,nneg:]
nbaseline = 101
dint_pos_base = smooth(dint_pos,size=nbaseline,axis=2)
dint_neg_base = smooth(dint_neg,size=nbaseline,axis=2)


#%%---Compute and correct positions of MP  

def diffm(dlaser, append = 1):
    """ Calculate the differences between laser positions at adjacent time sample 
        
    Parameters: 
    dlaser: 
        position of the laser 
    append: 
        Append value. Default:1 #Y.C. Why use 1? 
    """
    diffm = np.diff(dlaser,prepend = mpos1[-1])
    diffm[-1] = 1

    return diffm 
#%%

# Initial position of MP
mpos1 = dlaser1_pos[0,:]
mpos2 = dlaser2_pos[0,:]
mneg1 = dlaser1_neg[0,:]
mneg2 = dlaser2_neg[0,:]

dmpos1 = diffm(mpos1)
dmpos2 = diffm(mpos2)
dmneg1 = diffm(mneg1)
dmneg2 = diffm(mneg2)

# Fix bad data
mpos1b = kids_itp_flag(dmpos1, flag = dmpos1 > 1e-4)
mpos2b = kids_itp_flag(dmpos2, flag = dmpos2 > 1e-4)
mneg1b = kids_itp_flag(dmneg1, flag = dmpos1 > 1e-4)
mneg2b = kids_itp_flag(dmneg2, flag = dmpos1 > 1e-4)



#%%


def kids_itp_flag(data, flag = None):
    """ Replace bad (not flagged) data by interpolated the flagged data.
    
    Parameters
    ----------
    data: 1D array 
        Data to be fixed
    flag: boolean array
        flag of the data. True for good data, False for bad data
    
    """

    # Replace flagged out data by interpolation 
    idx = np.arange(data.shape[0], dtype =np.double)
    interp = interp1d(idx[flag],data[flag], kind = 'linear', fill_value="extrapolate" )
    
#    kiss.__dict__[t_type+'_itp'] = interp(tidx)

    data[~flag] = interp(idx[~flag])
    return data
#%%
plt_1d = lambda data, **kwargs: plt.plot(np.arange(data.shape[0]), data, **kwargs)

#%%

ikid = 123
ibeg = 0
plt.figure()
plt.clf()
isig =  np.median(dint_pos[ikid,:,:]-dint_pos_base[ikid,:,:], axis=0)
plt.plot(mpos1b,isig, lw = 1, marker = '.', label = "iKID: "+str(ikid))

intermed = np.median(dint_pos[ikid,ibeg:,:]-dint_pos_base[ikid,:,:],axis=0)
#intermed -= np.mean(intermed)                                                                                                                                            
intermed -=  medfilt(intermed,101)
plt.plot(mpos1b,intermed, lw = 1, marker = '^', ms = 1, ls = 'None', label = '101 medianfit')
plt.plot(mpos1b, intermed - isig)
plt.xlim(0, 0.25)

#return