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
import astropy.constants as cst
from src import kids_data


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



#%%---Compute and correct positions of MP  

def diffm(dlaser, 
          appendv = None, appendi = None,
          prependv = None, prependi = None,
          **kwargs):
    """ Calculate the differences between laser positions at adjacent time sample 
        
    Parameters: 
    dlaser: 
        position of the laser 
    append: 
        Append value. Default:1 #Y.C. Why use 1? 
    """
    diffm = np.diff(dlaser,prepend = mpos1[-1], **kwargs)
   
    if appendi is not None:
        appendv = diffm[appendi]
    if prependi is not None:
        prependv = diffm[prependi]
        
    if appendv is not None:
        diffm[-1] = appendv
    if prependv is not None:
        diffm[0]  = prependv 

    return diffm 
#%%


def kids_itp_flag(data, x = None,  flag = None, replace = False, **kwargs):
    """ Interpolate the flagged 1D data.
    
    Parameters
    ----------
    data: 1D array 
        Data to be fixed
    x: array_line
        1D array of same size as data
    flag: boolean array
        flag of the data. True for good data, False for bad data. 
    replace: 
        Replace bad data  by interpolated the flagged data.
    
    """

    # Replace flagged out data by interpolation 
    if x is None: 
        x = np.arange(data.shape[0], dtype =np.double)
    if flag is None:
        flag = np.full(data.shape, True, dtype=bool)
    interp = interp1d(x[flag],data[flag], fill_value="extrapolate", **kwargs )
    
#    kiss.__dict__[t_type+'_itp'] = interp(tidx)
    if replace:
        data[~flag] = interp(x[~flag])
        print ('Using interpolated data to replace the unflagged data.')
    return interp, data
#%%
plt_1d = lambda data, **kwargs: plt.plot(np.arange(data.shape[0]), data, **kwargs)

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



#%% Compute baseline for each interferogram    
dint_pos = self.kidfreq[:,:,npos:nneg]
dint_neg = self.kidfreq[:,:,nneg:]
nbaseline = 101
dint_pos_base = smooth(dint_pos,size=nbaseline,axis=2)
dint_neg_base = smooth(dint_neg,size=nbaseline,axis=2)

#%% Median filterred and base-line substracted, calibrated interfergram signals 
# Show an example on a particular KID detector

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
plt.legend()
plt.xlim(0, 0.25)

#%% Generate new positions by what standards?  
pzero = np.argmax(np.abs(intermed)) # Y.C. largest in signal as the modulation starts?   
xpos = mpos1b - mpos1b[pzero]
xpos2 = mpos2b - mpos2b[pzero] # path differences 

dxpos = diffm(xpos, prependi = -1) # velocity? 
dxpos2 = diffm(xpos2, prependi = -1) 

xposmax = np.max(np.abs(xpos))
xposmax = 20.0

delta_xpos = np.max(np.gradient(xpos)) # maxinum velocity 
#delta_xpos = np.max(np.gradient(xpos))*0.8                                                                                                                               

nzero =  np.int(xposmax/delta_xpos)
xposmaxnew = nzero * delta_xpos
xposnew = np.arange(-xposmaxnew,xposmaxnew,delta_xpos)


#%% ??

c_val = cst.c.to('mm/s')
Delta_X = xposnew.max()
delta_freq = c_val.value/4.0/Delta_X/1e9
freq = np.arange(len(xposnew)/2)*delta_freq
freqbase = np.concatenate((freq,np.flip(-freq[1:])))
#%% extrapolate measurements on negtive positions for all the newly derived positions 

dintmednew_pos = np.zeros(kiss.ndet)
flag = (xpos <= 0.0)
interp, dumm = kids_itp_flag(intermed, x = np.abs(xpos), 
                             kind='slinear', flag = flag) # did not see diff between slinear and linear
intermednew = interp(np.abs(xposnew))
intermednew  -=  np.median(intermednew)

intermed2 =  interp(np.abs(xpos))
#%% To all KIDS; intermed_arr has a different treatment than on the single one; 
#show different results 
#intermed_arr = np.median(dint_pos[:,ibeg:,:]-np.median(dint_pos[:,ibeg:,:],axis=2,keepdims=True),axis=1)                                                                 
intermed_arr = dint_pos-dint_pos_base 
intermed_arr = np.median(intermed_arr,axis=1)
finter_arr = interp1d(np.abs(xpos[flag]),intermed_arr[:,flag],
                      kind='slinear',axis=1, fill_value="extrapolate")
intermednew_arr = finter_arr(np.abs(xposnew))
intermed2_arr =  finter_arr(np.abs(xpos))

#%%
plt.figure()
plt.plot(xpos, intermed, label='Original')
plt.plot(xposnew,intermednew,'.',label='New interpolated')
plt.plot(xpos, intermed2,'o',label='Original interpolated')
plt.legend()

#%%
ikid = 0
plt.plot(xpos, intermed, label='Original, ikid = '+str(ikid))
plt.plot(xposnew,intermednew_arr[ikid,:],'.',label='New interpolated')
plt.plot(xpos, intermed2_arr[ikid,:],'o',label='Original interpolated')
plt.legend()

#%%
#return