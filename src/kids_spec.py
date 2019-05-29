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
import numpy.ma as ma

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
dumm, mpos1b = kids_itp_flag(dmpos1, flag = dmpos1 > 1e-4, replace = True)
dumm, mpos2b = kids_itp_flag(dmpos2, flag = dmpos2 > 1e-4, replace = True)
dumm, mneg1b = kids_itp_flag(dmneg1, flag = dmneg1 > 1e-4, replace = True)
dumm, mneg2b = kids_itp_flag(dmneg2, flag = dmneg2 > 1e-4, replace = True)


#%% Compute baseline for each interferogram    
dint_pos = self.kidfreq[:,:,npos:nneg]
dint_neg = self.kidfreq[:,:,nneg:]
nbaseline = 101 # Why this number? 
dint_pos_base = smooth(dint_pos,size=nbaseline,axis=2) #baseline of each interfergram
dint_neg_base = smooth(dint_neg,size=nbaseline,axis=2)
#%%
# Plot base line and compare with orignial inferfergram
ikid = 2
iint = 4
plt.plot(dmneg1, dint_neg[ikid, iint,:])
plt.plot(dmneg1, dint_neg_base[ikid,iint,:])

#%% Median filterred and base-line substracted, calibrated interfergram signals 
# Show an example on a particular KID detector
ikid = 123
ibeg = 0
#Median of interfergrams starting from ibeg, baseline substracted 
#intermed -= np.mean(intermed)  
intermed = np.median(dint_pos[ikid,ibeg:,:]-dint_pos_base[ikid,:,:],axis=0)
intermed -=  medfilt(intermed,101) # substracted the median baseline 

#%% Plot the comparison 

plt.figure()
plt.clf()
isig =  np.median(dint_pos[ikid,:,:]-dint_pos_base[ikid,:,:], axis=0)
plt.plot(mpos1b,isig, lw = 1, marker = '.', label = "iKID: "+str(ikid))

plt.plot(mpos1b,intermed, lw = 1, marker = '^', ms = 1, ls = 'None', label = '101 medianfit')
plt.plot(mpos1b, intermed - isig, label = 'Diff')
plt.legend()
plt.xlim(0, 0.25)

#%% Generate new positions by what standards?  
# Why create this new position instead of using the original ones? 
pzero = np.argmax(np.abs(intermed)) # Y.C. largest in signal as the modulation starts?   
xpos = mpos1b - mpos1b[pzero]
xpos2 = mpos2b - mpos2b[pzero] # path differences 

dxpos = diffm(xpos, prependi = -1) # velocity? 
dxpos2 = diffm(xpos2, prependi = -1) 

xposmax = np.max(np.abs(xpos)) # 0.2125: continuum? 
#xposmax = 20.0

delta_xpos = np.max(np.gradient(xpos)) # maxinum velocity 
#delta_xpos = np.max(np.gradient(xpos))*0.8                                                                                                                               

nzero =  np.int(xposmax/delta_xpos) # Number of sampling
xposmaxnew = nzero * delta_xpos
xposnew = np.arange(-xposmaxnew,xposmaxnew,delta_xpos)

#%% Get freqency in units of ??

c_val = cst.c.to('mm/s')
delta_x = xposnew.max()
delta_freq = c_val.value/4.0/delta_x/1e9

freq = np.arange(len(xposnew)/2)*delta_freq
freqbase = np.concatenate((freq,np.flip(-freq))) # Why use np.flip(-freq[-1:]) before? 

#%% extrapolate measurements on negtive positions for all the newly derived positions 

#dintmednew_pos = np.zeros(kiss.ndet) #not used... 
flag = (xpos <= 0.0)
interp, dumm = kids_itp_flag(intermed, x = np.abs(xpos), 
                             kind='slinear', flag = flag) # did not see diff between slinear and linear
intermed2 =  interp(np.abs(xpos))
intermednew = interp(np.abs(xposnew))
intermednew  -=  np.median(intermednew)

#%% To all KIDS; intermed_arr has a different treatment than on the single one; 
#show different results 
#intermed_arr = np.median(dint_pos[:,ibeg:,:]-np.median(dint_pos[:,ibeg:,:],axis=2,keepdims=True),axis=1)                                                                 
intermed_arr = dint_pos - dint_pos_base 
intermed_arr = np.median(intermed_arr,axis=1)
finter_arr = interp1d(np.abs(xpos[flag]),intermed_arr[:,flag],
                      kind='slinear',axis=1, fill_value="extrapolate")
intermed2_arr =  finter_arr(np.abs(xpos))
intermednew_arr = finter_arr(np.abs(xposnew))

#%%
plt.figure()
plt.plot(xpos, intermed, label='Original')
plt.plot(xpos, intermed2,'o',label='Original interpolated', ms = 2)
plt.plot(xposnew,intermednew,'.',label='New interpolated')
plt.legend()

#%%
ikid = 0
plt.plot(xpos, intermed, label='Original, ikid = '+str(ikid))
plt.plot(xpos, intermed2_arr[ikid,:],'o',label='Original interpolated')
plt.plot(xposnew,intermednew_arr[ikid,:],'.',label='New interpolated')
plt.legend()

#%% FT interfergrams to spectra, replace places with freqbase > 100 to 0, 
# and get new baseline for interfergrams, then FT the new interfergram  

specbase =  np.fft.fft(np.array(intermednew,dtype=np.complex))
posbase = np.abs(freqbase) > 100.0 
#specbase_m = ma.array(specbase, mask = posbase, fill_value=0)
specbase[posbase] = 0 # later: use masks
internewbase =  np.fft.ifft(specbase)

# Apply to all
specbase_arr =  np.fft.fft(np.array(intermednew_arr,dtype=np.complex),axis=1)
specbase_arr[:,posbase]=0.0
internewbase_arr =  np.fft.ifft(specbase_arr,axis=1)

#FT spectra use the new baseline. 
specnew = np.fft.fft(-1.0*np.array(np.roll(intermednew-internewbase,-nzero),dtype=np.complex))[0:len(xposnew)//2].real
specnew_arr = np.fft.fft(-1.0*np.array(np.roll(intermednew_arr-internewbase_arr,-nzero,axis=1),dtype=np.complex),axis=1)[:,0:len(xposnew)//2].real

#%% Plot spectra. 
frt = freq[0:len(xposnew)//2] #xposnew here is twice as large as frt, so frt = freq. 
spt = specnew[0:len(xposnew)//2]
spt  = spt.real
#%%
#frt = freq[len(xposnew)//2:]
#spt = specnew[len(xposnew)//2:]
#pt  = spt.real
#%%

lp = (frt > 90) & (frt< 350)

sptmax = np.max(spt[lp])

sptmed = medfilt(spt/sptmax,3)
sptmed /= sptmed.max()

plt.figure()
#plt.plot(frt,spt/sptmax,'r')                                                                                                                                             
#plt.plot(frt,spt/sptmax,'r.')                                                                                                                                            
plt.plot(frt,sptmed)
plt.plot(frt,sptmed,'ro')

#plt.plot(frt,smooth(spt/sptmax,5))                                                                                                                                       

plt.xlabel('Frequency [GHz]')
plt.ylabel('Normalized Transmission')
plt.xlim([50,350])
#plt.ylim([-0.2,1.1])
plt.hlines(0.0,50,400,linewidth=2,linestyles='dashed')

#%%

wfreqok = (freq > 100.0) & (freq < 400.0)
fig5,ax5 = plt.subplots(1,1,figsize=(8,6))

w_okdet = np.arange(20) + 103
for idet in w_okdet:
 #   ax5.plot(freq,specnew_arr[idet,:]/np.max(specnew_arr[idet,:]),label=data.kidpar['name'][idet])                                                                       
     ax5.plot(freq,specnew_arr[idet,:]/np.max(specnew_arr[idet,wfreqok]), label=kiss.kidpar['namedet'][idet])
ax5.set_xlabel('Frequency [GHz]')
ax5.set_ylabel('Transmission [nomalized units]')
#ax5.set_title(kiss_run)                                                                                                                                                  
ax5.set_xlim([70,400])
ax5.legend(ncol=10,fontsize=4)
#fig5.savefig('spectra_all_det.jpeg')


#return