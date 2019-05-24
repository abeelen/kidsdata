#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May 24 14:12:42 2019
"""


datadir = '/Users/yixiancao/Work/Concerto/Data/kissRaw/'
pltdir = '/Users/yixiancao/workCodes/kidsdata/plots/'

#filename = 'X_2018_12_13_19h09m31_AA_man' # This one show erros for header reading. 
#filename = 'X_2018_12_14_11h58m14_AA_man'
#filename = 'X_2018_12_14_11h55m15_AA_man'
filename = 'X20190427_0910_S0319_Moon_SCIENCEMAP' 
filename = datadir + filename
kiss = kids_data.KissRawData(filename)
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


#%% Need calibrated data
dint_pos = self.kidfreq[:,:,npos:nneg]
dint_neg = self.kidfreq[:,:,nneg:]


#%%---Compute and correct positions of MP  

# Initial position of MP
mpos1 = dlaser1_pos[0,:]
mpos2 = dlaser2_pos[0,:]

dmpos1 = np.diff(mpos1)
dmpos2 = np.diff(mpos2)

#return