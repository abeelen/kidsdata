#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:43:58 2019

@author: yixiancao
"""

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

#%% Calibration 
kiss.read_data()
kiss.calib_raw()

fig = kiss.calib_plot()
fig.savefig(pltdir + 'calib.pdf')

#%% Define time variables for interpolation
datadir = '/Users/yixiancao/Work/Concerto/Data/kissRaw/'
filename = 'X20190427_0910_S0319_Moon_SCIENCEMAP' 
filename = datadir + filename
kiss = kids_data.KissRawData(filename)

list_data = 'A_time_ntp A_time_pps A_time A_hours' 
kiss.read_data(list_data = list_data)
#%%
from src import kids_validate
kids_validate.kids_validate(kiss)

#%% Check pointing 
list_data = 'F_sky_Az F_sky_El F_tl_Az F_tl_El F_diff_Az F_diff_El  F_state'
kiss.read_data(list_data = list_data)

#%% Check pointing 

list_data = 'F_azimuth F_elevation F_state F_subsc F_nbsubsc E_X  E_status u_itfamp'
kiss.read_data(list_data = list_data)
#%%
fig = kids.pointing_plot()
fig.savefig(pltdir + 'pointing.pdf', bbox_inches='tight')
#%% Photometry 
fig = kids.photometry_plot()
fig.savefig(pltdir + 'photometry.pdf')

#%% Beammap
fig = kids.beammap_plot(testikid = 195)
fig.savefig(pltdir + 'beammap.pdf')
