#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:55:54 2019

@author: ycao
"""
import numpy as np
import ctypes
#import gc
#import os
#import string

#libpath = './NIKA_lib_AB_OB_gui/Readdata/C'
#readnikadata = '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/libreadnikadata.so'
#%%

def codelistdet(det2red):
    switcher={'ALL':1,
              'KOD':2,
              'KID':3, 
              'A1': 4,
              'A2': 5,
              'A3': 6                
             }
    return switcher.get(det2red,-1)

def sel_listdata():
    import string
    list_data = 'sample subscan scan El retard 0 ofs_Az ofs_El Az Paral scan_st obs_st MJD LST flag k_flag k_angle k_width I Q dI dQ F_tone dF_tone RF_didq'
    boxes = string.ascii_uppercase[0:20]
    for box in boxes:
        list_data = list_data + ' '+box+'_o_pps'
        list_data = list_data + ' '+box+'_pps'
        list_data = list_data + ' '+box+'_t_utc'
        list_data = list_data + ' '+box+'_freq'
        list_data = list_data + ' '+box+'_masq'
        list_data = list_data + ' '+box+'_n_inf'
        list_data = list_data + ' '+box+'_n_mes'
        
    list_data = list_data + ' antxoffset anttrackAz'
    #anttrackAz anttrackEl antyoffset'     
    return list_data

#%%

def readKissData(filename, list_data=sel_listdata(), det2red = 'KID', 
                   silent = True, nodata = True):
    """ Read KISS data. 
        
    """
    #global readnikadata
    
    libpath =  '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C'
    readnikadata = ctypes.cdll.LoadLibrary(libpath+'/libreadnikadata.so')

    
    length_header = 130000
    buffer_header = np.zeros(length_header,dtype=np.long)
    hptr = buffer_header.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    # head pointer
    
    nb_max_det = 8001
    listdet = np.zeros(nb_max_det,dtype=np.long)
    ldptr = listdet.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    # detector pointer

    nb_total_samples = np.zeros(1,dtype=np.int)
    nbtsptr = nb_total_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    # sample pointer
    
    nb_char_nom= 16
    nom_var_all_length = 200000
    #

    readnikadata.Py_read_start(filename,list_data,length_header,
                               hptr,codelistdet(det2red),ldptr,nbtsptr,silent)

    nb_det = listdet[0]
    nb_ts = nb_total_samples[0]
    
    nb_param_c = buffer_header[13]
    
    return buffer_header

#%%%
dir_data = '/data/KISS/Raw/'
dateval = '2018_12_13_19h09m31'
filename = dir_data + 'X_'+dateval+'_AA_man'
test =  read_kiss_data(filename,silent=False,
                       list_data='all', nodata=True)
