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
#rnd = '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/libreadnikadata.so'
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
class KissInfo(ctypes.Structure):
    _fields_ = [ ("filename", ctypes.c_char_p), 
                 ("p_buffer_header", ctypes.POINTER(ctypes.c_long)), 
                 ("length_header", ctypes.c_int), 
                 ("var_name_buffer", ctypes.c_char_p),
                 ("var_name_length", ctypes.c_int), 
                 ("list_data", ctypes.c_char_p), 
                 ("type_listdet", ctypes.c_int), 
                 ("listdet", ctypes.POINTER(ctypes.c_long)),
                 ("nb_Sc", ctypes.POINTER(ctypes.c_long)), 
                 ("nb_Sd", ctypes.POINTER(ctypes.c_long)),
                 ("nb_Uc", ctypes.POINTER(ctypes.c_long)),
                 ("nb_Ud", ctypes.POINTER(ctypes.c_long)),
                 ("idx_param_c", ctypes.POINTER(ctypes.c_long)), 
                 ("idx_param_d", ctypes.POINTER(ctypes.c_long)),
                 ("silent", ctypes.c_int)]
    
    def ptint(self, length_header=130000, nb_max_det=8001,
                 nb_char_nom = 16, nom_var_all_length = 200000, 
                 **kwargs):
        buffer_header = 0
        self.p_buffer_header = ctypes.cast(buffer_header,
                                           ctypes.POINTER(ctypes.c_long))
        return
# Later: make args and return of Ctype as structures? 
#%%

#%%    

def readKissData(filename, list_data=sel_listdata(), det2red = 'KID', 
                   silent = True, nodata = True):
    """ Read KISS data. 
    

    """
    
    filename = bytes(str(filename), 'ascii')
    list_data  = bytes(str(list_data), 'ascii')
    #global readnikadata
    length_header = 130000
    nb_max_det = 8001

    kiss = KissInfo(filename = filename, list_data = list_data,
                    length_header = length_header)
    
    kiss.ptint()
    
    libpath =  '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C'
    rnd = ctypes.cdll.LoadLibrary(libpath+'/libreadnikadata.so')

    
    buffer_header = np.zeros(length_header,dtype=np.long)
    hptr = buffer_header.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    #header pointer
    #kiss.buffer_header = kiss.length_header * 
    
    listdet = np.zeros(nb_max_det,dtype=np.long)
    ldptr = listdet.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    # detector pointer

    nb_total_samples = np.full(1,-1, dtype=np.int)
#    nbtsptr = nb_total_samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) 
    # sample pointer
    
    nb_char_nom= 16
    nom_var_all_length = 200000
    nom_var_all= np.zeros(nb_char_nom*nom_var_all_length,dtype=np.uint8)
    nvaptr =  nom_var_all.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
    # datanames pointer
    
    nb_Sc= np.full(1,-1, dtype=np.long)
    nb_Scptr = nb_Sc.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    nb_Sd= np.full(1,-1, dtype=np.long)
    nb_Sdptr = nb_Sd.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    nb_Uc= np.full(1,-1, dtype=np.long)
    nb_Ucptr = nb_Uc.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    nb_Ud= np.full(1,-1, dtype=np.long)
    nb_Udptr = nb_Ud.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    # Number of the dataset pointers. 

    
    idx_param_c = np.full(1,-1, dtype=np.long)
    idxpcptr = idx_param_c.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    idx_param_d = np.full(1,-1, dtype=np.long)
    idxpdptr = idx_param_d.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    # param_c and param_d pointers.

#    rnd.Py_read_infos.argtypes = [ctypes.c_char_p,
#                                  ctypes.POINTER(ctypes.c_long), 
#                                  ctypes.c_int, 
#                                  ctypes.c_char_p,
#                                  ctypes.c_int, 
#                                  ctypes.c_char_p, 
#                                  ctypes.c_int, 
#                                  ctypes.POINTER(ctypes.c_long),
#                                  ctypes.POINTER(ctypes.c_long), 
#                                  ctypes.POINTER(ctypes.c_long),
#                                  ctypes.POINTER(ctypes.c_long),
#                                  ctypes.POINTER(ctypes.c_long),
#                                  ctypes.POINTER(ctypes.c_long), 
#                                  ctypes.POINTER(ctypes.c_long),
#                                  ctypes.c_int
#                                  ]

    nb_read_samples = rnd.Py_read_infos(kiss.filename,
                                        kiss.p_buffer_header,
                                        kiss.length_header,
                                        nvaptr,
                                        nom_var_all_length,
                                        kiss.list_data,
                                        codelistdet(det2red),
                                        ldptr,
                                        nb_Scptr,
                                        nb_Sdptr,
                                        nb_Ucptr,
                                        nb_Udptr,
                                        idxpcptr,
                                        idxpdptr,
                                        silent)

#    nb_det = listdet[0] 
#    nb_ts = nb_total_samples[0]
#    
#    buffer_header = 
#    
#    nb_param_c = buffer_header[13]
#    print (nb_det, nb_ts, nb_param_c)
#    
#    return buffer_header,nb_read_samples
    return kiss, nb_read_samples

#%%%
dir_data = '/data/KISS/Raw/'
dateval = '2018_12_13_19h09m31'
filename = dir_data + 'X_'+dateval+'_AA_man'
test, nb_sample =  readKissData(filename,silent=False,
                       list_data='all', nodata=True)
