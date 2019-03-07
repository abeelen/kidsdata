#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:55:54 2019

@author: Yixian Cao 

"""

import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

#import gc
#import os
#import string

#libpath = './NIKA_lib_AB_OB_gui/Readdata/C'
#rnd = '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/libreadnikadata.so'


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
class KissInfo(object):
    
    def __init__(self, filename, list_data = sel_listdata(),
                 length_header = 130000, nb_max_det = 8001, 
                 nb_char_nom= 16, nom_var_all_length = 200000):
        self.filename = bytes(str(filename), 'ascii')
        self.list_data  = bytes(str(list_data), 'ascii')
        self.length_header = length_header
        self.nb_max_det = nb_max_det
        self.nb_char_nom = nb_char_nom
        self.nom_var_all_length = nom_var_all_length
        
        self.buffer_header = np.empty(self.length_header,dtype=np.long)
        self.listdet = np.empty(self.nb_max_det,dtype=np.long)
        self.nom_var_all= np.empty(self.nb_char_nom*
                                   self.nom_var_all_length,
                                   dtype=np.uint8)    
        self.nb_Sc = np.empty(1,dtype=np.long)
        self.nb_Sd = np.empty(1,dtype=np.long)
        self.nb_Uc = np.empty(1,dtype=np.long)
        self.nb_Ud = np.empty(1,dtype=np.long)
        self.idx_param_c = np.empty(1,dtype=np.long)
        self.idx_param_d = np.empty(1,dtype=np.long)
        
    
    def readIn(self, det2red = 'KID', silent = False):
        """ Read KISS info to the buffer. 

        """
        
        libpath =  '../NIKA_lib_AB_OB_gui/Readdata/C'
        nlib = ctypes.cdll.LoadLibrary(libpath+'/libreadnikadata.dylib')

        # param_c and param_d pointers.

        nlib.Py_read_infos.restype = ctypes.c_long
        nlib.Py_read_infos.argtypes = [ctypes.c_char_p,
                                       ndpointer(ctypes.c_long), 
                                       ctypes.c_int, 
                                       ndpointer(ctypes.c_uint8),
                                       ctypes.c_int, 
                                       ctypes.c_char_p, 
                                       ctypes.c_int, 
                                       ndpointer(ctypes.c_long),
                                       ndpointer(ctypes.c_long), 
                                       ndpointer(ctypes.c_long),
                                       ndpointer(ctypes.c_long),
                                       ndpointer(ctypes.c_long),
                                       ndpointer(ctypes.c_long), 
                                       ndpointer(ctypes.c_long),
                                       ctypes.c_int
                                       ]

        nb_read_samples = nlib.Py_read_infos(self.filename,
                                        self.buffer_header,
                                        self.length_header,
                                        self.nom_var_all,
                                        self.nom_var_all_length,
                                        self.list_data, 
                                        codelistdet(det2red),
                                        self.listdet,
                                        self.nb_Sc,
                                        self.nb_Sd,
                                        self.nb_Uc,
                                        self.nb_Ud,
                                        self.idx_param_c,
                                        self.idx_param_d,
                                        silent)

        return nb_read_samples
    
    def printinfo(self):
        print ('pname')

# Later: make args and return of Ctype as structures? 



#%%
dir_data = '../../Data/kissRaw/'
dateval = '2018_12_13_19h09m31'
filename = dir_data + 'X_'+dateval+'_AA_man'
kiss =  KissInfo(filename)
nb_sampe = kiss.readIn()
