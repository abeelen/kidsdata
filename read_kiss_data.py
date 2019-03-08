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

# Transform byte into string                                                                                                                                    
# Use to retrieve names of parameters, detectors and data                                                                                                       
def charvect2string(nbchar,vect_char):
    """                                                                                                                                                         
    Transform byte variable into string format                                                                                                                  
    """
    str=""
    for idx in range(nbchar):
        if (vect_char[idx] < 0):
            print ("Problems with names")
        if (vect_char[idx]==0):
            break
        else:
            dummy = "%c" %(vect_char[idx])
#        if dummy != '\x00' and dummy != '\x7f':                                                                                                                
            str+=dummy
    return str

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
    """ Kiss header infomation 
        Attributes
        ----------
        Methods
        -------
    """
    
    def __init__(self, filename, list_data = sel_listdata(),
                 length_header = 130000, nb_max_det = 8001, 
                 nb_char_nom= 16, nom_var_all_length = 200000):
        self.filename = bytes(str(filename), 'ascii')
        self.list_data  = bytes(str(list_data), 'ascii')
        self.length_header = length_header
        self.nb_max_det = nb_max_det
        self.nb_char_nom = nb_char_nom
        self.nom_var_all_length = nom_var_all_length
        
        self.listdet = np.empty(self.nb_max_det,dtype=np.int32)
        self.nom_var_all= np.empty(self.nb_char_nom * 
                                   self.nom_var_all_length,
                                   dtype=np.uint8)
        self.nvaptr =  self.nom_var_all.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
        self.nb_Sc = np.empty(1,dtype=np.int32)
        self.nb_Sd = np.empty(1,dtype=np.int32)
        self.nb_Uc = np.empty(1,dtype=np.int32)
        self.nb_Ud = np.empty(1,dtype=np.int32)
        self.idx_param_c = np.empty(1,dtype=np.int32)
        self.idx_param_d = np.empty(1,dtype=np.int32)
        
    
    def readIn(self, det2red = 'KID', silent = False):
        """ Read KISS info to the buffer. 

        """
        
        libpath =  '../NIKA_lib_AB_OB_gui/Readdata/C'
        nlib = ctypes.cdll.LoadLibrary(libpath+'/libreadnikadata.dylib')
        
        # param_c and param_d pointers.
        buffer_header = np.empty(self.length_header,dtype=np.long)


        nlib.Py_read_infos.restype = ctypes.c_int32
        nlib.Py_read_infos.argtypes = [ctypes.c_char_p,
                                       ndpointer(ctypes.c_long), 
                                       ctypes.c_int, 
                                       ndpointer(ctypes.c_uint8), 
                                       ctypes.c_int, 
                                       ctypes.c_char_p, 
                                       ctypes.c_int, 
                                       ndpointer(ctypes.c_int32),
                                       ndpointer(ctypes.c_int32), 
                                       ndpointer(ctypes.c_int32),
                                       ndpointer(ctypes.c_int32),
                                       ndpointer(ctypes.c_int32),
                                       ndpointer(ctypes.c_int32), 
                                       ndpointer(ctypes.c_int32),
                                       ctypes.c_int32
                                       ]

        self.nb_read_samples = nlib.Py_read_infos(self.filename,
                                        buffer_header,
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
        
        
        
        

        return self.nb_read_samples
    
    def printinfo(self):
        print ('Number of detectors: ', kiss.listdet[0])

# Later: make args and return of Ctype as structures? 


#%%
dir_data = '../../Data/kissRaw/'
dateval = '2018_12_13_19h09m31'
filename = dir_data + 'X_'+dateval+'_AA_man'
kiss =  KissInfo(filename, list_data = 'all')
nb_sampe = kiss.readIn()
#%%
def testinf(kissInfo):
    nb_detecteurs = kissInfo.listdet[0]

    buffer_header = kissInfo.buffer_header
    nb_boites_mesure = buffer_header[6];
#    nb_detecteurs = buffer_header[7];                                                                                                                                    
    nb_pt_bloc = buffer_header[8];
    nb_param_c = buffer_header[13];
    nb_param_d = buffer_header[14];
    size_motor_module_table = buffer_header[4]
    nb_brut_periode =  buffer_header[18]
#    nb_data_communs = nb_Sc[0] # buffer_header[19];                                                                                                                      
#    nb_data_detecteurs = nb_Sd[0]#buffer_header[20];                                                                                                                     
    nb_data_Sc =  kissInfo.nb_Sc[0]
    nb_data_Sd =  kissInfo.nb_Sd[0]
    nb_data_Uc =  kissInfo.nb_Uc[0]
    nb_data_Ud =  kissInfo.nb_Ud[0]
    
    
    nb_read_samples_U =  kissInfo.nb_read_samples/ nb_pt_bloc

#print nb_data_Sc,  nb_data_Sd,nb_data_Uc,  nb_data_Ud                                                                                                                

    nb_champ_reglage = buffer_header[21];
    version_header  = buffer_header[12]/65536;

    indexdetecteurdebut=0
    nb_detecteurs_lu = nb_detecteurs
    buffer_header[2]= indexdetecteurdebut
    buffer_header[3]= nb_detecteurs_lu
#    buffer_header[7] = nb_detecteurs                                                                                                                                     

#print "Total number of detectors %d " %(nb_detecteurs)                                                                                                               

#  Obtain common and detector data parameters                                                                                                                             

    nbtotdet =  buffer_header[7]


# Reconstructing names of detectors, data and parameters                                                                                                                  
    idxinit= 0
    nom_param_c = kissInfo.nom_var_all[idxinit:idxinit+kissInfo.nb_char_nom * 
                              nb_param_c]
    name_param_c = []
    val_param_c = np.zeros(nb_param_c,dtype=np.int)
    
    for idx in range(nb_param_c):
        name_param_c.append(charvect2string(kissInfo.nb_char_nom,
                                            nom_param_c[idx*kissInfo.nb_char_nom:idx*kissInfo.nb_char_nom+kissInfo.nb_char_nom]))
        val_param_c[idx] = buffer_header[kissInfo.idx_param_c[0]+idx]

    param_c = {'pname':name_param_c, 'pvalue':val_param_c}

    print (param_c)

testinf(kiss)
