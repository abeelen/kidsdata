#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:47:51 2019

@author: ycao
"""
import numpy as np
#import astropy

import read_kiss_data as rkd

from astropy.table import Table

class KidPar(Table):
    """ Kid parameters. 
    Table of Kid detector parameters. 
    
    Attributes
    ----------
    filename: str
    	Name of the file from with the KID parameters are imported. 
    acqbox: ndarray:'float64'
	
    array: ndarray:'flat64'
	Parameter of each KID detector.
    flag: ndarray:'int64'
	Detector flag. 
    level: ndarray:'int64'
	
    name: ndarray:'str'
    	Name of the detector. 
    num: ndarray:'int64'
	
    typedet: ndarray:'int64'
    	Type of the detector. 
    width: ndarray:'int64'
	
    Methods
    -------

    """
    __slots__ = ('acqbox', 'array','flag', 'freq', 'level', 
                 'name', 'num', 'typedet', 'width')

    def __init__(self, filename):
        self.filename = filename
        super().read(filename)
        

      
class Param(object):
    """ Global Parameters 

    Attributes
    ----------
    namexp1: int64
    namexp2: int64
    namexp3: int64
    namexp4: int64
    div_kid: int64
    data_dIdQ: int64
    data_pIpQ: int64
    acqui_ip: int64
    retard: int64
    tuning_seuil: int64
    tuning_ofset: int64
    tuning_moyenne: int64
    l_central_ip: int64
    l_clockCode: int64
    l_syntheCode: int64
    l_baseFreq: int64
    l_modulFreq: int64
    a_enable: int64
    a_data_ip: int64
    a_modulDelay: int64
    a_calib; int64
    a_sRF_ip: int64
    a_sRF_port: int64
    freqacq: int64
    a_maxbin: int64
    a_nb_det: int64
    a_nb_bande: int64
    a_tone_bande: int64
    a_att_inj: int64
    a_att_mes: int64
    a_gain_dac1: int64
    a_gain_dac2: int64
    a_gain_dac3: int64
    a_gain_dac4: int64
    a_gain_dac5: int64
    b_enable: int64
    b_data_ip: int64
    b_modulDelay: int64
    b_calib: int64
    b_sRF_ip: int64
    b_sRF_port: int64
    b_maxbin: int64
    b_nb_det: int64
    b_nb_bande: int64
    b_tone_bande: int64
    b_att_inj: int64
    b_att_mes: int64
    b_gain_dac1: int64
    b_gain_dac2: int64
    b_gain_dac3: int64
    b_gain_dac4: int64
    b_gain_dac5: int64
    c_enable: int64
    c_data_ip: int64
    c_mcm_start_pos: int64
    c_mcm_ampli: int64
    c_mcm_equi: int64
    d_enable: int64
    d_data_port: int64
    e_enable: int64
    e_data_ip: int64
    e_data_port: int64
    f_enable: int64
    f_data_ip: int64
    f_data_port: int64
    g_enable: int64
    g_data_ip: int64
    g_data_port: int64
    g_idx_j1: int64
    g_idx_j2: int64
    g_idx_j3: int64
    g_idx_Jpi: int64
    """
    __slots__ = ('nomexp1',
              'nomexp2',
              'nomexp3',
              'nomexp4',
              'div_kid',
              'data_dIdQ',
              'data_pIpQ',
              'acqui_ip',
              'retard',
              'tuning_seuil',
              'tuning_ofset',
              'tuning_moyenne',
              'l_central_ip',
              'l_clockCode',
              'l_syntheCode',
              'l_baseFreq',
              'l_modulFreq',
              'a_enable',
              'a_data_ip',
              'a_modulDelay',
              'a_calib',
              'a_sRF_ip',
              'a_sRF_port',
              'freqacq',
              'a_maxbin',
              'a_nb_det',
              'a_nb_bande',
              'a_tone_bande',
              'a_att_inj',
              'a_att_mes',
              'a_gain_dac1',
              'a_gain_dac2',
              'a_gain_dac3',
              'a_gain_dac4',
              'a_gain_dac5',
              'b_enable',
              'b_data_ip',
              'b_modulDelay',
              'b_calib',
              'b_sRF_ip',
              'b_sRF_port',
              'b_maxbin',
              'b_nb_det',
              'b_nb_bande',
              'b_tone_bande',
              'b_att_inj',
              'b_att_mes',
              'b_gain_dac1',
              'b_gain_dac2',
              'b_gain_dac3',
              'b_gain_dac4',
              'b_gain_dac5',
              'c_enable',
              'c_data_ip',
              'c_mcm_start_pos',
              'c_mcm_ampli',
              'c_mcm_equi',
              'd_enable',
              'd_data_port',
              'e_enable',
              'e_data_ip',
              'e_data_port',
              'e_enable',
              'e_data_ip',
              'f_data_port',
              'g_enable',
              'g_data_ip',
              'g_data_port',
              'g_idx_j1',
              'g_idx_j2',
              'g_idx_j3',
              'g_idx_Jpi')

    def __init__(self, filename):
        self.filename = filename
        rdk.read_kiss_data(filename,list_data = 'param')

class DataSet(object):
    __slots__ = ('nb_sample')
    def __init__(self, filename):
        self.filename = filename

    def read(self, list_data = list_data):
        new_list = []
        for data_name in list_data:
            new_list.append(data_name)
 
        rdk.readKissData(filename, list_data = new_list)

class ScData(DataSet):
    """ len: nb_samples
    """
    __slots__ = ('sample', 'indice', 'sample_dur', 't_mac', 'A_time', 'A_hours',
       'A_time_ntp', 'A_time_pps', 'A_pulse_pps', 'A_freq', 'A_masq',
       'A_n_inj', 'B_time', 'A_n_mes', 'B_hours', 'B_time_ntp',
       'B_time_pps', 'B_pulse_pps', 'B_freq', 'B_masq', 'C_time',
       'B_n_inj', 'B_n_mes', 'C_hours', 'C_time_ntp', 'C_time_pps',
       'C_pulse_pps', 'C_motor1_step', 'C_motor2_step', 'C_motor1_pos',
       'C_motor2_pos', 'C_laser1_pos', 'C_laser2_pos', 'F_azimuth',
       'F_elevation' )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.type = 'ScData'

    def read(self, list_data = list_data):
        rdk.read_kiss_data(self.filename, list_data = list_data) 
    
class SdData(DataSet):
    """ len: nb_det x nb_samples
    """
    __slots__ = ('I', 'Q', 'RF_deco', 'RF_didq', 'F_tone', 'amplitude', 
                 'logampl', 'ph_IQ', 'ph_rel', 'k_flag', 'sampleU', 'flag')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.dtype = 'SdData'
        self.nb_det = rdk.read_kiss_data(self.filename, list_data = 'nb_det')


class UcData(Dataset):
    __slots__ = ('E_X', 'E_status', 'F_state', 'F_nbsubsc', 'F_subsc', 
                 'G_j1', 'G_j2', 'G_j3', 'G_Jpi' )

    def __init__(self):
        self.dtype = "UcData"

class UdData(Dataset):
    __slots__ = ('freqResp', 'interferoAmp')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.dtype = "UdData"
        self.nb_det = rdk.read_kiss_data(self.filename, list_data = 'nb_det')

class KissData(object):
    """ General KISS data.
        
    Attributes
    ----------
    filename: str
    	KISS data file name. 
    	
    """

    def __init__(self, filename):
        self.filename = filename
        
class KissRawData(KissData):
    """ Arrays of (I,Q) with assiciated information from KISS raw data. 
        
    Attributes
    ----------
    kidPar: :obj:'KidPar'
    	KID parameter. 
    param: :obj:'Param'
    	Param_c 
    I: array 
    	Stokes I measured by KID detectors.  
    Q: array 
	Stokes Q measured by KID detectors.
    ScData: obj:'ScData', optional
    	Sample data set. 
    SdData: 

    Methods
    ----------
    listInfo()
    	Display the basic infomation about the data file. 
    listData(list_data = 'all')
	List selected data. 
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kidPar = KidPar(super.filename)
        self.param = Param(super.filename)
        self.cache = None
        self.I = None
        self.Q = None
        self.ScData = None
        self.SdData = None 
        self.UcData = None
        self.UdData = None

    @property
    def cache(self):
        if not cahche:
            # read the data into the buffer. 
       
    def listInfo(self):
        print ("KISS RAW DATA")
        print ("==================")
        print ("File name: " + self.filename)
            

    def listData(self, list_data = 'all'):
        readKissData(self.filename, ScData, SdData, UcData, UdData, listData)
        # self.ScData = ScData().read()        
        # self.SdData = SdData().read(list_data = list_data)
        # self.UcData = UcData().read(list_data = list_data)    
        # self.UdData = UdData().read(list_data = list_data)     
        

    def getParam():
        return None
    get_param = staticmethod(get_param)
        
        
