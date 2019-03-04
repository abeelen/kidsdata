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
    """ Kid parameters
    """
    __slots__ = (acqbox, flag, freq, level, name, num, typedet, width)

    def __init__(self, filename):
        self.filename = filename
        super().read(filename)
        
      
class Param(object):
    """ Global Parameters
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
              'l-baseFreq',
              'l-modulFreq',
              'A_enable',
              'A_data_ip',
              'A_modulDelay',
              'A_calib',
              'A_sRF_ip',
              'A_sRF_port',
              'freqacq',
              'A_maxbin',
              'A_nb_det',
              'A_nb_bande',
              'A_tone_bande',
              'A_att_inj',
              'A_att_mes',
              'A_gain_dac1',
              'A_gain_dac2',
              'A_gain_dac3',
              'A_gain_dac4',
              'A_gain_dac5',
              'B_enable',
              'B_data_ip',
              'B_modulDelay',
              'B_calib',
              'B_sRF_ip',
              'B_sRF_port',
              'B_maxbin',
              'B_nb_det',
              'B_nb_bande',
              'B_tone_bande',
              'B_att_inj',
              'B_att_mes',
              'B_gain_dac1',
              'B_gain_dac2',
              'B_gain_dac3',
              'B_gain_dac4',
              'B_gain_dac5',
              'C_enable',
              'C_data_ip',
              'C_mcm_start_pos',
              'C_mcm_ampli',
              'C_mcm_equi',
              'D_enable',
              'D_data_port',
              'E_enable',
              'E_data_ip',
              'E_data_port',
              'F_enable',
              'F_data_ip',
              'F_data_port',
              'G_enable',
              'G_data_ip',
              'G_data_port',
              'G_idx_j1',
              'G_idx_j2',
              'G_idx_j3',
              'G_idx_Jpi')

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

    def __init__(self):
        self.dtype = "UdData"

class KissData(object):
    """ General KISS data.
        
    Attributes
    ----------
    
    """

    def __init__(self, filename):
        self.filename = filename
        
class KissRawData(KissData):
    """ KISS raw data.
        
    Attributes
    ----------
    
    Methods
    ----------
    listInfo
    listData()    
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.KidPar = KidPar(super.filename)
        self.Param = Param(super.filename)
        self.ScData = None
        self.SdData = None 
        self.UcData = None
        self.UdData = None
       
    def listInfo(self):
        print ("KISS RAW ")
        print ("==================")
        print ("File name: " + self.filename)
            

    def listData(self, list_data = 'all'):
        self.ScData = ScData().read(list_data = list_data)        
        self.SdData = SdData().read(list_data = list_data)
        self.UcData = UcData().read(list_data = list_data)    
        self.UdData = UdData().read(list_data = list_data)     
        
    def getParam():
        return None
    get_param = staticmethod(get_param)
        
        
