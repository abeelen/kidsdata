#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:47:51 2019

@author: ycao
"""
#import astropy

import read_kidsdata
import numpy as np 
import kiss_calib

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
    kidpar: :obj: Astropy.Table
    	KID parameter. 
    param_c: :dict
    	Global parameters. 
    I: array 
    	Stokes I measured by KID detectors.  
    Q: array 
    	Stokes Q measured by KID detectors.
    __dataSc: obj:'ScData', optional
    	Sample data set. 
    __dataSd: 
        Sample data set. 

    Methods
    ----------
    listInfo()
    	Display the basic infomation about the data file. 
    listData(list_data = 'all')
	List selected data. 
    """
    
    def __init__(self, filename):
#        self.kidPar = KidPar(super.filename)
#        self.param = Param(super.filename)
        self.filename = filename
        self.header, self.version_header, self.param_c, \
            self.kidpar, self.names, self.nsamples \
                = read_kidsdata.read_info(self.filename)
        self.ndet = len(self.kidpar[~self.kidpar['index'].mask])
        
        self.emptyData()

#    @property
#    def cache(self):
#        if not cahche:
            # read the data into the buffer. 
    def emptyData(self):
        # Empty the datasets. 
        if 'data_Sc' in self.__dict__.keys():
            for ckey in self._dataSc.keys():
                self.__dict__[ckey] = None
        if 'data_Sd' in self.__dict__.keys():
            for dkey in self._dataSd.keys():
                self.__dict__[dkey] = None
        self._dataSc = None
        self._dataSd = None
        self._dataUc = None
        self._dataUd = None
        
            
    def listInfo(self):
        print ("KISS RAW DATA")
        print ("==================")
        print ("File name: " + self.filename)
            

    def listData(self, list_data = 'indice A_masq C_laser1_po s C_laser2_pos I Q'):
        self.emptyData()
        
        self._dataSc, self._dataSd, self._dataUc, self._dataUd \
            = read_kidsdata.read_all(self.filename, list_data = list_data)   
        self.nptint = np.int(np.max(self._dataSc['indice'])+1)
        self.nint = np.int(self.nsamples/self.nptint)  # number of interferograms
        
        for ckey in self._dataSc.keys():
            self.__dict__[ckey] = self._dataSc[ckey]
        for dkey in self._dataSd.keys():
            self.__dict__[dkey] = self._dataSd[dkey]
            
        print ('Data listed: ' + list_data.replace(' ', ', ') )
        
    def calibRaw(self):
        self.calfact, self.Icc, self.Qcc,\
            self.P0, self.R0, self.kidfreq  = kiss_calib.get_calfact(self)
        
