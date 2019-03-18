#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:47:51 2019

@author: ycao
"""
#import astropy

import numpy as np 
from . import read_kidsdata
from . import kiss_calib
from . import kiss_plots

class KissData(object):
    """ General KISS data.
        
    Attributes
    ----------
    filename: str
    	KISS data file name. 
    	
    """

    def __init__(self, filename):
        self.filename = filename
    
    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.filename)

        
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
        if '__data_Sc' in self.__dict__.keys():
            for ckey in self._dataSc.keys():
                self.__dict__[ckey] = None
        if '__data_Sd' in self.__dict__.keys():
            for dkey in self._dataSd.keys():
                self.__dict__[dkey] = None
        self.__dataSc = None
        self.__dataSd = None
        self.__dataUc = None
        self.__dataUd = None
        
        
    def listInfo(self):
        print ("KISS RAW DATA")
        print ("==================")
        print ("File name: " + self.filename)
       
    def __len__(self):
        return self.nsamples

    def read_data(self, *args, **kwargs):
        self.emptyData()
        
        self.__dataSc, self.__dataSd,self.__dataUc, self.__dataUd = read_kidsdata.read_all(self.filename, *args, **kwargs)
        self.nptint = np.int(np.max(self.__dataSc['indice'])+1)
        self.nint = np.int(self.nsamples/self.nptint)  # number of interferograms
        
        for ckey in self.__dataSc.keys():
            self.__dict__[ckey] = self.__dataSc[ckey]
        for dkey in self.__dataSd.keys():
            self.__dict__[dkey] = self.__dataSd[dkey]
                    
    def calib_raw(self, **args):
        # Exeptions: data needed for the calibration have not been imported yet. 
        self.calfact, self.Icc, self.Qcc,\
            self.P0, self.R0, self.kidfreq = kiss_calib.get_calfact(self, **args)
            
    def calib_plot(self, *args, **kwargs):
        return kiss_plots.calibPlot(self, *args, **kwargs)
