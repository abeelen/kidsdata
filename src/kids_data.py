#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:47:51 2019

@author: ycao
"""
#import astropy

import numpy as np 
from . import read_kidsdata
from . import kids_calib
from . import kids_plots

class KidsData(object):
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

        
class KidsRawData(KidsData):
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
    read_data(list_data = 'all')
	List selected data. 
    """
    
    def __init__(self, filename):
        self.filename = filename
        info = read_kidsdata.read_info(self.filename)
        self.header, self.version_header, self.param_c, self.kidpar, self.names, self.nsamples = info
        self.ndet = len(self.kidpar[~self.kidpar['index'].mask]) # Number of detectors. 
        self.nptint = self.header.nb_pt_bloc # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms


        # Minimum dataset
        self.I = None
        self.Q = None
        self.A_masq = None

        
    def listInfo(self):
        print ("KISS RAW DATA")
        print ("==================")
        print ("File name: " + self.filename)
       
    def __len__(self):
        return self.nsamples

    def read_data(self, *args, **kwargs):
        self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd \
            = read_kidsdata.read_all(self.filename, *args, **kwargs)
        
        if 'indice' in self.__dataSc.keys():
            assert self.nptint == np.int(self.__dataSc['indice'].max() - self.__dataSc['indice'].min()+1), \
                                        "Problem with 'indice' or header"
        
        for ckey in self.__dataSc.keys():
            self.__dict__[ckey] = self.__dataSc[ckey]
        for dkey in self.__dataSd.keys():
            self.__dict__[dkey] = self.__dataSd[dkey]
                    
    def calib_raw(self, *args, **kwargs):
        assert (self.I is not None) & \
            (self.Q is not None) & \
            (self.A_masq is not None), "I, Q or A_masq data not present"

        self.calfact, self.Icc, self.Qcc,\
            self.P0, self.R0, self.kidfreq = kids_calib.get_calfact(self, *args, **kwargs)
            
    def calib_plot(self, *args, **kwargs):
        return kids_plots.calibPlot(self, *args, **kwargs)

    def check_pointing(self, *args, **kwargs):
        """ Plot azimuth and elevation to check pointing."""
        return kids_plots.checkPointing(self, *args, **kwargs)
    