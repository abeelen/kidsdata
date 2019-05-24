#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configure loggers for KIDS pipeline.

Created on Thu May 23 11:36:30 2019

"""
import logging
import sys
#from datetime import datetime

HISTORY = 5

def history(self, message, *args, **kws):
    if self.isEnabledFor(HISTORY):
        self._log(HISTORY, 'PIPELINE HISTORY: '+message, args, **kws) 


FORMATTER = logging.Formatter('%(asctime)s  %(levelname)-10s %(processName)s  %(name)s %(message)s')
LOG_FILE = "kiss_history.log"
# Add more into log_file string later. 
#%%

def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(FORMATTER)
   return console_handler
def get_file_handler(log_file = LOG_FILE):
   file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight')
   file_handler.setFormatter(FORMATTER)
   return file_handler


def history_logger(logger_name, **kwargs):
    logging.addLevelName(HISTORY, 'HISTORY')
    logging.Logger.history = history
    logging.basicConfig(format = FORMATTER)
    l = logging.getLogger(logger_name)
    l.setLevel(HISTORY)
    l.addHandler(get_console_handler())
    l.addHandler(get_file_handler(**kwargs))
    l.propagate = False

    return l
#%% Tests 
#l = history_logger('log_name')
#l.history('test')
