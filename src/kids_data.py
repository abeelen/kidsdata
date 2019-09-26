#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import astropy
import warnings
import numpy as np
from pathlib import Path
from copy import copy

from functools import lru_cache
from itertools import chain

from dateutil.parser import parse
from astropy.time import Time
from astropy.table import join

from . import read_kidsdata
from . import kids_log


class KidsData(object):
    """General KISS data.

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
    """ Arrays of (I,Q) with associated information from KIDs raw data.

    Attributes
    ----------
    filename: str
        Name of the raw data file.
    obs: dict
        Basic information in observation log.
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
        self.header, self.version_header, self.param_c, self._kidpar, self.names, self.nsamples = info
        self.list_detector = np.where(~self._kidpar["index"].mask)[0]
        self._extended_kidpar = None

        self.logger = kids_log.history_logger(self.__class__.__name__)

    def __len__(self):
        return self.nsamples

    @property
    def ndet(self):
        return len(self.list_detector)

    @property
    @lru_cache(maxsize=1)
    def obsdate(self):
        """Return the obsdate of the observation, based on filename."""
        date = Path(self.filename).name[1:].split("_")[0]
        return Time(parse(date), scale="utc", out_subfmt="date")  # UTC ??

    @property
    @lru_cache(maxsize=1)
    def scan(self):
        """Return the scan number of the observation, based on filename."""
        return int(Path(self.filename).name[1:].split("_")[2][1:])

    @property
    @lru_cache(maxsize=1)
    def source(self):
        """Return the source name, based on filename."""
        return Path(self.filename).name[1:].split("_")[3]

    @property
    @lru_cache(maxsize=1)
    def obstype(self):
        """Return the observation type, based on filename."""
        return Path(self.filename).name[1:].split("_")[4]

    def info(self):
        print("RAW DATA")
        print("==================")
        print("File name:\t" + self.filename)
        print("------------------")
        print("Source name:\t" + self.source)
        print("Observed date:\t" + self.obsdate.iso)
        print("Description:\t" + self.obstype)
        print("Scan number:\t" + str(self.scan))

        print("------------------")
        print("No. of KIDS detectors:\t", self.ndet)
        print("No. of time samples:\t", self.nsamples)

    def read_data(self, *args, **kwargs):
        nb_samples_read, self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd = read_kidsdata.read_all(
            self.filename, *args, **kwargs
        )

        if "list_detector" in kwargs:
            self.list_detector = kwargs["list_detector"]

        if self.nsamples != nb_samples_read:
            self.nsamples = nb_samples_read

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read some are computed...
    def __check_attributes(self, attr_list, dependancies=None):
        """Check if attributes have been read, read them if needed.

        Parameters
        ----------
        attr_list: list
            list of parameter to check
        dependancies: list of tuple
            list the parameters dependancies

        Returns
        -------
        dependancies : list
            if the dependancies keyword is set, return the list of
            requested attributes for each dependancy (see Notes)

        Notes
        -----
        If some parameters checks are inter-dependant, you can declare
        them in the dependancies list. For example, any calibrated
        quantity depends on the raw data themselves, thus, when
        checking on 'P0' or 'R0', one should actually check for 'I', 'Q' and
        'A_masq', this dependancy can be described with :

        dependancies = [(['P0', 'R0'], ['I', 'Q', 'A_masq'])]

        """
        # Adapt the list if we have dependancies
        _dependancies = []
        if dependancies:
            for request_keys, depend_keys in dependancies:
                # Check if these attributes are requested...
                _dependancy = [attr_list.pop(attr_list.index(key)) for key in request_keys if key in attr_list]
                if _dependancy:
                    # Replace them by the dependancies..
                    attr_list += depend_keys
                _dependancies.append(_dependancy)

        missing = [attr for attr in attr_list if not hasattr(self, attr) or (getattr(self, attr) is None)]
        if missing:
            # TODO: check that there attributes are present in the file
            list_data = " ".join(missing)
            print("Missing data : ", list_data)
            print("-----Now reading--------")

            self.read_data(list_data=list_data, list_detector=self.list_detector)

        # Check that everything was read
        for key in attr_list:
            assert hasattr(self, key) & (getattr(self, key) is not None), "Missing data {}".format(key)

        # Check in _dependancies that everything was read, if not we are missing something
        if _dependancies:
            missing = [
                attr for attr in chain(*_dependancies) if not hasattr(self, attr) or (getattr(self, attr) is None)
            ]
            return missing or None
        return None

    def get_list_detector(self, pattern=None):
        """Retrieve the valid detector list given a pattern.

        Attributes
        ----------
        pattern: str
            any string pattern a KID name should match in a `in` operation

        Returns
        -------
        list_detector: list
            the list which should be used for the `.read_data()` method
        """
        list_detector = np.where(~self._kidpar["index"].mask & [pattern in name for name in self._kidpar["namedet"]])[0]
        return list_detector

    @property
    def kidpar(self):
        """Retrieve the kidpar of the observation.

        Notes
        -----
        If there is an extended kidpar, it will be merge with the data kidpar

        """
        if self._extended_kidpar:
            kidpar = copy(self._kidpar)
            # astropy.table.join fails when index is masked so..
            mask = kidpar["index"].mask
            kidpar["index"].mask = False
            kidpar = join(kidpar, self._extended_kidpar, join_type="outer", keys="index")
            kidpar["index"].mask = mask
        else:
            kidpar = self._kidpar

        return kidpar
