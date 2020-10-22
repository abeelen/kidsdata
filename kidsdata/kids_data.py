import os
import re
import numpy as np
import warnings

from pathlib import Path
from copy import deepcopy

from functools import lru_cache
from itertools import chain

from dateutil.parser import parse
from astropy.time import Time
from astropy.table import Table, join
import astropy.units as u

import h5py
from autologging import logged

from .read_kidsdata import read_info, read_all
from .read_kidsdata import read_info_hdf5, read_all_hdf5
from .read_kidsdata import info_to_hdf5, data_to_hdf5

from .db import RE_SCAN, RE_EXTRA

CACHE_DIR = Path(os.getenv("CACHE_DIR", "/data/KISS/Cache"))

if not CACHE_DIR.exists():
    warnings.warn("Cache dir created : {}".format(CACHE_DIR))
    CACHE_DIR.mkdir()


@logged
class KidsRawData(object):
    """Arrays of (I,Q) with associated information from KIDs raw data.

    Attributes
    ----------
    filename: str, or int
        Name of the raw data file. Could also be a scan number
    header : TconfigHeader (nametuple)
        the raw header of the file
    version_header : int
        the version of the header
    param_c: :dict
        common parameters
    kidpar: :obj: Astropy.Table
        KID parameters.
    names : TName (namedtuple)
        the variable names contained in the raw file
    list_detector : array_like
        names of the read detectors
    __dataSc, __dataSd: dict
        Fully sampled common and detector data
    __dataUc, __dataUd: dict
        Undersampled common and detector data

    Methods
    -------
    listInfo()
        Display the basic infomation about the data file.
    read_data(list_data = 'all')
        List selected data.

    Notes
    -----
    All read data `__dataSc, __dataSd, __dataUc, __dataUd` are linked as top level attributes.
    """

    def __init__(self, filename):

        if isinstance(filename, (int, np.int)):
            from .db import get_scan  # To avoid circular import

            filename = get_scan(filename)

        self.filename = Path(filename)

        info = read_info_hdf5(self.filename) if h5py.is_hdf5(self.filename) else read_info(self.filename)

        self.header, self.version_header, self.param_c, _kidpar, self.names, self.nsamples = info

        # insure that kidpar is a masked table for kidpar and as an index
        if "index" not in _kidpar.keys():
            _kidpar["index"] = np.arange(len(_kidpar))

        if not _kidpar.has_masked_values:
            _kidpar = Table(_kidpar, masked=True, copy=False)

        _kidpar.add_index("namedet")
        self._kidpar = _kidpar

        self.list_detector = np.array(self._kidpar[~self._kidpar["index"].mask]["namedet"])

        self._extended_kidpar = None
        self.__dataSc = {}
        self.__dataSd = {}
        self.__dataUc = {}
        self.__dataUd = {}

        # TODO: Need to decide what file structure we want, flat, or by dates
        self._cache = None
        self._cache_filename = (
            self.filename if h5py.is_hdf5(self.filename) else CACHE_DIR / self.filename.with_suffix(".hdf5").name
        )

        if self._cache_filename.exists():
            self.__log.info("Cache file found")
            # Need to keep it as open file if we use dask arrays (memmap)
            self._cache = h5py.File(self._cache_filename, "r")

    def __len__(self):
        return self.nsamples

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.filename)

    @property
    def ndet(self):
        return len(self.list_detector)

    @property
    @lru_cache(maxsize=1)
    def obsdate(self):
        """Return the obsdate of the observation, based on filename."""
        filename = self.filename.name
        if RE_SCAN.match(filename):
            date = RE_SCAN.match(filename).groups()[0]
            return Time(parse(date), scale="utc")  # UTC ?
        elif RE_EXTRA.match(filename):
            date = "".join(RE_EXTRA.match(filename).groups()[0:3])
            return Time(parse(date), scale="utc")  # UTC ?
        else:
            # Probably not a data scan
            self.__log.error("No time on filename")
            return Time.now()

    @property
    @lru_cache(maxsize=1)
    def obstime(self):
        """Recompute the proper obs time in UTC per interferograms."""
        times = ["time"]
        self.__check_attributes(times)

        time = self.time
        obstime = self.obsdate

        # Getting only median time per interferograms here :
        return obstime + np.median(time.data.reshape((self.nint, self.nptint)), axis=1) * u.s

    @property
    def exptime(self):
        return (self.obstime[-1] - self.obstime[0]).to(u.s)

    @property
    @lru_cache(maxsize=1)
    def scan(self):
        """Return the scan number of the observation, based on filename."""
        filename = self.filename.name
        if RE_SCAN.match(filename):
            return int(RE_SCAN.match(filename).groups()[2])
        else:
            self.__log.warning("No scan from filename")
            return None

    @property
    @lru_cache(maxsize=1)
    def source(self):
        """Return the source name, based on filename."""
        filename = self.filename.name
        if RE_SCAN.match(filename):
            return RE_SCAN.match(filename).groups()[3]
        else:
            self.__log.warning("No source name from filename")
            return None

    @property
    @lru_cache(maxsize=1)
    def obstype(self):
        """Return the observation type, based on filename."""
        return self.filename.name[1:].split("_")[4]

    def info(self):
        print("RAW DATA")
        print("==================")
        print("File name:\t" + str(self.filename))
        print("------------------")
        print("Source name:\t" + self.source)
        print("Observed date:\t" + self.obsdate.iso)
        print("Description:\t" + self.obstype)
        print("Scan number:\t" + str(self.scan))

        print("------------------")
        print("No. of KIDS detectors:\t", self.ndet)
        print("No. of time samples:\t", self.nsamples)

    def read_data(self, *args, cache=False, array=np.array, list_data=["indice", "A_masq", "I", "Q"], **kwargs):
        """Read raw data.

        Parameters
        ----------
        cache : bool or 'only', optional
            use the cache file if present, by default False, see Notes
        array : function, (np.array|dask.array.from_array|None) optional
            function to apply to the largest cached value, by default np.array, if None return h5py.Dataset
        list_data : list, optional
            list of data to read, by default ["indice", "A_masq", "I", "Q"]
        **kwargs
            additionnal parameters to be passed to the  `kidsdata.read_kidsdata.read_all`, in particular
                list_detector : list, optional
                    the list of detector indexes to be read, by default None: read all detectors
                start : int
                    the starting block, default 0.
                end : type
                    the ending block, default full available dataset.
        Notes
        -----
        if `cache=True`, the function reads all possible data from the cache file, and read the missing data from the raw binary file
        if `cache='only'`, the function reads all possible data from the cache file
        """
        self.__log.debug("Reading data")

        if cache and self._cache is not None:
            self.__log.info("Reading cached raw data :")

            *datas, extended_kidpar = read_all_hdf5(self._cache, array=array)

            dataSc, dataSd, dataUc, dataUd = datas

            self.__log.debug("Updating dictionnaries with cached data")
            self.__dataSc.update(dataSc)
            self.__dataSd.update(dataSd)
            self.__dataUc.update(dataUc)
            self.__dataUd.update(dataUd)

            self._extended_kidpar = extended_kidpar

            # Removing cache data from the requested list_data:
            if list_data == "all":
                list_data = self.names.DataSc + self.names.DataSd + self.names.DataUc + self.names.DataUd

            keys = [key for data in datas for key in data if key in list_data]

            self.__log.debug("Read cached raw data : {}".format(keys))
            for key in keys:
                list_data.remove(key)
            self.__log.debug("Remaining raw data : {}".format(list_data))

            # TODO: list_detectors and nsamples

        if cache != "only" and list_data and not h5py.is_hdf5(self.filename):
            self.__log.debug("Reading raw data")

            nb_samples_read, *datas = read_all(self.filename, *args, list_data=list_data, **kwargs)

            dataSc, dataSd, dataUc, dataUd = datas
            self.__log.debug("Updating dictionnaries")
            self.__dataSc.update(dataSc)
            self.__dataSd.update(dataSd)
            self.__dataUc.update(dataUc)
            self.__dataUd.update(dataUd)

            if "list_detector" in kwargs:
                self.list_detector = np.asarray(kwargs["list_detector"])
            else:
                self.list_detector = np.asarray(self.names.RawDataDetector)

            if self.nsamples != nb_samples_read:
                self.__log.warning("Read less sample than expected : {} vs {}".format(nb_samples_read, self.nsamples))
                self.nsamples = nb_samples_read

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]

    def _write_data(self, filename=None, dataSd=False, mode="a", file_kwargs=None, **kwargs):
        """write internal data to hdf5 file

        Parameters
        ----------
        filename : [str]
            output filename, default None, use the cache file name
        dataSd : bool, optional
            flag to output the fully sampled data, by default False
        mode : str
            the open mode for the h5py.File object, by default 'a'
        file_kwargs, dict, optionnal
            additionnal keyword for h5py.File object, by default None
        **kwargs
            additionnal keyword argument for the h5py.Dataset, see Note

        Notes
        -----
        Usual kwargs could be :

        kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}

        """
        if filename is None:
            filename = self._cache_filename

        _file_kwargs = {"mode": mode}
        if file_kwargs is not None:
            _file_kwargs.update(**file_kwargs)

        self.__log.debug("Saving info")
        info_to_hdf5(
            filename,
            self.header,
            self.version_header,
            self.param_c,
            self._kidpar,
            self.names,
            self.nsamples,
            file_kwargs=_file_kwargs,
            **kwargs
        )

        self.__log.debug("Saving data")
        _file_kwargs["mode"] = "a"
        data_to_hdf5(
            filename,
            self.__dataSc,
            self.__dataSd if dataSd else None,
            self.__dataUc,
            self.__dataUd,
            self._extended_kidpar,
            file_kwargs=_file_kwargs,
            **kwargs
        )

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read some are computed...
    def __check_attributes(self, attr_list, dependancies=None, read_missing=True):
        """Check if attributes have been read, read them if needed.

        Parameters
        ----------
        attr_list: list
            list of parameters to check
        dependancies: list of tuple
            list the parameters dependancies
        read_missing : bool
            read the missing parameters (default: True)

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
        if dependancies is None:
            dependancies = []

        dependancies += [
            # hours will need at least some data, default first time
            (["time"], ["A_time_pps", "A_hours"]),
        ]

        # First check if some attributes are indeed missing...
        missing = [attr for attr in attr_list if not hasattr(self, attr) or (getattr(self, attr) is None)]

        # Adapt the list if we have dependancies
        _dependancies = []
        if dependancies and missing:
            for request_keys, depend_keys in dependancies:
                # Check if these attributes are requested...
                _dependancy = [missing.pop(missing.index(key)) for key in request_keys if key in missing]
                if _dependancy:
                    # Replace them by the dependancies..
                    missing += depend_keys
                _dependancies.append(_dependancy)

        missing = set([attr for attr in missing if not hasattr(self, attr) or (getattr(self, attr) is None)])
        if missing and read_missing:
            # TODO: check that there attributes are present in the file
            self.__log.warning("Missing data : {}".format(str(missing)))
            self.__log.info("-----Now reading--------")

            self.read_data(list_data=missing, list_detector=self.list_detector)

        # Check that everything was read
        for key in missing:
            assert hasattr(self, key) and (getattr(self, key) is not None), "Missing data {}".format(key)

        # Check in _dependancies that everything was read, if not we are missing something
        if _dependancies:
            missing = [
                attr for attr in chain(*_dependancies) if not hasattr(self, attr) or (getattr(self, attr) is None)
            ]
            return missing or None
        return None

    def get_list_detector(self, namedet=None, flag=None, typedet=None):
        """Retrieve the valid detector list given a pattern.

        Attributes
        ----------
        namedet: str
            any string pattern a KID name should match in a `in` operation
        flag: int
            select only KIDs with the given flag
        typedet: int or list of int
            select only KIDs with the given type or types

        Returns
        -------
        list_detector: list
            the list which should be used for the `.read_data()` method
        """
        mask = ~self._kidpar["index"].mask
        if namedet is not None:
            mask = mask & [namedet in name for name in self._kidpar["namedet"]]
        if flag is not None:
            mask = mask & self._kidpar["flag"] == flag
        if typedet is not None:
            if not isinstance(typedet, list):
                typedet = [typedet]
            mask = mask & [_typedet in typedet for _typedet in self._kidpar["typedet"]]

        return np.array(self._kidpar[mask]["namedet"])

    @property
    def kidpar(self):
        """Retrieve the kidpar of the observation.

        Notes
        -----
        If there is an extended kidpar, it will be merge with the data kidpar

        """
        if self._extended_kidpar:
            kidpar = deepcopy(self._kidpar)
            # astropy.table.join fails when index is masked so..
            # See astropy #9289
            # and fails to sort when there is an index.....
            kidpar.remove_indices("namedet")
            kidpar.sort("index")
            mask = deepcopy(kidpar["index"].mask)
            kidpar["index"].mask = False
            kidpar = join(kidpar, self._extended_kidpar, join_type="outer", keys="namedet")
            kidpar.sort("index")
            kidpar["index"].mask = mask
            kidpar.add_index("namedet")
        else:
            self.__log.warning("No extended kidpar found")
            kidpar = self._kidpar

        return kidpar
