import os
import re
import numpy as np
import logging

from pathlib import Path
from copy import deepcopy

from functools import lru_cache
from itertools import chain

from custom_inherit import DocInheritMeta

from datetime import datetime

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, join
from astropy.utils.metadata import MetaData

import h5py
from autologging import logged

from .read_kidsdata import read_info, read_all, read_raw
from .read_kidsdata import read_info_hdf5, read_all_hdf5
from .read_kidsdata import info_to_hdf5, data_to_hdf5
from .read_kidsdata import filename_to_meta

from .utils import roll_fft, interp_nan

from .database.constants import RE_ASTRO, RE_MANUAL, RE_DIR, RE_TABLEBT
from .database.api import get_kidpar
from kidsdata.settings import CACHE_DIR

if not CACHE_DIR.exists():
    logging.warning("Cache dir do not exist : {}".format(CACHE_DIR))

_meta_doc = """`dict`-like : Additional meta information about the dataset."""


@logged
class KidsRawData(metaclass=DocInheritMeta(style="numpy_with_merge", include_special_methods=True)):
    """Class dealing with KIDS raw data.

    Attributes
    ----------
    filename: str, or int
        Name of the raw data file. Could also be a scan number
    header : TconfigHeader (nametuple)
        the raw header of the file
    param_c: :dict
        common parameters
    names : TName (namedtuple)
        the variable names contained in the raw file
    nsamples : int
        the total number of sample
    nptint : int
        the number of points per bloc
    nint : int
        the number of bloc
    ndet : int
        the number of KIDS detectors
    kidpar: :obj: Astropy.Table
        the kidpar contatenation of the infile _kidpar, and external _extended_kidpar
    list_detector : array_like
        names of the read detectors
    exptime : float
        the exposure time of the object
    position_shift : float
        the shift in sample between the position of the telescope and the KIDs data

    __dataSc, __dataSd: dict
        Fully sampled common and detector data
    __dataUc, __dataUd: dict
        Undersampled common and detector data

    Methods
    -------
    info()
        Display the basic infomation about the data file.
    meta()
        Default meta data for products.
    read_data(**kwargs)
        Read the raw data.
    get_list_detector(**kwargs)
        Retrieve the valid detector list given a pattern.
    get_telescope_positions(coord="pdiff") (cached)
        Retrieve the telescope position, with shifts applied.

    Notes
    -----
    All read data `__dataSc, __dataSd, __dataUc, __dataUd` are linked as top level attributes.
    """

    meta = MetaData(doc=_meta_doc, copy=False)

    filename = None
    header = None
    param_c = None
    names = None
    nsamples = None
    nptint = None
    nint = None

    _kidpar = None
    _extended_kidpar = None

    __raw = False
    __position_shift = None
    telescope_positions = None

    _cache = None  # The open cache file
    _cache_filename = None

    list_detector = None
    __dataSc = None
    __dataSd = None
    __dataUc = None
    __dataUd = None
    __dataRg = None

    def __init__(self, filename, position_shift=None, e_kidpar="auto", raw=True):
        """Initialize a KidsRawData object....

        Parameters
        ----------
        filename: str or Path
            the path to the raw or hdf5 filename
        position_shift : int of float, optional
            shit of the telescope positions
        e_kidpar: str of Path or 'auto'
            the path to the extended kidpar
        raw: bool
            Use the read_raw routine

        Returns
        -------
        kd : KidsRawData
            the object initialized with the header informations
        """
        if isinstance(filename, (int, np.int)):
            from .database.api import get_scan  # To avoid circular import

            filename = get_scan(filename)

        self.filename = Path(filename)
        self.meta.update(filename_to_meta(self.filename))

        self.__raw = raw
        self.__dataSc = dict()
        self.__dataSd = dict()
        self.__dataUc = dict()
        self.__dataUd = dict()
        self.__dataRg = dict()

        if h5py.is_hdf5(self.filename):
            info = read_info_hdf5(self.filename)
        elif self.__raw:
            *info, _, _, _, _, _, _ = read_raw(self.filename, list_data=[])
        else:
            info = read_info(self.filename)

        self.header, self.param_c, _kidpar, self.names, self.nsamples = info
        del info  # Need to release the reference counting

        # TODO : Check with NIKA data that nptint exist, otherwise should be moved to KissRawData
        self.nptint = self.header.nb_pt_bloc  # Number of points for one interferogram
        self.nint = self.nsamples // self.nptint  # Number of interferograms

        self._kidpar = _kidpar

        self._extended_kidpar = None
        if e_kidpar == "auto":
            # Beware obdate is based on the filename.....
            try:
                e_kidpar = get_kidpar(self.meta["OBSDATE"])
            except ValueError:
                e_kidpar = None
        if e_kidpar is not None:
            e_kidpar_name = Path(e_kidpar).name
            self.__log.info("Using extended kidpar {}".format(e_kidpar_name))

            self._extended_kidpar = Table.read(e_kidpar)
            # Check if filename is set properly in the extended kidpar (for later use)
            if self._extended_kidpar.meta["FILENAME"] != e_kidpar_name:
                self.__log.warning("Updating filename in extended kidpar")
                self._extended_kidpar.meta["FILENAME"] = e_kidpar_name

        # Position related attributes, move to meta ?
        self.__position_shift = position_shift

        # Default detector list, everything not masked
        self.list_detector = np.array(self._kidpar[~self._kidpar["index"].mask]["namedet"])

        # TODO: Need to decide what file structure we want, flat, or by dates
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

    # TODO: This could probably be done more elegantly
    @property
    def position_shift(self):
        return self.__position_shift

    @position_shift.setter
    def position_shift(self, value):
        if self.__position_shift is None:
            self.__position_shift = value
        else:
            self.__position_shift += value
        KidsRawData.get_telescope_positions.cache_clear()

    @property
    def exptime(self):
        return (self.obstime[-1] - self.obstime[0]).to(u.s)

    def info(self):
        """Display the basic infomation about the data file."""
        print("RAW DATA")
        print("==================")
        print("File name:\t" + str(self.filename))
        print("------------------")
        print("Source name:\t" + (self.meta["source"] or "Unknown"))
        print("Observed date:\t" + self.meta["OBSDATE"].iso)
        print("Description:\t" + self.meta["obstype"] or "Unknown")
        print("Scan number:\t" + str(self.meta["scan"] or "Unknown"))

        print("------------------")
        print("No. of KIDS detectors:\t", self.ndet)
        print("No. of time samples:\t", self.nsamples)
        print(
            "Typical size of fully sampled data (GiB):\t{:3.1f}".format(self.nsamples * self.ndet * 32 / 8 / 1024 ** 3)
        )

    def _update_meta(self):
        """Default meta data for products.
        Notes
        -----
        Follow fits convention http://archive.stsci.edu/fits/fits_standard/node40.html
        Should look at IVOA Provenance models
        """

        self.meta["OBS-ID"] = self.meta.get("scan", "Unknown")
        self.meta["DATE"] = datetime.now().isoformat()

        self.meta["EXPTIME"] = self.exptime.value
        self.meta["DATE-OBS"] = self.obstime[0].isot
        self.meta["DATE-END"] = self.obstime[-1].isot
        self.meta["INSTRUME"] = self.param_c.get("nomexp", "Unknown")
        self.meta["AUTHOR"] = "KidsData"
        self.meta["ORIGIN"] = os.environ.get("HOSTNAME")
        self.meta["TELESCOP"] = self.meta.get("TELESCOP", "")
        self.meta["INSTRUME"] = self.meta.get("INSTRUME", "")
        self.meta["OBSERVER"] = os.environ["USER"]

        # Add extra keyword
        self.meta["NKIDS"] = self.ndet
        self.meta["NINT"] = self.nint
        self.meta["NPTINT"] = self.nptint
        self.meta["NSAMPLES"] = self.nsamples
        self.meta["KIDPAR"] = self._extended_kidpar.meta["FILENAME"] if self._extended_kidpar else ""
        self.meta["POSITION_SHIFT"] = self.position_shift or 0

    def __pos_keys(self):
        """Return potential telescope position key in the object."""

        keys = self.__dict__.keys()
        pos_keys = {
            key.replace("Az", ""): (key, key.replace("Az", "El"))
            for key in [key for key in keys if key.endswith("Az")]
            if key in keys and key.replace("Az", "El") in keys
        }
        for key in list(pos_keys.keys()):
            if key == "":
                pos_keys["default"] = pos_keys.pop("")
            if "_" in key:
                pos_keys[key.split("_")[1]] = pos_keys.pop(key)
        return pos_keys

    def read_data(self, list_data=None, cache=False, array=np.array, **kwargs):
        """Read the raw data.

        Parameters
        ----------
        list_data : list of str or str
            list of data to read, see Notes
        cache : bool or 'only', optional
            use the cache file if present, by default False, see Notes
        array : function, (np.array|dask.array.from_array|None) optional
            function to apply to the largest cached value, by default np.array, if None return h5py.Dataset
        **kwargs
            additionnal parameters to be passed to the  `kidsdata.read_kidsdata.read_all`, in particular
                list_detector : list, optional
                    the list of detector indexes to be read, see Notes, by default None: read all detectors
                start : int
                    the starting block, default 0.
                end : type
                    the ending block, default full available dataset.
                silent : bool
                    Silence the output of the C library. The default is True
                diff_pps: bool
                    pre-compute pps time differences. The default is False
                correct_pps: bool
                    correct the pps signal. The default is False
                correct_time: bool or float
                    correct the time signal by interpolating jumps higher that given value in second. The default is False

        Notes
        -----
        if `cache=True`, the function reads all possible data from the cache file, and read the missing data from the raw binary file
        if `cache='only'`, the function reads all possible data from the cache file

        Depending on the `read_raw` flag when openning a kidsdata, the meaning of `list_data` and `list_detector` is changed:
        if raw is False:
            - `list_data` is a list of str within the data present in the files, see the `names` property.
            - `list_detector` is a list or array of detector names within the `kidpar` of the file. See also `get_list_detector`.
        if raw is True:
            - `list_data` is a list or array contains elements from ['Sc', 'Sd, 'Uc', 'Ud', 'Rg'].
            - `list_detector` is either a list or array of detector names within the `kidpar` of the file or
                - 'all' : to read all kids
                - 'one' or None : to read all kids of type 1
                - 'array?' : to read kids from crate/array '?'. '?' must be an int.
                - 'array_one?' : to read kids of type 1 from crate/array '?'. '?' must be an int.
                - 'box?' : to read kids from  box '?'. '?' must be an int or a letter
                - 'box_one?' : to read kids of type 1 from box '?'. '?' must be an int or a letter

                For CONCERTO, crate/array '?' must be from 2 to 3, or :
                - 'arrayT' : to read the kids from the array in transmission
                - 'arrayT_one' : to read the kids of type 1 from the array in transmission
                ' 'arrayR' : to read the kids from array in reflection
                - 'arrayR_one': to read the kids of type 1 from the array in reflection

        `None` or 'all' means read all data or detectors.
        """
        self.__log.debug("Reading data")

        if cache and self._cache is not None:
            self.__log.info("Reading cached raw data :")

            nb_samples_read, list_detector, *datas, extended_kidpar = read_all_hdf5(self._cache, array=array)

            dataSc, dataSd, dataUc, dataUd, dataRg = datas

            self.__log.debug("Updating dictionnaries with cached data")
            self.__dataSc.update(dataSc)
            self.__dataSd.update(dataSd)
            self.__dataUc.update(dataUc)
            self.__dataUd.update(dataUd)
            self.__dataRg.update(dataRg)

            self._extended_kidpar = extended_kidpar

            # Removing cache data from the requested list_data:
            if list_data == "all" or list_data is None:
                list_data = self.names.DataSc + self.names.DataSd + self.names.DataUc + self.names.DataUd

            keys = [key for data in datas for key in data if key in list_data]

            self.__log.debug("Read cached raw data : {}".format(keys))
            for key in keys:
                list_data.remove(key)
            self.__log.debug("Remaining raw data : {}".format(list_data))

            self.list_detector = np.array(list_detector)

            del datas  # Need to release the reference counting
            del (dataSc, dataSd, dataUc, dataUd, dataRg)  # Need to release the reference counting

            # TODO: list_detectors and nsamples
            if self.nsamples != nb_samples_read:
                self.__log.warning("Read less sample than expected : {} vs {}".format(nb_samples_read, self.nsamples))
                self.nsamples = nb_samples_read
                # Assuming one loose a full block
                self.nint = nb_samples_read // self.nptint

        if cache != "only" and list_data and not h5py.is_hdf5(self.filename):
            self.__log.debug("Reading raw data")

            if self.__raw:
                _, _, _, _, nb_samples_read, list_detector, *datas = read_raw(
                    self.filename, list_data=list_data, **kwargs
                )
            else:
                nb_samples_read, list_detector, *datas = read_all(self.filename, list_data=list_data, **kwargs)

            dataSc, dataSd, dataUc, dataUd, dataRg = datas

            self.__log.debug("Updating dictionnaries")
            self.__dataSc.update(dataSc)
            self.__dataSd.update(dataSd)
            self.__dataUc.update(dataUc)
            self.__dataUd.update(dataUd)
            self.__dataRg.update(dataRg)

            del datas  # Need to release the reference counting
            del (dataSc, dataSd, dataUc, dataUd, dataRg)  # Need to release the reference counting

            self.list_detector = np.array(list_detector)

            if self.nsamples != nb_samples_read:
                self.__log.warning("Read less sample than expected : {} vs {}".format(nb_samples_read, self.nsamples))
                self.nsamples = nb_samples_read
                # Assuming one loose a full block
                self.nint = nb_samples_read // self.nptint

        # Expand keys :
        # Does not double memory, but it will not be possible to
        # partially free memory : All attribute read at the same time
        # must be deleted together
        for _dict in [self.__dataSc, self.__dataSd, self.__dataUc, self.__dataUd]:
            for ckey in _dict.keys():
                self.__dict__[ckey] = _dict[ckey]

        # TODO: Define telescope position if present...
        # from .telescope_positions import BasicPositions

        # pos_keys = self.__pos_keys()

        # for key in pos_keys:
        #     lon, lat = pos_keys[key]
        #     pos = np.array([getattr(self, lon).flatten(), getattr(self, lat).flatten()])
        #     mjd = getattr(self, "obstime").flatten()

        #     if mjd.shape[0] == pos.shape[1]:
        #         # fully sampled position :
        #         pos_keys[key] = BasicPositions(mjd, pos)
        #     elif mjd.shape[0] // self.nptint == pos.shape[1]:
        #         # Undersamped position, assuming center of block
        #         u_mjd = Time(mjd.mjd.reshape(self.nint, self.nptint).mean(1), scale=mjd.scale, format="mjd")
        #         pos_keys[key] = BasicPositions(u_mjd, pos)
        #     else:
        #         raise ValueError("Do not known how to handle position {}".format(key))

        # self.__position_key = pos_keys

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
            additionnal keyword argument for the h5py.Dataset, see Notes

        Notes
        -----
        Usual kwargs could be :

        kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}

        """
        if filename is None:
            if self._cache:
                self._cache.close()
            filename = self._cache_filename

        self.__log.info("Saving in {}".format(filename))

        _file_kwargs = {"mode": mode}
        if file_kwargs is not None:
            _file_kwargs.update(**file_kwargs)

        self.__log.debug("Saving info")
        info_to_hdf5(
            filename,
            self.header,
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
            self.list_detector,
            self.__dataSc,
            self.__dataSd if dataSd else None,
            self.__dataUc,
            self.__dataUd,
            self.__dataRg,
            self._extended_kidpar,
            file_kwargs=_file_kwargs,
            **kwargs
        )

    def _clean_data(self, key):
        """Remove data from the object.

        Parameters
        ----------
        key : str _KidsRawData__(dataSc|dataSd|dataUc|dataUd)
            the data to clean
        """
        if not isinstance(getattr(self, key, None), dict):
            self.__log.warning("{} is not present or is not a dict".format(key))
            return None

        self.__log.debug("Cleaning top level {}".format(key))
        # delete from the top level dictionnary
        for _key in getattr(self, key).keys():
            delattr(self, _key)
        del _key
        # delete the dictionnary itself
        self.__log.debug("Cleaning {}".format(key))
        delattr(self, key)
        # all references should be freed !!
        # default value, if read again :
        setattr(self, key, dict())

    # Check if we can merge that with the asserions in other functions
    # Beware that some are read some are computed...
    def __check_attributes(self, attr_list, dependancies=None, read_missing=False):
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
            any regular expression pattern a KID name should match
        flag: int
            select only KIDs with the given flag
        typedet: int or list of int
            select only KIDs with the given type or types

        Returns
        -------
        list_detector: list
            the list which should be used for the `.read_data()` method

        Notes
        -----
        namedet='KA' will match all KA detectors
        namedet='K(A|B)' will match both KA and KB detectors

        One can combine selection :
        namedet='KA', flag=0 : will match all KA detectors with flag=0
        """

        mask = ~self._kidpar["index"].mask
        if namedet is not None:
            mask = mask & [re.match(namedet, name) is not None for name in self._kidpar["namedet"]]
        if flag is not None:
            mask = mask & (self._kidpar["flag"] == flag)
        if typedet is not None:
            if not isinstance(typedet, list):
                typedet = [typedet]
            mask = mask & [_typedet in typedet for _typedet in self._kidpar["typedet"]]

        return np.array(self._kidpar[mask]["namedet"])

    def flag_telescope_speed(self, coord="pdiff", **kwargs):
        """Flag telescope position based on telescope speed.

        Parameters
        ----------
        coord : str
            the type of position to retrieve
        sigma : float
            the sigma-kappa threshold for the speed
        min_speed : float in arcsec / s
            the minimum speed to flag before sigma clipping, default 0
        savgol_args : tuple
            the arguments of the savgol function
        """
        self.telescope_positions.flag_speed(key=coord, **kwargs)
        KidsRawData.get_telescope_positions.cache_clear()
        KidsRawData._get_positions.cache_clear()

    ## Another solution would be to shift the mjds and interpolate, but this would require too much interpolation when searching for position_shift...
    @lru_cache(maxsize=2)
    def get_telescope_positions(self, coord="pdiff", undersampled=True):
        """Retrieve the telescope position, with shifts applied.

        Parameters
        ----------
        coord : str
            the type of position to retrieve
        undersampled : bool
            retrieve undersamped positions, default True

        Returns
        ------
        lon, lat, mask : array_like
            the corresponding longitude, latitude with shifts applied
        """
        lon, lat, mask = self._get_positions(coord=coord, undersampled=undersampled)

        lon = lon.copy()
        lat = lat.copy()

        position_shift = self.position_shift

        if position_shift is None:
            return lon, lat, mask

        self.__log.info("Rolling telescope position by {}".format(position_shift))

        if not undersampled:
            position_shift = position_shift * self.nptint

        if isinstance(position_shift, (int, np.int, np.int16, np.int32, np.int64)):
            lon = np.roll(lon, position_shift)
            lat = np.roll(lat, position_shift)

            if mask is not False:
                mask = mask.copy()
                mask = np.roll(mask, position_shift)
        elif isinstance(position_shift, (float, np.float, np.float32, np.float64)):

            # replace flagged values by interpolated values before performing the ffts...
            for item in [lon, lat]:
                item[mask] = np.nan
                item[:] = interp_nan(item)
                item[:] = roll_fft(item, position_shift)

            if mask is not False:
                mask = mask.copy()
                mask = roll_fft(mask.astype(float), position_shift) > 0.5

        return lon, lat, mask

    @lru_cache(maxsize=2)
    def _get_positions(self, coord="pdiff"):
        """Retrieve interpolated telescope position, without shift.

        Parameters
        ----------
        coord : str
            the type of position to retrieve

        Returns
        ------
        lon, lat, mask : array_like
            the corresponding longitude, latitude and mask
        """

        mjd = self.obstime

        return self.telescope_positions.get_interpolated_positions(mjd, key=coord)

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
