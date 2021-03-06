# pylint: disable=C0301,C0103
import gc
import h5py
from functools import partial, wraps
from enum import IntEnum

import logging
from autologging import logged
import ctypes
from pathlib import Path
from collections import namedtuple
from ipaddress import ip_address

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_closing

from datetime import datetime
from dateutil.parser import parse

import astropy.units as u
from astropy.table import Table, MaskedColumn
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
from astropy.time import Time

from . import settings
from .utils import _import_from, sizeof_fmt, pprint_list, mad_med
from .database.constants import RE_ASTRO, RE_MANUAL, RE_DIR, RE_TABLEBT

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

READDATA_LIB_SO = Path(settings.READDATA_LIB_PATH) / "libreaddata.so"
READRAW_LIB_SO = Path(settings.READRAW_LIB_PATH) / "libreadraw.so"

if READDATA_LIB_SO.exists():
    READDATA = ctypes.cdll.LoadLibrary(str(READDATA_LIB_SO))
else:
    READDATA = None
    logging.error(
        "Could not find {} : Please check $READDATA_LIB_PATH ({})".format(
            READDATA_LIB_SO.name, settings.READDATA_LIB_PATH
        )
    )

if READRAW_LIB_SO.exists():
    READRAW = ctypes.cdll.LoadLibrary(str(READRAW_LIB_SO))
else:
    READRAW = None
    logging.error(
        "Could not find {} : Please check $READRAW_LIB_PATH ({})".format(READRAW_LIB_SO.name, settings.READRAW_LIB_PATH)
    )


# Defining TconfigHeader following Acquisition/Library/configNika/TconfigNika.h
TconfigHeader = namedtuple(
    "TconfigHeader",
    [
        "size_MotorModulTable",
        "nb_brut_ud",
        "nb_boites_mesure",
        "nb_detecteurs",
        "nb_pt_bloc",
        "nb_sample_fichier",
        "nb_det_mini",
        "nb_brut_uc",
        "version_header",
        "nb_param_c",
        "nb_param_d",
        "lg_header_util",
        "nb_brut_c",
        "nb_brut_d",
        "nb_brut_periode",
        "nb_data_c",
        "nb_data_d",
        "nb_champ_reglage",
    ],
)

# TName list the name of the computed/requested variables and detectors
TName = namedtuple("TName", ["DataSc", "DataSd", "DataUc", "DataUd", "RawDataDetector"])


DETECTORS_CODE = {"kid": 3, "kod": 2, "all": 1, "a1": 4, "a2": 5, "a3": 6}


@logged
def filename_to_meta(filename):
    """Fill meta data from the filename."""

    filename = Path(filename)

    re_astro = RE_ASTRO.match(filename.name)
    re_manual = RE_MANUAL.match(filename.name)
    re_tablebt = RE_TABLEBT.match(filename.name)
    re_dir = RE_DIR.match(filename.parent.name)

    scan = None
    if re_astro:
        scan = re_astro.groups()[2]
        if scan is not None and scan != "":
            scan = int(scan)
        else:
            scan = -1
    elif re_tablebt:
        scan = int(re_tablebt.groups()[2])
    else:
        filename_to_meta._log.warning("No source from filename")

    source = None
    if re_astro:
        source = re_astro.groups()[3]

    obstype = None
    if re_astro:
        obstype = re_astro.groups()[4]
    elif re_manual:
        obstype = "ManualScan"
    elif re_tablebt:
        obstype = "TableScan"
    else:
        filename_to_meta._log.warning("No obstype from filename")

    date = None
    if re_astro:
        date = re_astro.groups()[0]
    elif re_manual:
        date = "".join(re_manual.groups()[0:3])
    elif (re_tablebt is not None) & (re_dir is not None):
        date = "".join(re_dir.groups()[1:])
    else:
        # Probably not a data scan
        filename_to_meta._log.error("No time on filename")
    if date is not None:
        date = Time(parse(date), scale="utc")  # UTC ?

    return {"FILENAME": str(filename), "SCAN": scan, "OBJECT": source, "OBSTYPE": obstype, "OBSDATE": date}


def decode_ip(int32_val):
    """Decode an int32 bit into ip string."""
    # binary = np.binary_repr(int32_val, width=32)
    # int8_arr = [int(binary[0:8], 2), int(binary[8:16], 2), int(binary[16:24], 2), int(binary[24:32], 2)]
    # return ".".join([str(int8) for int8 in int8_arr])
    return str(ip_address(int32_val.tobytes()))


# @profile
def read_info(filename, det2read="KID", list_data="all", fix=True, flat=True, silent=True):
    """Read header information from a binary file.

    Parameters
    ----------
    filename : str
        Full filename
    det2read : str (kid, kod, all, a1, a2, a3)
        Detector type to read
    list_data : list or 'all' or 'raw'
        A list containing the list of data to be read, or the string 'all' or 'raw'
    fix : boolean
        fix the end of the detector names
    silent : bool (default:True)
        Silence the output of the C library

    Returns
    -------
    tconfigheader: namedtuple
        a copy of the TconfigHeader of the file
    param_c: dict
        the common variables for the file
    kidpar: :class:`~astropy.table.Table`
        the kidpar of the file
    TName: namedtuple
        the full name of the requested data and detectors
    nb_read_sample: int
        the total  number of sample in the file

    """
    assert Path(filename).exists(), "{} does not exist".format(filename)
    assert Path(filename).stat().st_size != 0, "{} is empty".format(filename)

    if list_data in ["all", "raw"]:
        str_data = list_data
    else:
        str_data = " ".join(list_data)

    codelistdet = DETECTORS_CODE.get(det2read.lower(), -1)

    # nb_char_nom = 16
    length_header = 300000
    var_name_length = 200000
    nb_max_det = 8001

    p_int32 = ctypes.POINTER(ctypes.c_int32)

    read_nika_info = READDATA.read_nika_info
    read_nika_info.argtypes = [
        ctypes.c_char_p,
        p_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int32,
        p_int32,
        p_int32,
        p_int32,
        p_int32,
        p_int32,
        p_int32,
        p_int32,
        ctypes.c_int,
    ]

    buffer_header = np.zeros(length_header, dtype=np.int32)
    var_name_buffer = ctypes.create_string_buffer(var_name_length)
    list_detector = np.zeros(nb_max_det, dtype=np.int32)
    nb_Sc = ctypes.c_int32()
    nb_Sd = ctypes.c_int32()
    nb_Uc = ctypes.c_int32()
    nb_Ud = ctypes.c_int32()
    idx_param_c = ctypes.c_int32()
    idx_param_d = ctypes.c_int32()

    nb_read_samples = read_nika_info(
        bytes(str(filename), "ascii"),
        buffer_header.ctypes.data_as(p_int32),
        length_header,
        var_name_buffer,
        var_name_length,
        bytes(str_data, "ascii"),
        codelistdet,
        list_detector.ctypes.data_as(p_int32),
        nb_Sc,
        nb_Sd,
        nb_Uc,
        nb_Ud,
        idx_param_c,
        idx_param_d,
        silent,
    )

    nb_Sc = nb_Sc.value
    nb_Sd = nb_Sd.value
    nb_Uc = nb_Uc.value
    nb_Ud = nb_Ud.value
    idx_param_c = idx_param_c.value
    idx_param_d = idx_param_d.value

    # See Acquisition/Library/configNika/TconfigNika.h
    header = TconfigHeader(*(buffer_header[4:22].tolist()))

    nb_detectors = list_detector[0]
    idx_detectors = list_detector[1 : list_detector[0] + 1]

    # Proper way to do it :
    # var_name = [var_name_buffer.raw[ind: ind+16].strip(b'\x00').decode('ascii') for ind in range(0, len(var_name_buffer.raw), 16)]
    # Faster way :
    # var_name = [name for name in var_name_buffer.raw.decode("ascii").split("\x00") if name != ""]
    # Mix of the two :
    var_name_buffer = var_name_buffer.raw.decode("ascii")
    var_name = [var_name_buffer[ind : ind + 16].strip("\x00") for ind in range(0, len(var_name_buffer), 16)]

    # Retrieve the param commun
    idx = 0
    name_param_c = var_name[idx : idx + header.nb_param_c]
    val_param_c = buffer_header[idx_param_c : idx_param_c + header.nb_param_c]
    param_c = dict(zip(name_param_c, val_param_c))
    param_c = clean_param_c(param_c, flat=flat)

    # Retrieve the param detector
    idx += header.nb_param_c
    name_param_d = var_name[idx : idx + header.nb_param_d]
    val_param_d = buffer_header[idx_param_d : idx_param_d + header.nb_param_d * header.nb_detecteurs].reshape(
        header.nb_param_d, header.nb_detecteurs
    )
    param_d = dict(zip(name_param_d, val_param_d))

    # nom1 & nom2 are actually defined as a struct TName8 namedet (TconfingNika.h), ie char[8] ie 64 bit (public_def.h)
    namedet = np.append(param_d["nom1"], param_d["nom2"]).reshape(header.nb_detecteurs, 2)
    for key in ["nom1", "nom2"]:
        del param_d[key]
    param_d["namedet"] = clean_namedet(namedet, fix=fix)

    kidpar = clean_param_d(param_d)
    # unmask the detectors present in the file
    kidpar["index"].mask[idx_detectors] = False

    idx += header.nb_param_d
    name_data_Sc = var_name[idx : idx + nb_Sc]
    idx += nb_Sc
    name_data_Sd = var_name[idx : idx + nb_Sd]
    # name_data_Sd.append('flag') # WHY ???
    idx += nb_Sd
    name_data_Uc = var_name[idx : idx + nb_Uc]
    idx += nb_Uc
    name_data_Ud = var_name[idx : idx + nb_Ud]
    idx += nb_Ud
    name_detectors = var_name[idx : idx + nb_detectors]

    names = TName(name_data_Sc, name_data_Sd, name_data_Uc, name_data_Ud, name_detectors)

    del (buffer_header, var_name_buffer, list_detector)
    gc.collect()
    ctypes._reset_cache()

    return header, param_c, kidpar, names, nb_read_samples


@logged
def read_all(
    filename,
    list_data=None,
    list_detector=None,
    start=None,
    end=None,
    silent=True,
    diff_pps=False,
    correct_pps=False,
    correct_time=False,
    ordering="K",
):
    """Read Raw data from the binary file using the `read_nika_all` C function.

    Parameters
    ----------
    filename : str
        Full filename
    list_data : list of string, or  'all'
        A list containing the  data to be read, or the string 'all' to read all data
    list_detector : array_like of str or 'None'
        The names of detectors to read, by default `None` read all available KIDs.
    start : int
        The starting block, default 0.
    end : type
        The ending block, default full available dataset.
    silent : bool
        Silence the output of the C library. The default is True
    diff_pps: bool
        pre-compute pps time differences. The default is False
    correct_pps: bool
        correct the pps signal. The default is False
    correct_time: bool or float
        correct the time signal by interpolating jumps higher that given value in second. The default is False
    ordering: str
        memory ordering requested to convert data from NIKA reading library to python numpy array
        The default is 'K' which speedup the conversion by keeping memory ordering. It can be changed to 'C'. This
        variable must be really checked for further analysis and how it impact performances
    Returns
    -------
    nb_samples_read : int
        The number of sample read
    dataSc : dict:
        A dictionnary containing all the requested sampled common quantities data as 2D :class:`~numpy.array` of shape (n_bloc, nb_pt_bloc)
    dataSd : dict
        A dictionnary containing all the requested sampled data as 3D :class:`~numpy.array` of shape (n_det, n_bloc, nb_pt_bloc)
    dataUc : dict:
        A dictionnary containing all the requested under-sampled common quantities data as 1D :class:`~numpy.array` of shape (n_bloc,)
    dataUd : dict
        A dictionnary containing all the requested under-sampled data as 2D :class:`~numpy.array` of shape (n_det, n_bloc)
    dataRg: dict
        An empty dictionnary for the moment, as `readdata_nika_data` do not handle reglage

    Notes
    -----
    `list_detector` is a list or array of detector names within the `kidpar` of the file.
    `list_data` must be within the data present in the files, see the `names` property of read_info

    """
    assert Path(filename).exists(), "{} does not exist".format(filename)
    assert Path(filename).stat().st_size != 0, "{} is empty".format(filename)

    if list_data is None:
        raise ValueError("You must provide a list_data")
    elif list_data in ["all", "raw"]:
        str_data = list_data
    else:
        str_data = " ".join(list_data)

    # Read the basic header from the file and the name of the data
    header, param_c, kidpar, names, nb_read_info = read_info(filename, list_data=list_data, silent=silent)

    if list_detector is None:
        list_detector = np.where(~kidpar["index"].mask)[0]
    else:
        list_detector = np.array(kidpar.loc_indices[list_detector])

    # Append the number of detectors as expected by the C library
    list_detector = np.insert(list_detector, 0, len(list_detector)).astype(np.int32)

    assert len(names.RawDataDetector) >= list_detector[0]

    # Number of blocks to read
    start = (start or 0) * header.nb_pt_bloc
    end = (end or (nb_read_info // header.nb_pt_bloc)) * header.nb_pt_bloc
    nb_to_read = end - start

    p_int32 = ctypes.POINTER(ctypes.c_int32)
    p_float = ctypes.POINTER(ctypes.c_float)

    read_nika_all = READDATA.read_nika_all
    read_nika_all.argtype = [
        ctypes.c_char_p,
        p_float,
        p_float,
        ctypes.c_char_p,
        p_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    # Number of data to read
    nb_Sc = len(names.DataSc)
    nb_Sd = len(names.DataSd)
    nb_Uc = len(names.DataUc)
    nb_Ud = len(names.DataUd)
    nb_detectors = list_detector[0]

    np_pt_bloc = header.nb_pt_bloc

    _sample_S = nb_Sc + nb_Sd * nb_detectors
    _sample_U = nb_Uc + nb_Ud * nb_detectors

    buffer_dataS = np.zeros(nb_to_read * _sample_S, dtype=np.float32)
    buffer_dataU = np.zeros(nb_to_read // np_pt_bloc * _sample_U, dtype=np.float32)

    read_all._log.info(
        "buffer allocated : \n    - buffer_dataS  {}\n    - buffer_dataU  {}".format(
            sizeof_fmt(buffer_dataS.nbytes), sizeof_fmt(buffer_dataU.nbytes)
        )
    )
    read_all._log.debug("before read_nika_all")
    nb_samples_read = read_nika_all(
        bytes(str(filename), "ascii"),
        buffer_dataS.ctypes.data_as(p_float),
        buffer_dataU.ctypes.data_as(p_float),
        bytes(str_data, "ascii"),
        list_detector.ctypes.data_as(p_int32),
        start,
        nb_to_read,
        silent,
    )
    read_all._log.debug("after read_nika_all")

    if nb_samples_read != nb_to_read:
        read_all._log.warning("Did not read all requested data")

        buffer_dataS = buffer_dataS[0 : nb_samples_read * _sample_S]
        buffer_dataU = buffer_dataU[0 : nb_samples_read // np_pt_bloc * _sample_U]

    # Split the buffer into common and data part with proper shape
    _dataSc = buffer_dataS.reshape(nb_samples_read, _sample_S)[:, 0:nb_Sc].T
    _dataSd = np.moveaxis(
        buffer_dataS.reshape(nb_samples_read, _sample_S)[:, nb_Sc:].reshape(nb_samples_read, nb_Sd, nb_detectors), 0, -1
    )
    del buffer_dataS  # Do not relase actual memory here because _dataSc and _dataSd are view on it...

    _dataUc = buffer_dataU.reshape(nb_samples_read // np_pt_bloc, _sample_U)[:, 0:nb_Uc].T
    _dataUd = np.moveaxis(
        buffer_dataU.reshape(nb_samples_read // np_pt_bloc, _sample_U)[:, nb_Uc:].reshape(
            nb_samples_read // np_pt_bloc, nb_Ud, nb_detectors
        ),
        0,
        -1,
    )
    del buffer_dataU  # Do not release actual memory here...

    # Split the data by name, here data are 1D or 2D numpy array,
    # Cast ?d data to float32

    # WARNING : We need to copy the data, so that we loose previous
    # reference to buffer_data*, and hence we can really release its
    # memory
    # NOTE : This is causing a extra usage of RAM and CPU during
    # copy... Could be handled by the C library..

    # TODO: Proper reshaping of quantities here !!!

    dataSc = {name: data.reshape(-1, np_pt_bloc).copy() for name, data in zip(names.DataSc, _dataSc)}
    del _dataSc
    dataSd = {
        name: data.reshape(nb_detectors, -1, np_pt_bloc).astype(np.float32, order=ordering, casting="unsafe")
        for name, data in zip(names.DataSd, _dataSd)
    }
    del _dataSd

    dataUc = {name: data.copy() for name, data in zip(names.DataUc, _dataUc)}
    del _dataUc
    dataUd = {
        name: data.reshape(nb_detectors, -1).astype(np.float32, order=ordering, casting="unsafe")
        for name, data in zip(names.DataUd, _dataUd)
    }
    del _dataUd

    # Shift RF_didq if present
    dataSd = clean_dataSd(dataSd)

    # Convert units azimuth and elevation to degrees
    for data in [dataUc, dataSc]:
        clean_position_unit(data)

    # Compute median pps_time and differences and corrections
    obsdate = filename_to_meta(filename)["OBSDATE"]
    clean_dataSc(dataSc, param_c["acqfreq"], diff_pps, correct_pps, correct_time, obsdate=obsdate)

    gc.collect()
    ctypes._reset_cache()

    namedet = kidpar[list_detector[1:]]["namedet"]

    dataRg = {}

    return nb_samples_read, namedet, dataSc, dataSd, dataUc, dataUd, dataRg


PPS_TO_ANGLE = (2 * np.pi) / 3600
HOURS_TO_ANGLE = (2 * np.pi) / 12
TIME_TO_ANGLE = (2 * np.pi) / (24 * 3600)


@logged
def clean_dataSc(dataSc, acqfreq=1, diff_pps=False, correct_pps=True, fix_pps=False, correct_time=True, obsdate=None):

    freq_keys = [key for key in dataSc if key.endswith("_freq")]
    if freq_keys:
        clean_dataSc._log.debug("Cleaning *_freq")
        for key in freq_keys:
            freq = dataSc.get(key)
            mask = freq == 0
            if np.any(mask) and np.any(~mask):
                freq[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), freq[~mask])
            dataSc[key] = freq

    pps_keys = [key for key in dataSc if key.endswith("_time_pps")]
    if pps_keys:
        clean_dataSc._log.debug("Using {} to compute median pps time".format(pprint_list(pps_keys, "_time_pps")))
        shape = dataSc.get(pps_keys[0]).shape
        pps = [dataSc.get(key).flatten() for key in pps_keys]
        bad_pps = np.array(pps) == 0
        pps = np.ma.array(pps, mask=bad_pps)
        pps = np.ma.median(pps, axis=0)

        if diff_pps:
            pps_diff = {"pps-{}".format(key): (pps - dataSc[key].flatten()) * 1e6 for key in pps_keys}
            pps_diff["pps_diff"] = np.asarray(list(pps_diff.values())).max(axis=0).reshape(shape)

            dataSc.update(pps_diff)

        # Fake pps time if necessary,
        if correct_pps:
            clean_dataSc._log.info("Correcting pps time")

            pps = np.unwrap(pps * PPS_TO_ANGLE) / PPS_TO_ANGLE
            dummy = np.append(1 / acqfreq, np.diff(pps))
            bad = np.abs(1 / dummy - acqfreq) / acqfreq > 2 / 100
            # To avoid large jumps....
            bad = binary_closing(bad, iterations=4096)
            if any(bad):
                sample = dataSc["sample"].flatten()
                pps[bad] = interp1d(sample[~bad], pps[~bad])(sample[bad])

        if fix_pps:
            clean_dataSc._log.info("Fixing pps time")
            pps = np.unwrap(pps * PPS_TO_ANGLE) / PPS_TO_ANGLE
            sample = dataSc["sample"].flatten()
            # Two step polynomial fit to robustely remove outliers
            p = np.polynomial.polynomial.polyfit(sample, pps, 1)
            residual = pps - np.polynomial.polynomial.polyval(sample, p)
            med, std = mad_med(residual)
            bad = np.abs(residual - med) / std > 3
            p = np.polynomial.polynomial.polyfit(sample[~bad], pps[~bad], 1)
            pps = np.polynomial.polynomial.polyval(sample, p)

        pps %= 3600

        dataSc["time_pps"] = pps.reshape(shape)

    # Compute median hours
    hours_keys = [key for key in dataSc if key.endswith("_hours")]
    if hours_keys:
        clean_dataSc._log.debug("Using {} to compute median hours".format(pprint_list(hours_keys, "_hours")))
        shape = dataSc.get(hours_keys[0]).shape
        hours = [dataSc.get(key).flatten() for key in hours_keys]
        bad_hours = np.array(hours) == 0  # Beware that this flag midgnigth
        hours = np.ma.array(hours, mask=bad_hours, fill_value=0)
        hours = np.ma.median(hours, axis=0).filled(0)  # masked value goes back to midnight

        if correct_time:
            clean_dataSc._log.info("Correcting hours jumps")
            # When crossing midgnigth, there is a middle point at 12...
            hours = np.unwrap(hours * HOURS_TO_ANGLE) / HOURS_TO_ANGLE
            # Remove up to 64 sample jumps... hours should be monotonous
            # hours = median_filter(hours, size=64)

        hours %= 24  # For concerto hours is given as yearly hours, so safer here

        dataSc["hours"] = hours.reshape(shape)

    # Compute time if possible
    if "time_pps" in dataSc and "hours" in dataSc and obsdate is not None:

        shape = dataSc["time_pps"].shape
        time_pps = dataSc["time_pps"].flatten()
        hours = dataSc["hours"].flatten()

        # seconds from the beginning of the day
        time = hours * 3600 + time_pps

        # Wrapping around midnight
        time = np.unwrap(time * TIME_TO_ANGLE) / TIME_TO_ANGLE

        # Correct jumps in time
        if correct_time:
            clean_dataSc._log.info("Correcting time differences greather than {} s".format(float(correct_time)))
            bad = np.append(np.diff(np.unwrap(time)), 0) > correct_time
            if any(bad) and any(~bad):
                # Interpolate bad values
                idx = np.arange(time.shape[0])
                func = interp1d(idx[~bad], np.unwrap(time)[~bad], kind="linear", fill_value="extrapolate")
                time[bad] = func(idx[bad])

        dataSc["obstime"] = Time(obsdate + time.reshape(shape) * u.s, format="mjd")

    return dataSc


@logged
def clean_param_c(param_c, flat=False):

    # Decode the name
    nomexp_keys = sorted([key for key in param_c.keys() if "nomexp" in key], key=lambda x: int(x[6:]))
    param_c["nomexp"] = ""
    clean_param_c._log.debug("Merging {} nomexp field".format(len(nomexp_keys)))
    for key in nomexp_keys:
        param_c["nomexp"] += param_c[key].tobytes().strip(b"\x00").decode("ascii")
        del param_c[key]

    # Decode the IPs :
    for key in param_c.keys():
        if "_ip" in key:
            param_c[key] = decode_ip(param_c[key])

    # from ./Acquisition/instrument/kid_amc/server/TamcServer.cpp
    # -1 --> 4KHz   0 -> 2 kHz   1 -> 1 kHz  40=23Hz
    param_c["div_kid"] = int(param_c["div_kid"])
    # AB: Private Comm
    div_kid = param_c["div_kid"] + 2 if param_c["div_kid"] < 1 else param_c["div_kid"] * 4
    param_c["acqfreq"] = 5.0e8 / 2.0 ** 17 / div_kid

    # Regroup keys per box
    if flat is False:
        box_delimiters = [".", "-", "_"]
        box_keys = set([key[0] for key in param_c.keys() if key[1] in box_delimiters])
        for box_key in box_keys:
            items = {}
            for key in [key for key in param_c.keys() if box_key == key[0]]:
                items[key[2:]] = param_c[key]
                del param_c[key]
            param_c[box_key] = items

    return param_c


def clean_param_d(param_d):

    # typedet is actually a Utype typedet (TconfigNika.h & public_def.h) , ie a union of either a int32 val or a struct with 4 8 bit int.
    param_d["typedet"], param_d["masqdet"], param_d["acqbox"], param_d["array"] = (
        param_d["typedet"].view(np.byte).reshape(param_d["typedet"].shape[0], 4).T
    )

    # WHY ?
    param_d["frequency"] = param_d["frequency"].astype(np.float) * 10

    # Add a default index column
    if "index" not in param_d:
        param_d["index"] = np.arange(len(param_d["namedet"]))

    # Build the kidpar
    kidpar = Table(param_d, masked=True)

    # Reorder some columns
    colnames = kidpar.colnames
    for i, colname in enumerate(["index", "namedet"]):
        colnames.remove(colname)
        colnames.insert(i, colname)
    kidpar = kidpar[colnames]

    # by default mask all kids
    kidpar["index"].mask = True
    kidpar.add_index("namedet")

    return kidpar


def clean_dataSd(dataSd, shift_rf_didq=-49):
    if "RF_didq" in dataSd and shift_rf_didq is not None:

        shape = dataSd["RF_didq"].shape
        dataSd["RF_didq"] = np.roll(dataSd["RF_didq"].reshape(shape[0], -1), shift_rf_didq, axis=1).reshape(shape)

    return dataSd


@logged
def clean_namedet(namedet, fix=True):
    # In principle the last 2 bytes should be always 0, ie we code up to 6 character, instead of 8/16, but sometimes....
    namedet = namedet.view(np.byte)
    if np.any(namedet[:, 6:]):
        if fix:
            namedet[:, 6:] = 0
            clean_namedet._log.warning("Corrupted namedet truncated to 6 characters")
        else:
            clean_namedet._log.warning("Corrupted namedet")
    return [name.tobytes().strip(b"\x00").decode("ascii") for name in namedet]


def clean_position_unit(dataXc):
    """From milli radians to degrees"""
    for ckey in [
        "F_azimuth",
        "F_elevation",
        "F_tel_Az",
        "F_tel_El",
        "F_tl_Az",
        "F_tl_El",
        "F_sky_Az",
        "F_sky_El",
        "F_diff_Az",
        "F_diff_El",
    ]:
        if ckey in dataXc:
            dataXc[ckey] = np.rad2deg(dataXc[ckey] / 1000.0)

    # /data/PHOTOM/Processing/Pipeline/Readdata/C/v1/brut_to_data.c
    for ckey in [
        "antLST",
        "antxoffset",
        "antyoffset",
    ]:
        if ckey in dataXc:
            dataXc[ckey] = np.float32(dataXc[ckey] / 1e8)
    for ckey in [
        "antAz",
        "antEl",
    ]:
        if ckey in dataXc:
            dataXc[ckey] = np.rad2deg(dataXc[ckey] / 1e8)
    for ckey in ["antMJD", "antMJDf"]:
        if ckey in dataXc:
            dataXc[ckey] /= 1e-9
    for ckey in ["antactualAz", "antactualEl", "anttrackAz", "anttrackEl"]:
        if ckey in dataXc:
            dataXc[ckey] = np.rad2deg(dataXc[ckey] / (np.pi / 180 / 3600 * 9.0 / 1024.0))


def clean_raw_unit(dataXc):
    # Some raw quantities are stored with different units from read_nika_all
    unit_keys = []
    for key in dataXc:
        if key == "sample":
            dataXc[key] = np.uint32(dataXc[key])
        elif key.endswith("_pps") or key.endswith("_ntp"):
            # This avoid rounding errors from microsecond to second
            dataXc[key] = np.float64(np.uint32(dataXc[key]) * 1e-6)
        elif key.endswith("_freq"):
            dataXc[key] = np.float64(np.uint32(dataXc[key]) * 1e-2)
        elif key.endswith("_n_mes"):
            dataXc[key] = np.float64(np.int32(dataXc[key]) * 3600)
        elif key.endswith("_pos"):
            dataXc[key] = np.float64(np.uint32(dataXc[key]) * 1e-3)
        elif ":" in key:
            unit_keys.append(key)
    for key in unit_keys:
        _key, exp = key.split(":")
        if exp == "":
            exp = "0"
        if exp[0] == "e":
            exp = exp[1:]
        dataXc[_key] = np.float64(dataXc.pop(key) / 10 ** int(exp))

    clean_position_unit(dataXc)


def namept_to_names(name, nbname):
    return [
        item.tobytes().split(b"\x00")[0].decode()
        for item in np.ctypeslib.as_array(
            name,
            shape=(
                nbname,
                16,
            ),
        )
    ]


P_INT32 = ctypes.POINTER(ctypes.c_int32)
P_CHAR = ctypes.POINTER(ctypes.c_char)


class TdataPt(ctypes.Structure):
    _fields_ = [
        ("nbpt", ctypes.c_int32),
        ("nbph", ctypes.c_int32),
        ("nbBloc", ctypes.c_int32),
        ("nbBlocInFile", ctypes.c_int32),
        ("nbBlocRead", ctypes.c_int32),
        ("nbReglageRead", ctypes.c_int32),
        ("nbSample", ctypes.c_int64),
        ("nbDet", ctypes.c_int32),
        ("nbReglage", ctypes.c_int32),
        ("nbParanC", ctypes.c_int32),
        ("nbParanD", ctypes.c_int32),
        ("nbDataSc", ctypes.c_int32),
        ("nbDataSd", ctypes.c_int32),
        ("nbDataUc", ctypes.c_int32),
        ("nbDataUd", ctypes.c_int32),
        ("nbDataRg", ctypes.c_int32),
        ("headerPt", P_INT32),
        ("nameParamC", P_CHAR),
        ("nameParamD", P_CHAR),
        ("nameDataSc", P_CHAR),
        ("nameDataSd", P_CHAR),
        ("nameDataUc", P_CHAR),
        ("nameDataUd", P_CHAR),
        ("nameDet", P_CHAR),
        ("numBlocRg", P_INT32),
        ("ptParamC", P_INT32),
        ("ptParamD", P_INT32),
        ("ptDataSc", P_INT32),
        ("ptDataSd", P_INT32),
        ("ptDataUc", P_INT32),
        ("ptDataUd", P_INT32),
        ("ptDataPh", P_INT32),
        ("ptDataRg", P_INT32),
    ]


"""
filename = "/data/KISS/Raw/nika2c-data3/KISS/2020_05_27/X20200527_0438_S1050_Jupiter_ITERATIVERASTER"
filename = "/data/CONCERTO/InLab/Data/CNC040_X/X1_2020_12_09/X15_13_Tablebt_scanStarted_6"  # very small
filename = "/data/CONCERTO/InLab/Data/CNC041_X/X1_2020_12_18/X14_32_Tablebt_scanStarted_5"  #  big
filename = "/data/CONCERTO/InLab/Data/CNC041_X/X1_2020_12_18/X10_43_Tablebt_scanStarted_2"  # huge
output = read_raw(filename)
"""


class CODEDET(IntEnum):
    all = -1
    one = -2
    array = -100
    arrayt = -101
    arrayr = -102
    array_one = -200
    arrayt_one = -201
    arrayr_one = -202
    box = -1000
    box_one = -2000


def list_detector_to_codedet(list_detector=None):

    if list_detector is None:
        list_detector = "one"

    codeDet = None

    if isinstance(list_detector, str):
        list_detector = list_detector.lower()
        codeDet = getattr(CODEDET, list_detector.lower(), None)

        if list_detector.startswith("array_one") and codeDet is None:
            array = int(list_detector[9:])
            codeDet = CODEDET.array_one + array
        elif list_detector.startswith("array") and codeDet is None:
            array = int(list_detector[5:])
            codeDet = CODEDET.array + array
        elif list_detector.startswith("box_one"):
            box = list_detector[7:]
            try:
                box = int(box)
            except ValueError:
                # box is a str, try to convert that to box index starting from a = 1
                box = ord(box) - ord("a") + 1
            codeDet = CODEDET.box_one + box
        elif list_detector.startswith("box"):
            box = list_detector[3:]
            try:
                box = int(box)
            except ValueError:
                # box is a str, try to convert that to box index starting from a = 1
                box = ord(box) - ord("a") + 1
            codeDet = CODEDET.box + box
    elif isinstance(list_detector, (list, np.ndarray)):
        codeDet = len(list_detector)
        list_detector = np.int32(list_detector)

    if codeDet is None:
        raise ValueError("list_detector cat not be parsed")

    if codeDet < 0:
        list_detector = np.array([], dtype=np.int32)

    return codeDet, list_detector


@logged
def read_raw(
    filename,
    list_data=None,
    list_detector=None,
    start=None,
    end=None,
    fix=True,
    flat=True,
    diff_pps=False,
    correct_pps=False,
    fix_pps=True,
    correct_time=False,
    silent=True,
):
    """Read Raw data from the binary file using the `read_raw` C function.

    Parameters
    ----------
    filename : str
        Full filename
    list_data : list of string, or  'all'
        A list containing the data to be read within ('Sd'|'Sc'|'Uc'|'Ud') or the string 'all' to read all data
    list_detector : :class:`~numpy.array`
        The names of detectors to read, see Notes, by default `None` read all KIDs of type 1.
    start : int
        The starting block, default 0.
    end : type
        The ending block, default full available dataset.
    silent : bool
        Silence the output of the C library. The default is True
    diff_pps: bool
        pre-compute pps time differences. The default is False
    correct_pps: bool
        correct the pps signal. The default is False
    fix_pps: bool
        remplace the pps time by a semi-robust polynomial fit, True
    correct_time: bool or float
        correct the time signal by interpolating jumps higher that given value in second. The default is False

    Returns
    -------
    tconfigheader: namedtuple
        a copy of the TconfigHeader of the file
    param_c: dict
        the common variables for the file
    kidpar: :class:`~astropy.table.Table`
        the kidpar of the file
    TName: namedtuple
        the full name of the requested data and detectors
    nb_samples_read : int
        The number of sample read
    namedet: array of str
        the names of the read detectors
    dataSc : dict:
        A dictionnary containing all the requested sampled common quantities data as 2D :class:`~numpy.array` of shape (n_bloc, nb_pt_bloc)
    dataSd : dict
        A dictionnary containing all the requested sampled data as 3D :class:`~numpy.array` of shape (n_det, n_bloc, nb_pt_bloc)
    dataUc : dict:
        A dictionnary containing all the requested under-sampled common quantities data as 1D :class:`~numpy.array` of shape (n_bloc,)
    dataUd : dict
        A dictionnary containing all the requested under-sampled data as 2D :class:`~numpy.array` of shape (n_det, n_bloc)
    dataRg : dict
        A dictionnary containing all the reglage in the file, key is the bloc number and items is a 1D :class:`~numpy.array` of shape (n_det)

    Notes
    -----
    `list_detector` is either a list or array of detector names within the `kidpar` of the file or
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
    """

    assert Path(filename).exists(), "{} does not exist".format(filename)
    assert Path(filename).stat().st_size != 0, "{} is empty".format(filename)

    if list_data is None or (isinstance(list_data, str) and list_data.lower() == "all"):
        list_data = ["Sc", "Sd", "Uc", "Ud", "Ph", "Rg"]

    for item in list_data:
        assert item in [
            "Sc",
            "Sd",
            "Uc",
            "Ud",
            "Ph",
            "Rg",
        ], "{} can not be read with read_raw ['Sc', 'Sd', 'Uc', 'Ud', 'Rg']".format(item)

    codeDet, list_detector = list_detector_to_codedet(list_detector=list_detector)

    # default output
    param_c = kidpar = None
    dataSc = dataSd = dataUc = dataUd = dataRg = {}

    # Number of blocks to read
    start = start or 0
    nb_to_read = end - start if (end is not None) and (end > start) else -1  # read all

    c_read_raw = READRAW.readRawx
    c_read_raw.argtype = [
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_int32,
        P_INT32,
        ctypes.c_int,
    ]
    c_read_raw.restype = TdataPt

    read_raw._log.debug("before read_raw")
    Tdata = c_read_raw(
        bytes(str(filename), "ascii"),
        start,
        nb_to_read,
        "Sc" in list_data,
        "Sd" in list_data,
        "Uc" in list_data,
        "Ud" in list_data,
        "Ph" in list_data,
        "Rg" in list_data,
        codeDet,
        list_detector.ctypes.data_as(P_INT32),
        silent,
    )
    read_raw._log.debug("after read_raw")

    if Tdata.nbDet == 0:
        read_raw._log.error("No kid read, check list_detector")
        return None

    if nb_to_read == -1:
        nb_to_read = Tdata.nbBlocInFile

    if Tdata.nbBlocRead != nb_to_read:
        read_raw._log.warning("{} missing blocks, truncating".format(nb_to_read - Tdata.nbBlocRead))

    nb_read_samples = Tdata.nbBlocRead * Tdata.nbpt
    good_blocs = slice(0, Tdata.nbBlocRead)

    # See Acquisition/Library/configNika/TconfigNika.h
    header = TconfigHeader(*np.ctypeslib.as_array(Tdata.headerPt, (22,))[4:22])

    # ParamC
    read_raw._log.debug("ParamC")
    names = namept_to_names(Tdata.nameParamC, Tdata.nbParanC)
    values = np.ctypeslib.as_array(Tdata.ptParamC, (Tdata.nbParanC,))
    # TODO: Potentially make a copy here....
    param_c = dict(zip(names, values))
    del (names, values)
    param_c = clean_param_c(param_c, flat=flat)

    # Special treatment for namedet as some character are ill placed
    namedet = np.ctypeslib.as_array(
        Tdata.nameDet,
        shape=(
            Tdata.nbDet,
            16,
        ),
    )
    namedet = clean_namedet(namedet, fix=fix)

    # Reglage
    # F_tone ??
    if "Rg" in list_data:
        read_raw._log.debug("dataRg")
        bloc = np.ctypeslib.as_array(Tdata.numBlocRg, (Tdata.nbReglage,))
        values = np.ctypeslib.as_array(Tdata.ptDataRg, (Tdata.nbReglage, Tdata.nbDet)).astype(np.float32)
        dataRg = dict(zip(bloc, values))
        del (bloc, values)

    # ParamD
    read_raw._log.debug("ParamD")
    names = namept_to_names(Tdata.nameParamD, Tdata.nbParanD)
    values = np.ctypeslib.as_array(Tdata.ptParamD, (Tdata.nbParanD, Tdata.nbDet))
    param_d = dict(zip(names, values))
    del (names, values)
    # Adding the namedet
    param_d["namedet"] = namedet

    # Rename some columns
    for old_name, new_name in [("type", "typedet"), ("detnum", "index")]:
        if old_name in param_d:
            param_d[new_name] = param_d.pop(old_name)

    kidpar = clean_param_d(param_d)
    # with readraw the kidpar contains only the read data
    kidpar["index"].mask = False

    # dataSc
    # the .astype(np.float32) force a copy since we defined pointers on int32
    # the values need to be freed together if one want to free one
    TName_dataSc = namept_to_names(Tdata.nameDataSc, Tdata.nbDataSc)
    if "Sc" in list_data:
        read_raw._log.debug("dataSc")

        shape = (Tdata.nbDataSc, Tdata.nbBloc, Tdata.nbpt)
        values = np.ctypeslib.as_array(Tdata.ptDataSc, shape)[:, good_blocs, :].astype(np.int32)
        dataSc = dict(zip(TName_dataSc, values))
        del values
        clean_raw_unit(dataSc)
        obsdate = filename_to_meta(filename)["OBSDATE"]
        dataSc = clean_dataSc(dataSc, param_c["acqfreq"], diff_pps, correct_pps, fix_pps, correct_time, obsdate=obsdate)

    # dataSd
    # the .astype(np.float32) force a copy since we defined pointers on int32
    # the values need to be freed together if one want to free one
    TName_dataSd = namept_to_names(Tdata.nameDataSd, Tdata.nbDataSd)
    if "Sd" in list_data:
        read_raw._log.debug("dataSd")
        shape = (Tdata.nbDataSd, Tdata.nbDet, Tdata.nbBloc, Tdata.nbpt)
        values = np.ctypeslib.as_array(Tdata.ptDataSd, shape)[:, :, good_blocs, :].astype(np.float32)
        dataSd = dict(zip(TName_dataSd, values))
        del values
        dataSd = clean_dataSd(dataSd)

    # dataUc
    # the .astype(np.float32) force a copy since we defined pointers on int32
    # the values need to be freed together if one want to free one
    TName_dataUc = namept_to_names(Tdata.nameDataUc, Tdata.nbDataUc)
    if "Uc" in list_data:
        read_raw._log.debug("dataUc")
        shape = (Tdata.nbDataUc, Tdata.nbBloc)
        values = np.ctypeslib.as_array(Tdata.ptDataUc, shape)[:, good_blocs].astype(np.float32)
        dataUc = dict(zip(TName_dataUc, values))
        del values
        clean_raw_unit(dataUc)

    # dataUd
    # the .astype(np.float32) force a copy since we defined pointers on int32
    # the values need to be freed together if one want to free one
    TName_dataUd = namept_to_names(Tdata.nameDataUd, Tdata.nbDataUd)
    if "Ud" in list_data:
        read_raw._log.debug("dataUd")
        shape = (Tdata.nbDataUd, Tdata.nbDet, Tdata.nbBloc)
        values = np.ctypeslib.as_array(Tdata.ptDataUd, shape)[:, :, good_blocs].astype(np.float32)
        dataUd = dict(zip(TName_dataUd, values))
        del values

    names = TName(TName_dataSc, TName_dataSd, TName_dataUc, TName_dataUd, namedet)

    # Release C memory
    c_read_raw_intfree = READRAW.readRawFreeIntMalloc
    c_read_raw_intfree.argtype = [TdataPt]

    c_read_raw_extfree = READRAW.readRawFreeExtAlloc
    c_read_raw_extfree.argtype = [TdataPt]

    c_read_raw_intfree(Tdata)
    c_read_raw_extfree(Tdata)

    return header, param_c, kidpar, names, nb_read_samples, namedet, dataSc, dataSd, dataUc, dataUd, dataRg


# https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def filename_or_h5py_file(func=None, mode="r"):
    """Allow a filename/Path or a h5py.File instance as first argument.

    The decorator will instantiate an h5py.File, open and close the file for the function if needed.

    Parameters
    ----------
    mode : str,
        the mode to potentially open the hdf5 file, by default 'r'
    """

    if func is None:
        return partial(filename_or_h5py_file, mode=mode)

    @wraps(func)
    def wrapper(*args, **kwargs):
        filename = args[0]

        if isinstance(filename, (str, Path)):
            f = h5py.File(filename, mode=mode)
        else:
            f = filename
        output = func(f, *args[1:], **kwargs)

        if isinstance(filename, (str, Path)):
            f.close()

        return output

    return wrapper


@filename_or_h5py_file(mode="a")
def _to_hdf5(parent_group, key, data, **kwargs):  # noqa: C901
    """Save data to hdf5

    Parameters
    ----------
    parent_group : `h5py.Group`
        the parent group to save the dataset
    key : str
        the name of the dataset
    data : (np.array|astropy.table.Table|int|float|namedtuple|tuple|list|dict)
        the corresponding dataset

    Notes
    -----
    the dataset can only be a combinaison of numpy arrays, tuple, list or dictionaries

    Any keyword supported by h5py.create_dataset can be passed, usually :
    kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    """

    if isinstance(data, np.ndarray) and not isinstance(data, np.ma.MaskedArray):
        # numpy arrays
        if key in parent_group:
            del parent_group[key]
        # Special case for unicode strings
        if "U" in data.dtype.str:
            data = data.astype("S")
        return parent_group.create_dataset(key, data=data, **kwargs)

    if isinstance(data, Table):
        # astropy table
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_group(key)

        write_table_hdf5(data, parent_group, path=key, append=True, overwrite=True, serialize_meta=True)
        dset = parent_group[key]
        dset.attrs["type"] = "Table"
        return dset

    if isinstance(data, Time):
        # astropy Time data
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_dataset(key, data=data.to_value("mjd"), **kwargs)
        dset.attrs["type"] = "Time"
        dset.attrs["scale"] = data.scale
        dset.attrs["format"] = data.format
        return dset

    if isinstance(data, (str, int, np.int, np.int32, np.int64, float, np.float, np.float32, np.float64)):
        # Scalar are put into the attribute of the group
        if key in parent_group.attrs:
            del parent_group.attrs[key]
        parent_group.attrs[key] = data
        return None

    # list of same type
    if isinstance(data, list) and all([isinstance(_data, type(data[0])) for _data in data[1:]]):
        if isinstance(data[0], (str, np.str_)):
            # strings:
            dt = h5py.string_dtype()
            dset = parent_group.create_dataset(key, data=np.array(data, dtype="S"), dtype=dt)
        else:
            dset = parent_group.create_dataset(key, data=np.array(data))
        dset.attrs["type"] = "list-array"
        return dset

    if data is None:
        if key in parent_group.attrs:
            del parent_group.attrs[key]
        return parent_group.create_dataset(key, dtype="f")

    if isinstance(data, (tuple, list, dict, np.ma.MaskedArray)):
        if key not in parent_group:
            sub_group = parent_group.create_group(key)
            sub_group.attrs["type"] = type(data).__name__
        else:
            sub_group = parent_group[key]
    else:
        raise TypeError("Can not handle type of {} :  {}".format(key, type(data)))

    if isnamedtupleinstance(data):
        _to_hdf5(sub_group, str(type(data)), data._asdict(), **kwargs)
        sub_group.attrs["type"] = "namedtuple"
    elif isinstance(data, (tuple, list)):
        for i, item in enumerate(data):
            _to_hdf5(sub_group, str(i), item, **kwargs)
    elif isinstance(data, dict):
        # Force string keys.... will cause problem for KidsData__dataRg
        for _key in data:
            _to_hdf5(sub_group, str(_key), data[_key], **kwargs)
    elif isinstance(data, np.ma.MaskedArray):
        sub_group.attrs["fill_value"] = getattr(data, "fill_value")
        for _key in ["data", "mask"]:
            _to_hdf5(sub_group, _key, getattr(data, _key), **kwargs)


@filename_or_h5py_file
def _from_hdf5(parent_group, key, array=None):
    """Read any dataset saved by _to_hdf5

    Parameters
    ----------
    parent_group : `h5py.Group`
        the parent group to read the dataset from
    key : str
        the name of the dataset
    array : func
        apply this function to the returned Dataset, by default None

    Returns
    -------
    data : (np.array|astropy.table.Table|tuple|list|dict)
        the corresponding dataset

    Notes
    -----
    Choice of array function could be np.array or dask.array.from_array
    """
    if key not in parent_group:
        raise ValueError("{} not in {}".format(key, parent_group.name))

    item = parent_group.get(key)
    if isinstance(item, h5py.Dataset):
        if item.shape == ():
            # single float/int no shape
            return item[()]
        elif item.shape is None:
            return None

        if "type" in item.attrs:
            # Special types
            if item.attrs["type"] == "list-array":
                data = list(item[:])
                if len(data) > 0 and isinstance(data[0], bytes):
                    data = [item.decode() for item in data]
                return data
            elif item.attrs["type"] == "Table":
                return read_table_hdf5(item.parent, path=item.name, character_as_bytes=False)
            elif item.attrs["type"] == "Time":
                data = Time(item[:], format="mjd", scale=item.attrs["scale"])
                data.format = item.attrs["format"]
                return data
        else:
            # numpy arrays returned as references (need [:] to get the data)
            if array is not None:
                return array(item)
            else:
                return item

    elif isinstance(item, h5py.Group):
        _type = item.attrs.get("type", None)
        if _type == "tuple":
            return (_from_hdf5(item, key, array=array) for key in item)
        elif _type == "list":
            return [_from_hdf5(item, key, array=array) for key in item]
        elif _type in ["dict", "OrderedDict"]:
            dict_ = {key: _from_hdf5(item, key, array=array) for key in item}
            dict_attrs = dict(item.attrs.items())
            del dict_attrs["type"]  # Remove special metadata
            dict_.update(dict_attrs)
            return dict_
        elif _type == "namedtuple":
            key = list(item.keys())[0]
            data_asdict = _from_hdf5(item, key, array=array)
            key = key.split("'")[1]

            class_ = _import_from(key)
            return class_(**data_asdict)
        elif _type == "MaskedArray":
            return np.ma.array(
                _from_hdf5(item, "data"), mask=_from_hdf5(item, "mask"), fill_value=item.attrs["fill_value"]
            )
        else:
            raise ValueError("Unknown type for {}: {}".format(item.name, _type))


@filename_or_h5py_file
def read_info_hdf5(filename):
    """Read header information from an hdf5 file.

    Parameters
    ----------
    filename : str or ~h5py.File
        input filename or h5py.File instance

     Returns
    -------
    tconfigheader: namedtuple
        a copy of the TconfigHeader of the file, or None if not present
    param_c: dict
        the common variables for the file, or None if not present
    kidpar: :class:`~astropy.table.Table`
        the kidpar of the file, or None if not present
    TName: namedtuple
        the full name of the requested data and detectors,, or None if not present
    nb_read_sample: int
        the total  number of sample in the file, or None if not present

    """

    nb_read_samples = filename.attrs.get("nb_read_samples", None)

    header = _from_hdf5(filename, "header") if "header" in filename else None
    param_c = _from_hdf5(filename, "param_c") if "param_c" in filename else None
    names = _from_hdf5(filename, "names") if "names" in filename else None
    kidpar = _from_hdf5(filename, "kidpar") if "kidpar" in filename else None

    # kidpar must be a masked table and have an index
    if "index" not in kidpar.keys():
        kidpar["index"] = np.arange(len(kidpar))
    if not kidpar.has_masked_values:
        kidpar = Table(kidpar, masked=True, copy=False)
    kidpar.add_index("namedet")

    return header, param_c, kidpar, names, nb_read_samples


@filename_or_h5py_file
def read_all_hdf5(filename, array=np.array):
    """Short summary.

    Parameters
    ----------
    filename : str or ~h5py.File
        input filename or h5py.File instance
    array : function, (np.array|dask.array.from_array|None) optional
        function to apply to the largest cached value, by default np.array, if None return h5py.Dataset

    Returns
    -------
    nb_samples_read : int
        The number of sample read
    dataSc : dict:
        A dictionnary containing all the sampled common quantities data as 2D :class:`~numpy.array` of shape (n_bloc, nb_pt_bloc), {} if not present
    dataSd : dict
        A dictionnary containing all the sampled data as 3D arrays of shape (n_det, n_bloc, nb_pt_bloc), {} if not present
    dataUc : dict:
        A dictionnary containing all the under-sampled common quantities data as 1D :class:`~numpy.array` of shape (n_bloc,), {} if not present
    dataUd : dict
        A dictionnary containing all the under-sampled data as 2D :class:`~numpy.array` of shape (n_det, n_bloc), {} if not present
    extended_kidpar : ~astropy.table.Table
        The extended kidpar found in the file, None if not present
    """
    datas = []
    # Force np.array for small items, and array for the large fully sample data
    for item, _array in zip(
        ["dataSc", "dataSd", "dataUc", "dataUd", "dataRg"], [np.array, array, np.array, np.array, np.array]
    ):
        datas.append(_from_hdf5(filename, item, array=_array) if item in filename else {})

    nb_samples_read = filename.attrs.get("nb_read_samples", None)

    list_detector = _from_hdf5(filename, "list_detector", array=np.array) if "list_detector" in filename else None

    extended_kidpar = _from_hdf5(filename, "extended_kidpar") if "extended_kidpar" in filename else None

    output = nb_samples_read, list_detector, *datas, extended_kidpar

    return output


def info_to_hdf5(filename, header, param_c, kidpar, names, nb_read_samples, file_kwargs=None, **kwargs):
    """Save all header information to hdf5 attribute or group."""

    if file_kwargs is None:
        file_kwargs = {}

    with h5py.File(filename, **file_kwargs) as f:
        _to_hdf5(f, "header", header, **kwargs)
        _to_hdf5(f, "param_c", param_c, **kwargs)
        _to_hdf5(f, "names", names, **kwargs)
        _to_hdf5(f, "kidpar", kidpar, **kwargs)
        f.attrs["nb_read_samples"] = nb_read_samples


def data_to_hdf5(
    filename, list_detector, dataSc, dataSd, dataUc, dataUd, dataRg, extended_kidpar, file_kwargs=None, **kwargs
):
    """save data to hdf5"""

    if file_kwargs is None:
        file_kwargs = {}

    with h5py.File(filename, **file_kwargs) as f:

        if list_detector is not None:
            _to_hdf5(f, "list_detector", list_detector)

        for item, data in zip(
            ["dataSc", "dataSd", "dataUc", "dataUd", "dataRg"], [dataSc, dataSd, dataUc, dataUd, dataRg]
        ):
            if data is not None:
                _to_hdf5(f, item, data, **kwargs)

        if extended_kidpar is not None:
            _to_hdf5(f, "extended_kidpar", extended_kidpar, **kwargs)


if __name__ == "__main__":
    # input = "/data/KISS/Raw/nika2c-data3/KISS/X20190319_0727_S0230_Jupiter_SCIENCEMAP"
    input = "/data/KISS/Raw/nika2c-data3/KISS/2019_11_16/Y20191116_1611_S0636_Jupiter_ITERATIVERASTER"
    # output = "myfile.hdf5"
    # TODO: Save read_info output
    header, param_c, kidpar, names, nb_read_samples = read_info(input)
    # TODO: Some of the data must be computed on-line... check that in C library
    Data = [
        "u_ph_IQ",
        "u_ph_rel",
        "ph_IQ",
        "logampl",
        "ph_rel",
        "amp_dIdQ",
        "amp_pIQ",
        "rap_pIQdIQ",
        "F_tone",
        "k_width",
        "k_flag",
        "dF_tone",
        "k_angle",
        "RF_didq",
        "RF_deco",
        "dF_tone",
        "amplitude",
        "ph_rel",
        "k_angle",
        "u_freq",
        "u_ph_IQ",
        "u_ph_rel",
        "sample_U",
        "dI",
        "dQ",
        "sampleU",
    ]

    list_data = names.DataSc + names.DataSd + names.DataUc + names.DataUd
    for item in Data:
        if item in list_data:
            list_data.remove(item)

    nb_sample_read, list_detector, dataSc, dataSd, dataUc, dataUd = read_all(input, list_data=list_data)

    # file_kwargs = {'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    # info_to_hdf5(output, header, param_c, kidpar, names, nb_read_samples, file_kwargs=None)
    # data_to_hdf5(output, list_detector, dataSc, dataSd, dataUc, dataUd, None, file_kwargs=None)

    import deepdish as dd

    data = {
        "header": header,
        "param_c": param_c,
        "kidpar": kidpar,
        "names": names,
        "nb_read_samples": nb_read_samples,
        "dataSc": dataSc,
        "dataSd": dataSd,
        "dataUc": dataUc,
        "dataUd": dataUd,
    }
    dd.io.save("test_dd_blosc.h5", data, compression=("blosc", 9))
    dd.io.save("test_dd_zlib.h5", data, compression=("zlib", 9))
    dd.io.save("test_dd_bzip2.h5", data, compression=("bzip2", 9))
    dd.io.save("test_dd_gzip.h5", data, compression=("gzip", 9))
