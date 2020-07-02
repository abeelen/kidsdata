# pylint: disable=C0301,C0103
import os
import gc
import h5py
from functools import partial, wraps

import logging
import warnings
import ctypes
import importlib
import numpy as np
from pathlib import Path
from collections import namedtuple
from astropy.table import Table, MaskedColumn
from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# TODO: This should not be fixed here
NIKA_LIB_PATH = os.getenv("NIKA_LIB_PATH", "/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/")
NIKA_LIB_SO = Path(NIKA_LIB_PATH) / "libreadnikadata.so"

assert NIKA_LIB_SO.exists(), (
    "Could not find NIKA LIB so File [%s] \n" % NIKA_LIB_SO
    + "You might have forgotten to set $NIKA_LIB_PATH or compile the library"
)

READNIKADATA = ctypes.cdll.LoadLibrary(str(NIKA_LIB_SO))

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

# Tname list the name of the computed/requested variables and detectors
TName = namedtuple("TName", ["ComputedDataSc", "ComputedDataSd", "ComputedDataUc", "ComputedDataUd", "RawDataDetector"])


DETECTORS_CODE = {"kid": 3, "kod": 2, "all": 1, "a1": 4, "a2": 5, "a3": 6}


def decode_ip(int32_val):
    """Decode an int32 bit into 4 8 bit int."""
    bin = np.binary_repr(int32_val, width=32)
    int8_arr = [int(bin[0:8], 2), int(bin[8:16], 2), int(bin[16:24], 2), int(bin[24:32], 2)]
    return int8_arr


# @profile
def read_info(filename, det2read="KID", list_data="all", fix=True, silent=True):
    """Read header information from a binary file.

    Parameters
    ----------
    filename : str
        Full filename
    det2read : str (kid, kod, all, a1, a2, a3)
        Detector type to read
    list_data : list or 'all'
        A list containing the list of data to be read, or the string 'all'
    fix : boolean
        fix the end of the detector names
    silent : bool (default:True)
        Silence the output of the C library

    Returns
    -------
    tconfigheader: namedtuple
        a copy of the TconfigHeader of the file
    version_header: int
        the version of the header
    param_c: dict
        the common variables for the file
    kidpar: :class:`~astropy.table.Table`
        the kidpar of the file
    tname: namedtuple
        the full name of the requested data and detectors
    nb_read_sample: int
        the total  number of sample in the file

    """
    if list_data == "all":
        str_data = list_data
    else:
        str_data = " ".join(list_data)

    codelistdet = DETECTORS_CODE.get(det2read.lower(), -1)

    # nb_char_nom = 16
    length_header = 300000
    var_name_length = 200000
    nb_max_det = 8001

    p_int32 = ctypes.POINTER(ctypes.c_int32)

    read_nika_info = READNIKADATA.read_nika_info
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

    version_header = header.version_header // 65536

    # Proper way to do it :
    # var_name = [var_name_buffer.raw[ind: ind+16].strip(b'\x00').decode('ascii') for ind in range(0, len(var_name_buffer.raw), 16)]
    # Faster way :
    var_name = [name for name in var_name_buffer.raw.decode("ascii").split("\x00") if name != ""]

    # Retrieve the param commun
    idx = 0
    name_param_c = var_name[idx : idx + header.nb_param_c]
    val_param_c = buffer_header[idx_param_c : idx_param_c + header.nb_param_c]
    param_c = dict(zip(name_param_c, val_param_c))

    # Decode the name
    param_c["nomexp"] = ""
    for key in ["nomexp1", "nomexp2", "nomexp3", "nomexp4"]:
        param_c["nomexp"] += param_c[key].tobytes().strip(b"\x00").decode("ascii")
        del param_c[key]

    # Decode the IPs :
    for key in param_c.keys():
        if "_ip" in key:
            param_c[key] = ".".join([str(int8) for int8 in decode_ip(param_c[key])])

    # from ./Acquisition/instrument/kid_amc/server/TamcServer.cpp
    if param_c["div_kid"] > 0:
        param_c["div_kid"] *= 4
    else:
        param_c["div_kid"] += 2  # -1 --> 4KHz   0 -> 2 kHz   1 -> 1 kHz

    # Note: it was done later in the original code, bugged anyway
    param_c["acqfreq"] = 5.0e8 / 2.0 ** 19 / param_c["div_kid"]

    # Retrieve the param detector
    idx += header.nb_param_c
    name_param_d = var_name[idx : idx + header.nb_param_d]
    val_param_d = buffer_header[idx_param_d : idx_param_d + header.nb_param_d * header.nb_detecteurs].reshape(
        header.nb_param_d, header.nb_detecteurs
    )
    param_d = dict(zip(name_param_d, val_param_d))

    # nom1 & nom2 are actually defined as a struct Tname8 namedet (TconfingNika.h), ie char[8] ie 64 bit (public_def.h)
    namedet = np.append(param_d["nom1"], param_d["nom2"]).reshape(header.nb_detecteurs, 2).view(np.byte)

    # In principle the last 2 bytes should be always 0, ie we code up to 6 character, instead of 8, but sometimes....
    if np.any(namedet[:, 6:8]):
        if fix:
            namedet[:, 6:8] = 0
            warnings.warn("Corrupted namedet truncated to 6 characters")
        else:
            warnings.warn("Corrupted namedet")

    param_d["namedet"] = [name.tobytes().strip(b"\x00").decode("ascii") for name in namedet]
    for key in ["nom1", "nom2"]:
        del param_d[key]

    # typedet is actually a Utype typedet (TconfigNika.h), ie a union of either a int32 val or a struct with 4 8 bit int.
    param_d["typedet"], param_d["masqdet"], param_d["acqbox"], param_d["array"] = (
        param_d["typedet"].view(np.byte).reshape(header.nb_detecteurs, 4).T
    )

    # WHY ?
    param_d["frequency"] *= 10

    # Build the kidpar
    kidpar = Table(param_d, masked=True)
    detector_mask = np.ones(header.nb_detecteurs, dtype=np.bool)
    detector_mask[idx_detectors] = False
    kidpar.add_column(MaskedColumn(np.arange(header.nb_detecteurs), mask=detector_mask), index=0, name="index")
    kidpar.add_index("namedet")

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

    return header, version_header, param_c, kidpar, names, nb_read_samples


def read_all(
    filename,
    det2read="KID",
    list_data=["indice", "A_masq", "I", "Q"],
    list_detector=None,
    start=None,
    end=None,
    silent=True,
    correct_pps=False,
    ordering="K",
):
    """Read Raw data from the binary file using the `read_nika_all` C function.

    Parameters
    ----------
    filename : str
        Full filename
    det2read : str {kid, kod, all, a1, a2, a3}
        Detector type to read
    list_data : list or 'all'
        A list containing the list of data to be read, or the string 'all'
    list_detector : :class:`~numpy.array`
        The names of detectors to read, by default `None` read all available KIDs.
    start : int
        The starting block, default 0.
    end : type
        The ending block, default full available dataset.
    silent : bool
        Silence the output of the C library. The default is True
    correct_pps: bool
        correct the pps signal. The default is False
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

    Notes
    -----
    `list_detector` is an :class:`~numpy.array` of int listing the index of the requested KIDs.

    """

    if list_data == "all":
        str_data = list_data
    else:
        str_data = " ".join(list_data)

    # Read the basic header from the file and the name of the data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header, _, param_c, kidpar, names, nb_read_info = read_info(
            filename, det2read=det2read, list_data=list_data, silent=silent
        )

    if list_detector is None:
        list_detector = np.where(~kidpar["index"].mask)[0]
    else:
        list_detector = np.array(kidpar.loc_indices[list_detector])

    # Append the number of detectors as expected by the C library
    list_detector = np.append(len(list_detector), list_detector).astype(np.int32)

    assert len(names.RawDataDetector) >= list_detector[0]

    # Number of blocks to read
    start = (start or 0) * header.nb_pt_bloc
    end = (end or (nb_read_info // header.nb_pt_bloc)) * header.nb_pt_bloc
    nb_to_read = end - start

    p_int32 = ctypes.POINTER(ctypes.c_int32)
    p_double = ctypes.POINTER(ctypes.c_double)

    read_nika_all = READNIKADATA.read_nika_all
    read_nika_all.argtype = [
        ctypes.c_char_p,
        p_double,
        p_double,
        ctypes.c_char_p,
        p_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    # Number of data to read
    nb_Sc = len(names.ComputedDataSc)
    nb_Sd = len(names.ComputedDataSd)
    nb_Uc = len(names.ComputedDataUc)
    nb_Ud = len(names.ComputedDataUd)
    nb_detectors = list_detector[0]

    np_pt_bloc = header.nb_pt_bloc

    _sample_S = nb_Sc + nb_Sd * nb_detectors
    _sample_U = nb_Uc + nb_Ud * nb_detectors

    buffer_dataS = np.zeros(nb_to_read * _sample_S, dtype=np.double)
    buffer_dataU = np.zeros(nb_to_read // np_pt_bloc * _sample_U, dtype=np.double)

    logging.info(
        "buffer allocated : \n    - buffer_dataS  {} MiB\n    - buffer_dataU  {} MiB".format(
            buffer_dataS.nbytes / 1024 / 1024, buffer_dataU.nbytes / 1024 / 1024
        )
    )
    logging.debug("before read_nika_all")
    nb_samples_read = read_nika_all(
        bytes(str(filename), "ascii"),
        buffer_dataS.ctypes.data_as(p_double),
        buffer_dataU.ctypes.data_as(p_double),
        bytes(str_data, "ascii"),
        list_detector.ctypes.data_as(p_int32),
        start,
        end,
        silent,
    )
    logging.debug("after read_nika_all")

    if nb_samples_read != nb_to_read:
        logging.warning("Did not read all requested data")

        buffer_dataS = buffer_dataS[0 : nb_samples_read * _sample_S]
        buffer_dataU = buffer_dataU[0 : nb_samples_read // np_pt_bloc * _sample_U]

    # Split the buffer into common and data part with proper shape
    _dataSc = buffer_dataS.reshape(nb_samples_read, _sample_S)[:, 0:nb_Sc].T
    _dataSd = np.moveaxis(
        buffer_dataS.reshape(nb_samples_read, _sample_S)[:, nb_Sc:].reshape(nb_samples_read, nb_Sd, nb_detectors), 0, -1
    )
    del buffer_dataS  # Do not relase actual memory here...

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

    dataSc = {name: data.reshape(-1, np_pt_bloc).copy() for name, data in zip(names.ComputedDataSc, _dataSc)}
    del _dataSc
    dataSd = {
        name: data.reshape(nb_detectors, -1, np_pt_bloc).astype(np.float32, order=ordering, casting="unsafe")
        for name, data in zip(names.ComputedDataSd, _dataSd)
    }
    del _dataSd

    dataUc = {name: data.copy() for name, data in zip(names.ComputedDataUc, _dataUc)}
    del _dataUc
    dataUd = {
        name: data.reshape(nb_detectors, -1).astype(np.float32, order=ordering, casting="unsafe")
        for name, data in zip(names.ComputedDataUd, _dataUd)
    }
    del _dataUd

    # Shift RF_didq if present
    if "RF_didq" in dataSd:
        shift_rf_didq = -49
        dataSd["RF_didq"] = np.roll(dataSd["RF_didq"].reshape(nb_detectors, -1), shift_rf_didq, axis=1).reshape(
            dataSd["RF_didq"].shape
        )

    # Convert units azimuth and elevation to degrees
    for ckey in ["F_azimuth", "F_elevation", "F_tl_Az", "F_tl_El", "F_sky_Az", "F_sky_El", "F_diff_Az", "F_diff_El"]:
        for data in [dataSc, dataUc]:
            if ckey in data:
                data[ckey] = np.rad2deg(data[ckey] / 1000.0)

    # if "F_diff_Az" in dataUc:
    #     logging.warn("Correcting diff corrdinates")
    #     dataUc['F_diff_Az'] = dataUc['F_diff_Az'] * np.cos(np.radians(dataUc['F_sky_El']))

    # Compute time pps_time difference
    if "A_time" in dataSc:
        pps = dataSc["A_time"]
        other_time = [key for key in dataSc if key.endswith("_time") and key != "A_time"]
        if other_time and "sample" in dataSc:
            pps_diff = {"A_time-{}".format(key): (pps - dataSc[key]) * 1e6 for key in other_time}
            pps_diff["pps_diff"] = np.asarray(list(pps_diff.values())).max(axis=0)

            dataSc.update(pps_diff)

        # Fake pps time if necessary
        if correct_pps:
            dummy = np.append(np.diff(pps), 0)
            good = np.abs(dummy - 1 / param_c["acqfreq"]) < 0.02
            if any(~good):
                param = np.polyfit(dataSc["sample"][good], pps[good], 1)
                pps[~good] = np.polyval(param, dataSc["sample"][~good])

        dataSc["pps"] = pps

    gc.collect()
    ctypes._reset_cache()

    return nb_samples_read, dataSc, dataSd, dataUc, dataUd


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

        if isinstance(filename, str):
            f.close()

        return output

    return wrapper


@filename_or_h5py_file(mode="a")
def _to_hdf5(parent_group, key, data, **kwargs):
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
    if isinstance(data, np.ndarray):
        # numpy arrays
        if key in parent_group:
            del parent_group[key]
        return parent_group.create_dataset(key, data=data, **kwargs)

    if isinstance(data, Table):
        # astropy table
        if key in parent_group:
            del parent_group[key]
        write_table_hdf5(data, parent_group, path=key, append=True, overwrite=True, serialize_meta=True)
        dset = parent_group[key]
        dset.attrs["type"] = "Table"
        return dset

    if isinstance(data, (str, int, np.int, np.int32, np.int64, float, np.float, np.float32, np.float64)):
        # Scalar are put into the attribute of the group
        if key in parent_group.attrs:
            del parent_group.attrs[key]
        parent_group.attrs[key] = data
        return None

    # list of same type
    if isinstance(data, list) and all([isinstance(_data, type(data[0])) for _data in data[1:]]):
        if isinstance(data[0], str):
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

    if isinstance(data, (tuple, list, dict)):
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
        for _key in data:
            _to_hdf5(sub_group, _key, data[_key], **kwargs)


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
                # originally a list
                return list(item[:])
            elif item.attrs["type"] == "Table":
                return read_table_hdf5(item.parent, path=item.name, character_as_bytes=False)
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

            # Extract module and class from the key
            module = ".".join(key.split(".")[:-1])
            class_ = key.split(".")[-1]

            module = importlib.import_module(module)
            class_ = getattr(module, class_)
            return class_(**data_asdict)
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
    version_header: int
        the version of the header,, or None if not present
    param_c: dict
        the common variables for the file, or None if not present
    kidpar: :class:`~astropy.table.Table`
        the kidpar of the file, or None if not present
    tname: namedtuple
        the full name of the requested data and detectors,, or None if not present
    nb_read_sample: int
        the total  number of sample in the file, or None if not present

    """

    version_header = filename.attrs.get("version_header", None)
    nb_read_samples = filename.attrs.get("nb_read_samples", None)

    header = _from_hdf5(filename, "header") if "header" in filename else None
    param_c = _from_hdf5(filename, "param_c") if "param_c" in filename else None
    names = _from_hdf5(filename, "names") if "names" in filename else None
    kidpar = _from_hdf5(filename, "kidpar") if "kidpar" in filename else None

    return header, version_header, param_c, kidpar, names, nb_read_samples


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
    for item, _array in zip(["dataSc", "dataSd", "dataUc", "dataUd"], [np.array, array, np.array, np.array]):
        datas.append(_from_hdf5(filename, item, array=_array) if item in filename else {})

    extended_kidpar = _from_hdf5(filename, "extended_kidpar") if "extended_kidpar" in filename else None

    return (*datas, extended_kidpar)


def info_to_hdf5(filename, header, version_header, param_c, kidpar, names, nb_read_samples, file_kwargs=None, **kwargs):
    """Save all header information to hdf5 attribute or group."""

    if file_kwargs is None:
        file_kwargs = {}

    with h5py.File(filename, **file_kwargs) as f:
        _to_hdf5(f, "header", header, **kwargs)
        _to_hdf5(f, "param_c", param_c, **kwargs)
        _to_hdf5(f, "names", names, **kwargs)
        _to_hdf5(f, "kidpar", kidpar, **kwargs)
        f.attrs["nb_read_samples"] = nb_read_samples
        f.attrs["version_header"] = version_header


def data_to_hdf5(filename, dataSc, dataSd, dataUc, dataUd, extended_kidpar, file_kwargs=None, **kwargs):
    """save data to hdf5"""

    if file_kwargs is None:
        file_kwargs = {}

    with h5py.File(filename, **file_kwargs) as f:
        for item, data in zip(["dataSc", "dataSd", "dataUc", "dataUd"], [dataSc, dataSd, dataUc, dataUd]):
            if data is not None:
                _to_hdf5(f, item, data, **kwargs)

        if extended_kidpar is not None:
            _to_hdf5(f, "extended_kidpar", extended_kidpar, **kwargs)


if __name__ == "__main__":
    # input = "/data/KISS/Raw/nika2c-data3/KISS/X20190319_0727_S0230_Jupiter_SCIENCEMAP"
    input = "/data/KISS/Raw/nika2c-data3/KISS/2019_11_16/Y20191116_1611_S0636_Jupiter_ITERATIVERASTER"
    # output = "myfile.hdf5"
    # TODO: Save read_info output
    header, version_header, param_c, kidpar, names, nb_read_samples = read_info(input)
    # TODO: Some of the data must be computed on-line... check that in C library
    ComputedData = [
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

    list_data = names.ComputedDataSc + names.ComputedDataSd + names.ComputedDataUc + names.ComputedDataUd
    for item in ComputedData:
        if item in list_data:
            list_data.remove(item)

    nb_sample_read, dataSc, dataSd, dataUc, dataUd = read_all(input, list_data=list_data)

    # for dataset_name, dataset in zip(["dataSc", "dataSd", "dataUc", "dataUd"], [dataSc, dataSd, dataUc, dataUd]):
    #     data_to_hdf5(output, dataset_name, dataset)

    import deepdish as dd

    data = {
        "header": header,
        "version_header": version_header,
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
