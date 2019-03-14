#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ctypes
import numpy as np
from collections import namedtuple
from astropy.table import Table, MaskedColumn

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# TODO: This should not be fixed here
#libpath = '/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/'
#readnikadata = ctypes.cdll.LoadLibrary(libpath + '/libreadnikadata.so')

# For my mac
libpath = '../../NIKA_lib_AB_OB_gui/Readdata/C/'
readnikadata = ctypes.cdll.LoadLibrary(libpath + '/libreadnikadata.dylib')

p_int32 = ctypes.POINTER(ctypes.c_int32)
p_double = ctypes.POINTER(ctypes.c_double)

read_nika_info = readnikadata.read_nika_info
read_nika_info.argtypes = [ctypes.c_char_p,
                           p_int32, ctypes.c_int32,
                           ctypes.c_char_p, ctypes.c_int32,
                           ctypes.c_char_p,
                           ctypes.c_int32, p_int32,
                           p_int32, p_int32, p_int32, p_int32,
                           p_int32, p_int32,
                           ctypes.c_int]

read_nika_all = readnikadata.read_nika_all
read_nika_all.argtype = [ctypes.c_char_p,
                         p_double, p_double,
                         ctypes.c_char_p,
                         p_int32,
                         ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Defining TconfigHeader following Acquisition/Library/configNika/TconfigNika.h
TconfigHeader = namedtuple('TconfigHeader',
                           ['size_MotorModulTable',
                            'nb_brut_ud',
                            'nb_boites_mesure',
                            'nb_detecteurs',
                            'nb_pt_bloc',
                            'nb_sample_fichier',
                            'nb_det_mini',
                            'nb_brut_uc',
                            'version_header',
                            'nb_param_c',
                            'nb_param_d',
                            'lg_header_util',
                            'nb_brut_c',
                            'nb_brut_d',
                            'nb_brut_periode',
                            'nb_data_c',
                            'nb_data_d',
                            'nb_champ_reglage'])

# Tname list the name of the computed/requested variables and detectors
TName = namedtuple('TName',
                   ['ComputedDataSc',
                    'ComputedDataSd',
                    'ComputedDataUc',
                    'ComputedDataUd',
                    'RawDataDetector'])


detectors_code = {'kid': 3, 'kod': 2, 'all': 1, 'a1': 4, 'a2': 5, 'a3': 6}

# @profile


def read_info(filename, det2read='KID', list_data='all', silent=True):
    """Read header information from a binary file.

    Parameters
    ----------
    filename : str
        Full filename
    det2read : str (kid, kod, all, a1, a2, a3)
        Detector type to read
    list_data : str ('all', '...')
        A string containing the list of data to be read, could be 'all'
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

    codelistdet = detectors_code.get(det2read.lower(), -1)

    # nb_char_nom = 16
    length_header = 300000
    var_name_length = 200000
    nb_max_det = 8001

    buffer_header = np.zeros(length_header, dtype=np.int32)
    var_name_buffer = ctypes.create_string_buffer(var_name_length)
    list_detector = np.zeros(nb_max_det, dtype=np.int32)
    nb_Sc = ctypes.c_int32()
    nb_Sd = ctypes.c_int32()
    nb_Uc = ctypes.c_int32()
    nb_Ud = ctypes.c_int32()
    idx_param_c = ctypes.c_int32()
    idx_param_d = ctypes.c_int32()

    nb_read_samples = read_nika_info(bytes(filename, 'ascii'),
                                     buffer_header.ctypes.data_as(p_int32), length_header,
                                     var_name_buffer, var_name_length,
                                     bytes(list_data, 'ascii'),
                                     codelistdet, list_detector.ctypes.data_as(p_int32),
                                     nb_Sc, nb_Sd, nb_Uc, nb_Ud,
                                     idx_param_c, idx_param_d,
                                     silent)
    nb_Sc = nb_Sc.value
    nb_Sd = nb_Sd.value
    nb_Uc = nb_Uc.value
    nb_Ud = nb_Ud.value
    idx_param_c = idx_param_c.value
    idx_param_d = idx_param_d.value

    # See Acquisition/Library/configNika/TconfigNika.h
    header = TconfigHeader(*(buffer_header[4:22].tolist()))

    nb_detectors = list_detector[0]
    idx_detectors = list_detector[1:list_detector[0] + 1]

    version_header = header.version_header // 65536

    # Proper way to do it :
    # var_name = [var_name_buffer.raw[ind: ind+16].strip(b'\x00').decode('ascii') for ind in range(0, len(var_name_buffer.raw), 16)]
    # Faster way :
    var_name = [name for name in var_name_buffer.raw.decode('ascii').split('\x00') if name != '']

    # Retrieve the param commun
    idx = 0
    name_param_c = var_name[idx:idx + header.nb_param_c]
    val_param_c = buffer_header[idx_param_c:idx_param_c + header.nb_param_c]
    param_c = dict(zip(name_param_c, val_param_c))
    for key in ['nomexp1', 'nomexp2', 'nomexp3', 'nomexp4']:
        param_c[key] = param_c[key].tobytes().strip(b'\x00').decode('ascii')

    # Not later in the original code, bugged anyway
    param_c['acqfreq'] = 5.0e8 / 2.0**19 / param_c['div_kid']

    # Retrieve the param detector
    idx += header.nb_param_c
    name_param_d = var_name[idx: idx + header.nb_param_d]
    val_param_d = buffer_header[idx_param_d:idx_param_d + header.nb_param_d * header.nb_detecteurs].reshape(header.nb_param_d, header.nb_detecteurs)
    param_d = dict(zip(name_param_d, val_param_d))

    # nom1 & nom2 are actually defined as a struct Tname8 namedet (TconfingNika.h), ie char[8] ie 64 bit (public_def.h)
    param_d['namedet'] = [name.tobytes().strip(b'\x00').decode('ascii')
                          for name in np.append(param_d['nom1'], param_d['nom2']).reshape(header.nb_detecteurs, 2).view(np.byte)]
    for key in ['nom1', 'nom2']:
        del param_d[key]

    # typedet is actually a Utype typedet (TconfigNika.h), ie a union of either a int32 val or a struct with 4 8 bit int.
    param_d['typedet'], param_d['masqdet'], param_d['acqbox'], param_d['array'] = param_d['typedet'].view(
        np.byte).reshape(header.nb_detecteurs, 4).T

    # WHY ?
    param_d['frequency'] *= 10

    # Build the kidpar
    kidpar = Table(param_d, masked=True)
    detector_mask = np.ones(header.nb_detecteurs, dtype=np.bool)
    detector_mask[idx_detectors] = False
    kidpar.add_column(MaskedColumn(np.arange(header.nb_detecteurs),
                                   mask=detector_mask), index=0, name='index')
    kidpar.add_index('namedet')

    idx += header.nb_param_d
    name_data_Sc = var_name[idx: idx + nb_Sc]
    idx += nb_Sc
    name_data_Sd = var_name[idx: idx + nb_Sd]
    # name_data_Sd.append('flag') # WHY ???
    idx += nb_Sd
    name_data_Uc = var_name[idx: idx + nb_Uc]
    idx += nb_Uc
    name_data_Ud = var_name[idx: idx + nb_Ud]
    idx += nb_Ud
    name_detectors = var_name[idx: idx + nb_detectors]

    names = TName(name_data_Sc, name_data_Sd, name_data_Uc,
                  name_data_Ud, name_detectors)

    del(buffer_header, var_name_buffer, list_detector)

    return header, version_header, param_c, kidpar, names, nb_read_samples


def read_all(filename, det2read='KID', list_data='sample indice C_laser1_pos', 
             list_detector=None, start=None, end=None, silent=True, correct_pps=False):
    """Short summary.

    Parameters
    ----------
    filename : str
        Full filename
    det2read : str {kid, kod, all, a1, a2, a3}
        Detector type to read
    list_data : str ('all', '...')
        A string containing the list of data to be read, could be 'all'
    list_detector : :class:`~numpy.array`
        The list of detector to read, by default `None` read all available KIDs.
    start : int
        The starting block, default 0.
    end : type
        The ending block, default full available dataset.
    silent : bool
        Silence the output of the C library. The default is True
    correct_pps: bool
        correct the pps signal. The default is False

    Returns
    -------
    dataSc : dict:
        A dictionnary containing all the requested sampled common quantities data as 1D :class:`~numpy.array`
    dataSd : dict
        A dictionnary containing all the requested sampled data as 2D :class:`~numpy.array`
    dataUc : dict:
        A dictionnary containing all the requested under-sampled common quantities data as 1D :class:`~numpy.array`
    dataUd : dict
        A dictionnary containing all the requested under-sampled data as 2D :class:`~numpy.array`

    Notes
    -----
    `list_detector` is an :class:`~numpy.array` of int listing the index of the requested KIDs. The first element being the length of the array

    """
    # Read the basic header from the file and the name of the data
    header, _, param_c, kidpar, names, nb_read_info = read_info(filename,
                                                                det2read=det2read,
                                                                list_data=list_data,
                                                                silent=silent)

    if list_detector is None:
        list_detector = np.where(~kidpar['index'].mask)[0]
        list_detector = np.append(list_detector.shape, list_detector).astype(np.int32)

    assert len(names.RawDataDetector) >= list_detector[0]

    # Number of blocks to read
    start = start or 0
    end = end or nb_read_info
    nb_to_read = end - start

    # Number of data to read
    nb_Sc = len(names.ComputedDataSc)
    nb_Sd = len(names.ComputedDataSd)
    nb_Uc = len(names.ComputedDataUc)
    nb_Ud = len(names.ComputedDataUd)
    nb_detectors = list_detector[0]

    np_pt_bloc = header.nb_pt_bloc

    _sample_S = nb_Sc + nb_Sd * nb_detectors
    _sample_U = nb_Uc + nb_Ud * nb_detectors

    buffer_dataS = np.zeros(nb_to_read * _sample_S,
                            dtype=np.double)
    buffer_dataU = np.zeros(nb_to_read // np_pt_bloc * _sample_U,
                            dtype=np.double)

    nb_samples_read = read_nika_all(bytes(filename, 'ascii'),
                                    buffer_dataS.ctypes.data_as(p_double),
                                    buffer_dataU.ctypes.data_as(p_double),
                                    bytes(list_data, 'ascii'),
                                    list_detector.ctypes.data_as(p_int32),
                                    start, end, silent)

    assert nb_samples_read == nb_to_read, "Did not read all requested data"

    _dataSc = buffer_dataS.reshape(nb_to_read, _sample_S)[:, 0:nb_Sc].T
    _dataSd = np.moveaxis(
        buffer_dataS.reshape(nb_to_read, _sample_S)[:, nb_Sc:]
        .reshape(nb_to_read, nb_Sd, nb_detectors),
        0, -1)

    _dataUc = buffer_dataU.reshape(
        nb_to_read // np_pt_bloc, _sample_U)[:, 0:nb_Uc].T
    _dataUd = np.moveaxis(
        buffer_dataU.reshape(nb_to_read // np_pt_bloc, _sample_U)[:, nb_Uc:]
        .reshape(nb_to_read // np_pt_bloc, nb_Ud, nb_detectors),
        0, -1)

    # Split the data  by name, here data are 1D or 2D numpy array
    _dataSc = {name: data for name, data in zip(names.ComputedDataSc, _dataSc)}
    _dataSd = {name: data for name, data in zip(names.ComputedDataSd, _dataSd)}
    _dataUc = {name: data for name, data in zip(names.ComputedDataUc, _dataUc)}
    _dataUd = {name: data for name, data in zip(names.ComputedDataUd, _dataUd)}

    # Shift RF_didq if present
    if 'RF_didq' in _dataSd:
        shift_rf_didq = -49
        _dataSd['RF_didq'] = np.roll(_dataSd['RF_didq'], shift_rf_didq, axis=1)

    # Compute time pps_time difference
    if 'A_time' in _dataSc:
        pps = _dataSc['A_time']
        other_time = [key for key in _dataSc if key.endswith(
            '_time') and key != 'A_time']
        if other_time and 'sample' in _dataSc:
            pps_diff = {
                'A_time-{}'.format(key): (pps - _dataSc[key]) * 1e6 for key in other_time}
            pps_diff['pps_diff'] = np.asarray(
                list(pps_diff.values())).max(axis=0)

            _dataSc = {**_dataSc, **pps_diff}

        # Fake pps time if necessary
        if correct_pps:
            dummy = np.diff(pps, append=0)
            good = np.abs(dummy - 1 / param_c['acqfreq']) < 0.02
            if any(~good):
                param = np.polyfit(_dataSc['sample'][good], pps[good], 1)
                pps[~good] = np.polyval(param, _dataSc['sample'][~good])

        _dataSc = {**_dataSc, 'pps': pps}

    del(buffer_dataS, buffer_dataU)
    return _dataSc, _dataSd, _dataUc, _dataUd
