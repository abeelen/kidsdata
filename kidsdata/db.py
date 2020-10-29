import os
import re
import numpy as np
import logging
from pathlib import Path
from itertools import chain
from datetime import datetime
from functools import wraps, partial, update_wrapper

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, join, vstack, unique, MaskedColumn, Column
from astropy.utils.console import ProgressBar


BASE_DIRS = [Path(directory) for directory in os.getenv("DATA_DIR", "/data/KISS/Raw/nika2c-data3/KISS").split(";")]
DB_DIR = Path(os.getenv("DB_DIR", "."))

DB_SCAN_FILE = DB_DIR / ".kidsdb_scans.fits"
DB_EXTRA_FILE = DB_DIR / ".kidsdb_extra.fits"
DB_PARAM_FILE = DB_DIR / ".kidsdb_param.fits"
DB_TABLE_FILE = DB_DIR / ".kidsdb_table.fits"

__all__ = ["list_scan", "get_scan", "list_extra", "get_extra", "list_table", "get_filename"]

RE_DIR = re.compile(r"X(\d*)_(\d{4,4})_(\d{2,2})_(\d{2,2})$")

# Regular scans :
RE_SCAN = re.compile(r"^X(\d{8,8})_(\d{4,4})_S(\d{4,4})_([\w|\+]*)_(\w*)$")

# Extra files : X_2020_10_22_12h53m20_AA_man :
RE_EXTRA = re.compile(r"^X_(\d{4,4})_(\d{2,2})_(\d{2,2})_(\d{2,2})h(\d{2,2})m(\d{2,2})_AA_man$")

# CONCERTO InLab test : X14_04_Tablebt_scanStarted_10 :
RE_TABLE = re.compile(r"X(\d{2,2})_(\d{2,2})_Tablebt_scanStarted_(\d*)$")


def scan_columns(filename, re_pattern=None):
    date, hour, scan, source, obsmode = re_pattern.match(filename.name).groups()
    dtime = Time(datetime.strptime(" ".join([date, hour]), "%Y%m%d %H%M").isoformat())
    scan = int(scan)
    return {
        "date": dtime,
        "scan": scan,
        "source": source,
        "obsmode": obsmode,
    }


def extra_columns(filename, re_pattern=None):
    time_data = [int(item) for item in re_pattern.match(filename.name).groups()]
    return {"date": Time(datetime(*time_data).isoformat())}


def table_columns(filename, re_pattern=None):
    hour, minute, scan = re_pattern.match(filename.name).groups()
    _, year, month, day = RE_DIR.match(filename.parent.name).groups()
    dtime = Time(datetime.strptime(" ".join([year, month, day, hour, minute]), "%Y %m %d %H %M").isoformat())
    return {"date": dtime, "scan": scan}


def extend_database():
    """Read the header of each file and construct the parameter database."""
    ## WIP
    global DB_SCAN, DB_PARAM

    from .kids_rawdata import KidsRawData  # To avoid import loop

    data_rows = []
    param_rows = {}
    for item in ProgressBar(DB_SCAN):
        if "param_id" in item.colnames and item["param_id"]:
            # Skip if present
            continue
        filename = item["filename"]
        kd = KidsRawData(filename)
        hash_param = hash(str(kd.param_c))
        data_row = {"filename": filename, "param_id": hash_param}
        param_row = {"param_id": hash_param}
        param_row.update(kd.param_c)
        data_rows.append(data_row)
        param_rows[hash_param] = param_row
        del kd

    if len(param_rows) > 0:
        # We found new scans and/or new parameters
        param_rows = [*param_rows.values()]

        # Get unique parameter list
        param_set = set(chain(*[param.keys() for param in param_rows] + [DB_PARAM.colnames if DB_PARAM else []]))

        # Fill missing value
        missing = []
        for param in param_rows:
            for key in param_set:
                if key not in param:
                    param[key] = None
                    missing.append(key)
        missing = set(missing)

        NEW_PARAM = Table(param_rows)
        for key in missing:
            mask = NEW_PARAM[key] == None  # noqa: E711
            _dtype = type(NEW_PARAM[key][~mask][0])
            NEW_PARAM[key][mask] = _dtype(0)
            NEW_PARAM[key] = MaskedColumn(_dtype(NEW_PARAM[key].data), mask=mask)

        if DB_PARAM is not None:
            DB_PARAM = vstack([DB_PARAM, NEW_PARAM])
            DB_PARAM = unique(DB_PARAM, "param_id")
        else:
            DB_PARAM = NEW_PARAM

        DB_PARAM.write(DB_PARAM_FILE)

        # Update DB_SCAN
        NEW_PARAM = Table(data_rows)
        if "param_id" in DB_SCAN.colnames:
            DB_SCAN.add_index("filename")
            idx = DB_SCAN.loc_indices[NEW_PARAM["filename"]]
            DB_SCAN["param_id"][idx] = NEW_PARAM["param_id"]
        else:
            # a simple join
            DB_SCAN = join(DB_SCAN, NEW_PARAM, keys="filename", join_type="outer")

        DB_SCAN.sort("ctime")

        DB_PARAM.write(DB_PARAM_FILE, overwrite=True)
        DB_SCAN.write(DB_SCAN_FILE, overwrite=True)


def get_scan(scan=None):
    """Get filename of corresponding scan number.

    Parameters
    ----------
    scan : int or str
        the scan number to retrieve

    Returns
    -------
    filename : str
       the full path of the file
    """
    global DB_SCAN
    DB_SCAN.update()

    try:
        scan = int(scan)
    except ValueError as ex:
        raise ValueError("{} can not be converted to int: {}".format(scan, ex))

    mask = DB_SCAN["scan"] == scan

    if not np.any(mask):
        raise IndexError("Scan {} not found".format(scan))

    return DB_SCAN[mask]["filename"].data[0]


def get_extra(start=None, end=None):
    """Get path for extra scans (typically skydips or dome observation) between two timestamp.

    Parameters
    ----------
    start, end: datetime
        beginning and end of integration

    Returns
    -------
    filename : list
        the list of corresponding files
    """
    global DB_EXTRA
    DB_EXTRA.update()

    mask = (DB_EXTRA["date"] > start) & (DB_EXTRA["date"] < end)
    return DB_EXTRA[mask]["filename"].data


def get_filename(filename=None):
    """Get path base on filenames.

    Parameters
    ----------
    filename : str
        the filename of the observation

    Returns
    -------
    filename : str
       the full path of the file
    """
    global DBs
    for db in DBs:
        db.update()

    merge_DB = vstack([db[["filename", "name"]] for db in DBs if db is not None and len(db) != 0])

    mask = merge_DB["name"] == filename

    if not np.any(mask):
        raise IndexError("Scan {} not found".format(filename))

    return merge_DB[mask]["filename"].data[0]


def list_data(database=None, pprint_columns=None, output=False, **kwargs):
    """List (with filtering) all data in the database.

    Notes
    -----
    You can filter the list see `list_data`
    """
    if database is not None:
        database.update()

    if len(database) == 0:
        raise ValueError("No scans found, check the DATA_DIR variable")

    # Default output, no filtering
    _database = database

    # Filtering on all possible key from table
    for key in kwargs.keys():
        if key.split("__")[0] in database.keys():
            if "__gt" in key:
                _database = database[database[key.split("__")[0]] > kwargs[key]]
            elif "__lt" in key:
                _database = database[database[key.split("__")[0]] < kwargs[key]]
            else:
                _database = database[database[key] == kwargs[key]]

    if output:
        return _database
    elif pprint_columns is not None:
        _database[pprint_columns].pprint(max_lines=-1, max_width=-1)
    else:
        _database.pprint(max_lines=-1, max_width=-1)


class KidsDB(Table):
    def __init__(self, *args, filename=None, dirs=None, re_pattern=None, extract_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.dirs = dirs
        self.re_pattern = re_pattern
        self.extract_func = extract_func

        if self.filename is not None and self.filename.exists():
            table_read_kwd = {"astropy_native": True, "character_as_bytes": False}
            this = Table.read(self.filename, **table_read_kwd)
            self.add_columns(this.columns)
            self._correct_time()

    def _correct_time(self):
        logging.debug("Converting time in DB table")
        # Ugly hack for Time columns
        for key in ["date", "mtime", "ctime"]:
            if key in self.columns:
                self[key] = Time(self[key])
                self[key].format = "iso"

        # Astropy 4.0.3 introduces a regression on this (#10824)
        try:
            self.sort("ctime")
        except AttributeError:
            indexes = np.argsort(self["ctime"])
            with self.index_mode("freeze"):
                for name, col in self.columns.items():
                    new_col = col.take(indexes, axis=0)
                    try:
                        col[:] = new_col
                    except Exception:
                        self[col.info.name] = new_col

    def update(self):
        """Fill the table."""
        filenames = [file for path in self.dirs for file in Path(path).glob("**/*") if self.re_pattern.match(file.name)]

        data_rows = []
        for filename in filenames:
            # Removing already scanned files
            if self.__len__() != 0 and str(filename) in self.columns["filename"]:
                continue

            stat = filename.stat()
            row = {
                "filename": filename.as_posix(),
                "name": filename.name,
                "size": (stat.st_size * u.byte).to(u.MB),
                "ctime": Time(datetime.fromtimestamp(stat.st_ctime).isoformat()),
                "mtime": Time(datetime.fromtimestamp(stat.st_mtime).isoformat()),
                # "comment": " " * 128
            }

            if self.extract_func is not None:
                row.update(self.extract_func(filename, self.re_pattern))

            data_rows.append(row)

        if len(data_rows) > 0:
            logging.info("Found {} new scans".format(len(data_rows)))

            if self.__len__() != 0:
                for row in data_rows:
                    self.add_row(row)
            else:
                # Create the columns, taking quantity into account
                columns = [
                    Column(
                        [
                            _[key].to(data_rows[0][key].unit).value if isinstance(_[key], u.Quantity) else _[key]
                            for _ in data_rows
                        ],
                        name=key,
                        unit=getattr(data_rows[0][key], "unit", None),
                    )
                    for key in data_rows[0].keys()
                ]

                self.add_columns(columns)
                self._correct_time()

                # Put size in MB for all scans
                self["size"].info.format = "7.3f"

                self.write(self.filename, overwrite=True)


DB_SCAN = KidsDB(filename=DB_SCAN_FILE, re_pattern=RE_SCAN, extract_func=scan_columns, dirs=BASE_DIRS)
DB_EXTRA = KidsDB(filename=DB_EXTRA_FILE, re_pattern=RE_EXTRA, extract_func=extra_columns, dirs=BASE_DIRS)
DB_TABLE = KidsDB(filename=DB_TABLE_FILE, re_pattern=RE_TABLE, extract_func=table_columns, dirs=BASE_DIRS)

DBs = [DB_SCAN, DB_EXTRA, DB_TABLE]

list_scan = partial(list_data, database=DB_SCAN, pprint_columns=["date", "scan", "source", "obsmode", "size"])
list_extra = partial(list_data, database=DB_EXTRA, pprint_columns=["name", "date", "size"])
list_table = partial(list_data, database=DB_TABLE, pprint_columns=["name", "date", "scan", "size"])
