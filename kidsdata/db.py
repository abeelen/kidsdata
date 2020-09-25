import os
import re
import numpy as np
import logging
from pathlib import Path
from itertools import chain
from datetime import datetime
from functools import wraps, partial

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, join, vstack, unique, MaskedColumn
from astropy.utils.console import ProgressBar


BASE_DIRS = [Path(os.getenv("KISS_DATA", "/data/KISS/Raw/nika2c-data3/KISS"))]
KISSDB_DIR = Path(os.getenv("KISSDB_DIR", "."))

DB_SCAN_FILE = KISSDB_DIR / ".kissdb_scans.fits"
DB_EXTRA_FILE = KISSDB_DIR / ".kissdb_extra.fits"
DB_PARAM_FILE = KISSDB_DIR / ".kissdb_param.fits"

table_read_kwd = {"astropy_native": True, "character_as_bytes": False}
DATABASE_SCAN = Table.read(DB_SCAN_FILE, **table_read_kwd) if DB_SCAN_FILE.exists() else None
DATABASE_EXTRA = Table.read(DB_EXTRA_FILE, **table_read_kwd) if DB_EXTRA_FILE.exists() else None
DATABASE_PARAM = Table.read(DB_PARAM_FILE, **table_read_kwd) if DB_PARAM_FILE.exists() else None

# If we read the data, the time column will be in float... fix that to iso format
logging.debug("Converting time in DB table")
for table in [DATABASE_SCAN, DATABASE_EXTRA]:
    for key in ["date", "ctime", "mtime"]:
        if (table is not None) and (key in table.colnames):
            table[key] = Time(table[key], format="iso")

__all__ = ["list_scan", "get_scan", "list_extra", "get_extra"]

RE_SCAN = re.compile(r"^X(\d{8,8})_(\d{4,4})_S(\d{4,4})_([\w|\+]*)_(\w*)$")
RE_EXTRA = re.compile(r"^X_(\d{4,4})_(\d{2,2})_(\d{2,2})_(\d{2,2})h(\d{2,2})m(\d{2,2})_AA_man$")


def update_scan_database(dirs=None):
    """Fill the scan database with the filenames.

    Parameters
    ----------
    dirs : list
        list of directories to scan
    """
    global DATABASE_SCAN

    if dirs is None:
        dirs = BASE_DIRS

    # Regular scan files, actually faster to make a glob with pattern here
    filenames = chain(*[Path(path).glob("**/X*_*_S????_*_*") for path in dirs])

    data_rows = []
    for filename in filenames:
        # Cleaning other type of files ?!?
        if filename.suffix in [".fits", ".hdf5"]:
            continue
        # Removing already scanned files
        if (DATABASE_SCAN is not None) and (filename.as_posix() in DATABASE_SCAN["filename"]):
            continue

        if not RE_SCAN.match(filename.name):
            continue
        date, hour, scan, source, obsmode = RE_SCAN.match(filename.name).groups()
        dtime = datetime.strptime(" ".join([date, hour]), "%Y%m%d %H%M")
        scan = int(scan)
        stat = filename.stat()
        # Do not add empty files
        if stat.st_size == 0:
            continue
        data_rows.append(
            (
                filename.as_posix(),
                dtime,
                scan,
                source,
                obsmode,
                stat.st_size,
                datetime.fromtimestamp(stat.st_ctime),
                datetime.fromtimestamp(stat.st_ctime),
            )
        )

    if len(data_rows) > 0:
        logging.info("Found {} new scans".format(len(data_rows)))
        NEW_SCAN = Table(
            names=["filename", "date", "scan", "source", "obsmode", "size", "ctime", "mtime"], rows=data_rows
        )
        for key in ["date", "ctime", "mtime"]:
            NEW_SCAN[key] = Time(NEW_SCAN[key]).iso
        NEW_SCAN.sort("date")
        NEW_SCAN["size"].unit = "byte"
        NEW_SCAN["size"] = NEW_SCAN["size"].to(u.MB)
        NEW_SCAN["size"].info.format = "7.3f"

        if DATABASE_SCAN is not None:
            # TODO: This fails on Time Column...
            DATABASE_SCAN = vstack([DATABASE_SCAN, NEW_SCAN])
            DATABASE_SCAN.sort("scan")
        else:
            DATABASE_SCAN = NEW_SCAN

        DATABASE_SCAN.write(DB_SCAN_FILE, overwrite=True)


def update_extra_database(dirs=None):
    """Fill the extra database with the filenames.

    Parameters
    ----------
    dirs : list
        list of directories to scan
    """
    global DATABASE_EXTRA

    if dirs is None:
        dirs = BASE_DIRS

    # Extra files for skydips
    filenames = chain(*[Path(path).glob("**/X_*_AA_man") for path in dirs])

    data_rows = []
    for filename in filenames:
        # Cleaning fits files ?!?
        if filename.suffix == ".fits":
            continue
        # Removing already scanned files
        if DATABASE_EXTRA is not None and filename not in DATABASE_EXTRA["filename"]:
            continue

        if not RE_EXTRA.match(filename.name):
            continue
        time_data = [int(item) for item in RE_EXTRA.match(filename.name).groups()]
        dtime = datetime(*time_data)
        stat = filename.stat()
        data_rows.append(
            (
                filename.as_posix(),
                filename.name,
                dtime,
                stat.st_size,
                datetime.fromtimestamp(stat.st_ctime),
                datetime.fromtimestamp(stat.st_ctime),
            )
        )

    if len(data_rows) > 0:
        logging.info("Found {} new extra scans".format(len(data_rows)))

        NEW_EXTRA = Table(names=["filename", "name", "date", "size", "ctime", "mtime"], rows=data_rows)
        for key in ["date", "ctime", "mtime"]:
            NEW_EXTRA[key] = Time(NEW_EXTRA[key]).iso

        NEW_EXTRA.sort("date")
        NEW_EXTRA["size"].unit = "byte"
        NEW_EXTRA["size"] = NEW_EXTRA["size"].to(u.MB)
        NEW_EXTRA["size"].info.format = "7.3f"

        if DATABASE_EXTRA is not None:
            DATABASE_EXTRA = vstack([DATABASE_EXTRA, NEW_EXTRA])
            DATABASE_EXTRA.sort("scan")
        else:
            DATABASE_EXTRA = NEW_EXTRA

        DATABASE_EXTRA.write(DB_EXTRA_FILE, overwrite=True)


def auto_update(func=None, scan=True, extra=True):
    """Decorator to auto update the scan list."""

    if func is None:
        return partial(auto_update, scan=scan, extra=extra)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if scan:
            update_scan_database()
        if extra:
            update_extra_database()
        return func(*args, **kwargs)

    return wrapper


@auto_update
def extend_database():
    """Read the header of each file and construct the parameter database."""
    global DATABASE_SCAN, DATABASE_PARAM

    from .kids_data import KidsRawData  # To avoid import loop

    data_rows = []
    param_rows = {}
    for item in ProgressBar(DATABASE_SCAN):
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
        param_set = set(
            chain(*[param.keys() for param in param_rows] + [DATABASE_PARAM.colnames if DATABASE_PARAM else []])
        )

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

        if DATABASE_PARAM is not None:
            DATABASE_PARAM = vstack([DATABASE_PARAM, NEW_PARAM])
            DATABASE_PARAM = unique(DATABASE_PARAM, "param_id")
        else:
            DATABASE_PARAM = NEW_PARAM

        DATABASE_PARAM.write(DB_PARAM_FILE)

        # Update DATABASE_SCAN
        NEW_PARAM = Table(data_rows)
        if "param_id" in DATABASE_SCAN.colnames:
            DATABASE_SCAN.add_index("filename")
            idx = DATABASE_SCAN.loc_indices[NEW_PARAM["filename"]]
            DATABASE_SCAN["param_id"][idx] = NEW_PARAM["param_id"]
        else:
            # a simple join
            DATABASE_SCAN = join(DATABASE_SCAN, NEW_PARAM, keys="filename", join_type="outer")

        DATABASE_SCAN.sort("scan")

        DATABASE_PARAM.write(DB_PARAM_FILE, overwrite=True)
        DATABASE_SCAN.write(DB_SCAN_FILE, overwrite=True)


@auto_update(extra=False)
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
    try:
        scan = int(scan)
    except ValueError as ex:
        raise ValueError("{} can not be converted to int: {}".format(scan, ex))

    mask = DATABASE_SCAN["scan"] == scan

    if not np.any(mask):
        raise IndexError("Scan {} not found".format(scan))

    return DATABASE_SCAN[mask]["filename"].data[0]


@auto_update(scan=False)
def get_extra(start=None, end=None):
    """Get filename for extra scans (skydips) between two timestamp.

    Parameters
    ----------
    start, end: datetime
        beginning and end of integration

    Returns
    -------
    filename : list
        the list of corresponding files
    """
    mask = (DATABASE_EXTRA["date"] > start) & (DATABASE_EXTRA["date"] < end)
    return DATABASE_EXTRA[mask]["filename"].data


@auto_update(extra=False)
def list_scan(output=False, **kwargs):
    """List (with filtering) all scans in the database.

    Parameters
    ----------
    output: boolean
        Return a light table for the database

    Notes
    -----
    One can filter on any key of the database
    ["filename", "date", "scan", "source", "obsmode", "size", "ctime", "mtime"]
    by giving kwargs, for example

    >>> list_scan(source="Moon")

    will display all the scan on the moon. One can also use the `__gt` and
    `__lt` postfix to the keyword to filter greather_tan and lower_than values :

    >>> list_scan(scan__gt=400)

    will return all the scan greather than 400
    """

    global DATABASE_SCAN

    if DATABASE_SCAN is None:
        raise ValueError("No scans found, check the KISS_DATA variable")

    _database = DATABASE_SCAN

    # Filtering on all possible key from table
    for key in kwargs.keys():
        if key.split("__")[0] in DATABASE_SCAN.keys():
            if "__gt" in key:
                _database = _database[_database[key.split("__")[0]] > kwargs[key]]
            elif "__lt" in key:
                _database = _database[_database[key.split("__")[0]] < kwargs[key]]
            else:
                _database = _database[_database[key] == kwargs[key]]

    if output:
        return _database[["filename", "date", "scan", "source", "obsmode", "size"]]
    else:
        _database[["date", "scan", "source", "obsmode", "size"]].pprint(max_lines=-1, max_width=-1)


@auto_update(scan=False)
def list_extra(output=False, **kwargs):
    """List (with filtering) all extra scans in the database.

    Notes
    -----
    You can filter the list see `list_scan`
    """
    if DATABASE_EXTRA is None:
        raise ValueError("No scans found, check the KISS_DATA variable")

    _database = DATABASE_EXTRA

    # Filtering on all possible key from table
    for key in kwargs.keys():
        if key.split("__")[0] in DATABASE_EXTRA.keys():
            if "__gt" in key:
                _database = _database[_database[key.split("__")[0]] > kwargs[key]]
            elif "__lt" in key:
                _database = _database[_database[key.split("__")[0]] < kwargs[key]]
            else:
                _database = _database[_database[key] == kwargs[key]]

    if output:
        return _database
    else:
        _database[["name", "date", "size"]].pprint(max_lines=-1, max_width=-1)
