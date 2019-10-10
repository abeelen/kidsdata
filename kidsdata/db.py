import os
from pathlib import Path
from itertools import chain
from datetime import datetime
from functools import wraps

import astropy.units as u
from astropy.table import Table, join, vstack  # for now
from astropy.utils.console import ProgressBar

from .kids_data import KidsRawData

BASE_DIRS = [Path(os.getenv("KISS_DATA", "/data/KISS/Raw/nika2c-data3/KISS"))]
DATABASE_SCAN = None
DATABASE_EXTRA = None
DATABASE_PARAM = None

__all__ = ["list_scan", "get_scan", "list_extra", "get_extra"]


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

    # Regular scan files
    filenames = chain(*[Path(path).glob("**/X*_*_S????_*_*") for path in dirs])

    data_rows = []
    for filename in filenames:
        if DATABASE_SCAN is not None and filename not in DATABASE_SCAN["filename"]:
            continue
        date, hour, scan, source, obsmode = filename.name[1:].split("_")
        dtime = datetime.strptime(" ".join([date, hour]), "%Y%m%d %H%M")
        scan = int(scan[1:])
        stat = filename.stat()
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

        NEW_SCAN = Table(
            names=[
                "filename",
                "date",
                "scan",
                "source",
                "obsmode",
                "size",
                "ctime",
                "mtime",
            ],
            rows=data_rows,
        )
        NEW_SCAN.sort("date")
        NEW_SCAN["size"].unit = "byte"
        NEW_SCAN["size"] = NEW_SCAN["size"].to(u.MiB)
        NEW_SCAN["size"].info.format = "7.3f"

        if DATABASE_SCAN is not None:
            DATABASE_SCAN = vstack(DATABASE_SCAN, NEW_SCAN)
        else:
            DATABASE_SCAN = NEW_SCAN


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
        if DATABASE_EXTRA is not None and filename not in DATABASE_EXTRA["filename"]:
            continue
        year, month, day, hour = filename.name[2:].split("_")[0:4]
        dtime = datetime.strptime(
            " ".join([year, month, day, hour]), "%Y %m %d %Hh%Mm%S"
        )
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
        NEW_EXTRA = Table(
            names=["filename", "name", "date", "size", "ctime", "mtime"], rows=data_rows
        )
        NEW_EXTRA.sort("date")
        NEW_EXTRA["size"].unit = "byte"
        NEW_EXTRA["size"] = NEW_EXTRA["size"].to(u.MiB)
        NEW_EXTRA["size"].info.format = "7.3f"

        if DATABASE_EXTRA is not None:
            DATABASE_EXTRA = vstack(DATABASE_EXTRA, NEW_EXTRA)
        else:
            DATABASE_EXTRA = NEW_EXTRA


def auto_update(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DATABASE_SCAN is None:
            update_scan_database()
            update_extra_database()
        return func(*args, **kwargs)

    return wrapper


@auto_update
def extend_database():
    """Read the header of each file and construct the parameter database."""
    global DATABASE_SCAN, DATABASE_PARAM

    data_rows = []
    param_rows = {}
    for filename in ProgressBar(DATABASE_SCAN["filename"]):
        kd = KidsRawData(filename)
        hash_param = hash(str(kd.param_c))
        data_row = {"filename": filename, "param_id": hash_param}
        param_row = {"param_id": hash_param}
        param_row.update(kd.param_c)
        data_rows.append(data_row)
        param_rows[hash_param] = param_row
        del kd

    param_rows = [*param_rows.values()]

    # Get unique parameter list
    param_set = set(chain(*[param.keys() for param in param_rows]))
    # Fill missing value
    for param in param_rows:
        for key in param_set:
            if key not in param:
                param[key] = None

    DATABASE_PARAM = Table(param_rows)
    DATABASE_SCAN = join(DATABASE_SCAN, Table(data_rows))


@auto_update
def get_scan(scan=None):
    """Get filename of corresponding scan number.

    Parameters
    ----------
    scan : int
        the scan number to retrieve

    Returns
    -------
    filename : str
       the full path of the file
    """
    mask = DATABASE_SCAN["scan"] == scan
    return DATABASE_SCAN[mask]["filename"].data[0]


@auto_update
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


@auto_update
def list_scan(**kwargs):
    """List (with filtering) all scans in the database.

    Notes
    -----
    One can filter on any key of the database by giving kwargs, for example

    >>> list_scan(source="Moon")

    will display all the scan on the moon. One can also use the `__gt` and
    `__lt` postfix to the keyword to filter greather_tan and lower_than values :

    >>> list_scan(scan__gt=400)

    will return all the scan greather than 400
    """

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

    print(_database[["date", "scan", "source", "obsmode", "size"]])


@auto_update
def list_extra(**kwargs):
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

    print(_database[["name", "date", "size"]])
