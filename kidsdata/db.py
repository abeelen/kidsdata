import os
import re
import numpy as np
import logging
from pathlib import Path
from itertools import chain
from datetime import datetime, timedelta
from functools import wraps, partial, update_wrapper

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, join, vstack, unique, MaskedColumn, Column
from astropy.utils.console import ProgressBar

from .database.constants import CALIB_DIR, BASE_DIRS, RE_INLAB, RE_DIR, RE_EXTRA, RE_KIDPAR, RE_SCAN
from .database.helpers import scan_columns, extra_columns, inlab_columns

DB_DIR = Path(os.getenv("DB_DIR", "."))
DB_SCAN_FILE = DB_DIR / ".kidsdb_scans.fits"
DB_EXTRA_FILE = DB_DIR / ".kidsdb_extra.fits"
DB_PARAM_FILE = DB_DIR / ".kidsdb_param.fits"
DB_TABLE_FILE = DB_DIR / ".kidsdb_table.fits"
DB_KIDPAR_FILE = CALIB_DIR / "kidpardb.fits"

__all__ = ["list_scan", "get_scan", "list_extra", "get_extra", "list_table", "get_filename", "get_kidpar"]

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

    n_found = np.sum(mask)

    result = merge_DB[mask]["filename"].data

    if n_found == 0:
        raise IndexError("Scan {} not found".format(filename))

    if n_found > 1:
        logging.warning("{} scans found".format(n_found))
    else:
        result = result[0]

    return result


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
        if key.split("__")[0] in _database.keys():
            if "__gt" in key:
                _database = _database[_database[key.split("__")[0]] > kwargs[key]]
            elif "__lt" in key:
                _database = _database[_database[key.split("__")[0]] < kwargs[key]]
            else:
                _database = _database[_database[key] == kwargs[key]]

    if output:
        return _database
    elif pprint_columns is not None:
        _database[pprint_columns].pprint(max_lines=-1, max_width=-1)
    else:
        _database.pprint(max_lines=-1, max_width=-1)


class KidsDB(Table):
    def __init__(
            self, *args, filename=None, dirs=None, re_pattern=None, extract_func=None, pprint_columns=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.dirs = dirs
        self.re_pattern = re_pattern
        self.extract_func = extract_func
        self.pprint_columns = pprint_columns

        if self.filename is not None and self.filename.exists():
            table_read_kwd = {"astropy_native": True, "character_as_bytes": False}
            this = Table.read(self.filename, **table_read_kwd)
            self.add_columns(this.columns)
            self._correct_time()

    def list(self, output=False, **kwargs):
        self.update()

        if self.__len__() == 0:
            raise ValueError("No scans found, check the DATA_DIR variable")

        _database = self
        # Filtering on all possible key from table
        for key in kwargs.keys():
            if key.split("__")[0] in self.keys():
                if "__gt" in key:
                    _database = self[self[key.split("__")[0]] > kwargs[key]]
                elif "__lt" in key:
                    _database = self[self[key.split("__")[0]] < kwargs[key]]
                else:
                    _database = self[self[key] == kwargs[key]]

        if output:
            return _database
        elif self.pprint_columns is not None:
            _database[self.pprint_columns].pprint(max_lines=-1, max_width=-1)
        else:
            _database.pprint(max_lines=-1, max_width=-1)

    def _correct_time(self):
        logging.debug("Converting time in DB table")
        # Ugly hack for Time columns
        for key in ["date", "mtime", "ctime"]:
            if key in self.columns:
                self[key] = Time(self[key])
                self[key].format = "iso"

        # Astropy 4.0.3 introduces a regression on this (#10824)
        try:
            key = "date" if "date" in self.colnames else "ctime"
            logging.debug("Sorting on {} in DB table".format(key))
            self.sort(key)
        except AttributeError:
            logging.error("Astropy 4.0.3 bug , table not sorted, update !")
            indexes = np.argsort(self[key])
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


class KidparDB(Table):
    """ must contains start / end  / filename columns."""

    def __init__(self, *args, filename=None, calib_dir=None, re_pattern=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.calib_dir = calib_dir
        self.re_pattern = re_pattern

        if self.filename is not None and Path(self.filename).exists():
            table_read_kwd = (
                {"astropy_native": True, "character_as_bytes": False} if self.filename.suffix == ".fits" else {}
            )
            this = Table.read(self.filename, **table_read_kwd)

            # Replace data if present
            for colname in this.colnames:
                if colname in self.colnames:
                    self.remove_column(colname)

            self.add_columns(this.columns)
            self.meta.update(this.meta)
            self._correct_time()

        self.meta["filename"] = filename
        self.meta["CALIB_DIR"] = calib_dir

    def get_kidpar(self, time):
        self.update()

        if self.__len__() == 0:
            logging.error("No items in Kidpar DB")
            return None

        after_start = self["start"] <= time
        before_end = time <= self["end"]
        within = after_start & before_end

        if np.any(within):
            # Within covered dates
            item = self["filename"][within]
            assert len(item) == 1, "Multiple kidpar found, please check Kidpar DB"
            return item[0]
        elif np.any(after_start) & np.any(before_end):
            # Within the dabatase, but outside covered dates
            logging.warning("No kidpar for this date, using the closest one in Kidpar DB")
            next_start = self["start"][~after_start][0]
            previous_end = self["end"][~before_end][-1]
            # Closest date :
            if np.abs(next_start - time) < np.abs(previous_end - time):
                item = self["filename"][self["start"] == next_start]
            else:
                item = self["filename"][self["end"] == previous_end]
            return item[0]
        elif time > self["end"][-1]:
            # After the database
            logging.warning("No kidpar for this late date, using the latest one in Kidpar DB")
            return self["filename"][-1]
        elif time < self["start"][0]:
            # Before the database
            logging.warning("No kidpar for this early date, using the earliest one in Kidpar DB")
            return self["filename"][0]
        else:
            logging.error("Something is wrong, please report")
            return None

    def _correct_time(self):
        logging.debug("Converting time in DB table")
        # Ugly hack for Time columns
        for key in ["start", "end"]:
            if key in self.columns:
                self[key] = Time(self[key])
                self[key].format = "iso"
        self.sort("start")

    def update(self):
        filenames = [file for file in Path(self.calib_dir).glob("**/*") if self.re_pattern.match(file.name)]

        data_rows = []
        for filename in filenames:
            # Removing already scanned files
            if self.__len__() != 0 and str(filename) in self.columns["filename"]:
                continue

            header = fits.getheader(filename, 1)
            if not header.get("DB-START") and not header.get("DB-END"):
                continue

            row = {
                "filename": filename.as_posix(),
                "name": filename.name,
                "start": header.get("DB-START"),
                "end": header.get("DB-END"),
            }

            data_rows.append(row)

        if len(data_rows) > 0:
            logging.info("Found {} new kidpar".format(len(data_rows)))

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

            self.write(self.filename, overwrite=True)


DB_SCAN = KidsDB(filename=DB_SCAN_FILE, re_pattern=RE_SCAN, extract_func=scan_columns, dirs=BASE_DIRS)
DB_EXTRA = KidsDB(filename=DB_EXTRA_FILE, re_pattern=RE_EXTRA, extract_func=extra_columns, dirs=BASE_DIRS)
DB_TABLE = KidsDB(filename=DB_TABLE_FILE, re_pattern=RE_INLAB, extract_func=inlab_columns, dirs=BASE_DIRS)

DB_KIDPAR = KidparDB(filename=DB_KIDPAR_FILE, re_pattern=RE_KIDPAR, calib_dir=CALIB_DIR)

table_read_kwd = {"astropy_native": True, "character_as_bytes": False}
DB_PARAM = Table.read(DB_PARAM_FILE, **table_read_kwd) if DB_PARAM_FILE.exists() else None

DBs = [DB_SCAN, DB_EXTRA, DB_TABLE]

list_scan = partial(list_data, database=DB_SCAN, pprint_columns=["date", "scan", "source", "obsmode", "size"])
list_extra = partial(list_data, database=DB_EXTRA, pprint_columns=["name", "date", "size"])
list_table = partial(list_data, database=DB_TABLE, pprint_columns=["name", "date", "scan", "size"])
get_kidpar = DB_KIDPAR.get_kidpar

# self = KidparDB(
#     [
#        {
#            "filename": f"toto_{i}",
#            "name": f"toto_{i}",
#            "start": datetime.datetime(2019, i, 1, i, i, 0, 0),
#            "end": datetime.datetime(2019, i, 18, i, 60 - i, 0, 0),
#        }
#        for i in range(1, 10)
#    ],
#    re_pattern=RE_KIDPAR,
#    calib_dir="/data/KISS/Calib",
#    filename=Path("toto.fits"),
# )
