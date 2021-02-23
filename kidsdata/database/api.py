import logging
from datetime import datetime
from functools import partial
from typing import List

from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, Session

from kidsdata.database.constants import RE_KIDPAR, RE_ASTRO, RE_MANUAL, RE_TABLEBT
from kidsdata.settings import CALIB_DIR, DB_URI, BASE_DIRS

from kidsdata.database.helpers import (
    create_param_row,
    create_kidpar_row,
    get_closest,
    populate_func,
    astro_columns,
    manual_columns,
    tablebt_columns,
)
from kidsdata.database.models import Astro, Manual, Tablebt, Kidpar, Scan

logger = logging.getLogger(__name__)

global_session = None

__all__ = ["get_session", "get_scan", "get_filename", "get_manual", "get_file_path", "populate_params",
           "populate_kidpar", "get_kidpar", "populate_scans", "list_data", "list_manual", "list_astro", "list_tablebt",
           "edit_comment"]


# TODO def list_scan():


def get_session() -> Session:
    """Initialize a module level session

    This is useful when using kidsdata outside an outflow pipeline
    """
    global global_session
    if global_session is None:

        if DB_URI is None:
            raise ValueError('Cannot initialize session, environment variable "DB_URI" not found')
        else:
            global_session = scoped_session(sessionmaker((create_engine(DB_URI))))

    return global_session


def get_scan(scan, session=None) -> str:
    """Get file path of corresponding scan number.

    Parameters
    ----------
    scan : the scan number to retrieve
    session : optional, sqlalchemy session to query

    Returns
    -------
    file_path : the full path of the file
    """

    # update à chaque fois

    if session is None:
        session = get_session()

    populate_scans(session)

    file_paths = [file_path for file_path, in session.query(Astro.file_path).filter_by(scan=scan)]

    # renvoyer 1 file path si il y en a qu'un

    if len(file_paths) == 0:
        raise IndexError(f"Scan {scan} not found")

    if len(file_paths) > 1:
        logger.warning(f"Found more than one scan with scan number {scan}")

    return file_paths[0]


def get_filename(filename, session=None) -> str:
    """Get file path of corresponding filename

    Parameters
    ----------
    filename : file name of the scan to get
    session : optional, sqlalchemy session to query

    Returns
    -------
    file_path : the full path of the file
    """

    # update à chaque fois

    if session is None:
        session = get_session()

    populate_scans(session)

    file_paths = [file_path for file_path, in session.query(Scan.file_path).filter_by(name=filename)]

    if len(file_paths) == 0:
        raise IndexError(f"Scan {filename} not found")

    if len(file_paths) > 1:
        logger.warning(f"Found more than one scan with filename = {filename}")

    return file_paths[0]


def get_manual(start, end, session=None) -> List[str]:  # TODO
    """Get path for manual scans (typically skydips or dome observation) between two timestamp.

    Parameters
    ----------
    session : optional
        sqlalchemy session

    start, end: datetime
        beginning and end of integration

    Returns
    -------
    file_paths : list
        the list of paths of the corresponding files
    """

    if session is None:
        session = get_session()

    populate_scans(session)

    file_paths = [
        file_path
        for file_path, in session.query(Manual.file_path).filter(start < Manual.date).filter(Manual.date < end).all()
    ]

    return file_paths


def get_file_path(filename: str, session=None) -> List[str]:
    """Get path base on a filename

    Parameters
    ----------
    filename : the filename of the observation
    session : optional, sqlalchemy session to query

    Returns
    -------
    filename : list of paths with this file name
    """
    if session is None:
        session = get_session()

    populate_scans(session)

    rows = []
    for Model in [Astro, Manual, Tablebt]:
        rows.extend([file_path for file_path, in session.query(Model.file_path).filter_by(name=filename).all()])

    return rows


def populate_params(session=None):
    """Read the header of each file and construct the parameter database."""

    if session is None:
        session = get_session()

    logging.getLogger("kidsdata").setLevel("ERROR")

    rows = []

    for Model in [Astro, Manual, Tablebt]:
        rows.extend(session.query(Model).filter_by(param_id=None).all())

    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "Scan {task.completed}/{task.total}",
            TimeRemainingColumn(),
            refresh_per_second=1,
    ) as progress:

        task1 = progress.add_task("Update table params...", total=len(rows))

        for row in rows:
            # logger.info(filename)
            progress.update(task1, advance=1)
            progress.refresh()
            try:
                create_param_row(session, row.file_path)
            except Exception as e:
                logger.error(f"Updating params for file {row.file_path} failed for the following reason : ")
                logger.exception(e)

            session.commit()


def populate_kidpar(session=None):
    if session is None:
        session = get_session()

    file_paths = [file for file in CALIB_DIR.glob("**/*") if RE_KIDPAR.match(file.name)]

    logger.debug(f"Inserting {len(file_paths)} kidpars into database")

    for file_path in file_paths:
        logger.debug(f"{file_path}")
        try:
            session.add(create_kidpar_row(session, file_path))
        except AttributeError:
            logger.error(f"Cannot insert kidpar {file_path} into database")

    session.commit()


def get_kidpar(time: datetime, session=None) -> str:
    """Returns the file path of the kidpar valid for the given time"""

    if session is None:
        session = get_session()

    # update db before querying
    populate_kidpar(session)

    kidpars = session.query(Kidpar).filter(Kidpar.start < time).filter(Kidpar.end > time).all()

    if kidpars:
        if len(kidpars) > 1:
            logger.warning("Multiple kidpar found, please check table kidpar integrity. Returning one of them.")
        return kidpars[0].file_path
    else:

        first_kidpar_time = session.query(Kidpar).order_by(Kidpar.start).first()
        last_kidpar_time = session.query(Kidpar).order_by(Kidpar.end.desc()).first()

        if time < first_kidpar_time.start:
            logger.warning("No kidpar for this early date, using the earliest one in Kidpar table")
            return first_kidpar_time.file_path

        if time > last_kidpar_time.end:
            logger.warning("No kidpar for this late date, using the latest one in Kidpar table")
            return last_kidpar_time

        closest_start = get_closest(session, Kidpar, Kidpar.start, time)
        closest_end = get_closest(session, Kidpar, Kidpar.end, time)

        if abs(time - closest_start.start) < abs(time - closest_end.end):
            logger.warning("No kidpar for this date, using the closest one in Kidpar table")
            return closest_start.file_path
        else:
            logger.warning("No kidpar for this date, using the closest one in Kidpar table")
            return closest_end.file_path


def populate_scans(session=None):
    if session is None:
        session = get_session()

    dirs = BASE_DIRS

    populate_func(session, dirs, RE_ASTRO, astro_columns, Astro)
    populate_func(session, dirs, RE_MANUAL, manual_columns, Manual)
    populate_func(session, dirs, RE_TABLEBT, tablebt_columns, Tablebt)


def list_data(model=Scan, pprint_columns=None, output=False, session=None, **kwargs):
    """List (with filtering) all data in the database.

    Notes
    -----
    You can filter the list see `list_data`
    """
    if session is None:
        session = get_session()

    populate_scans(session)

    scans_query = session.query(model)

    # Filtering on all possible key from table
    for key in kwargs.keys():
        try:
            if "__gt" in key:
                scans_query = scans_query.filter(getattr(model, key.split("__")[0]) > kwargs[key])
            elif "__lt" in key:
                scans_query = scans_query.filter(getattr(model, key.split("__")[0]) < kwargs[key])
            else:
                scans_query = scans_query.filter(getattr(model, key.split("__")[0]) == kwargs[key])
        except AttributeError:
            logger.warning(f"Cannot filter on '{key}', row '{key}' does not exist in table {model.__tablename__}")

    scans = scans_query.all()

    if output:
        return scans

    from rich.table import Table

    table = Table(title=f"Filtered content of table {model.__tablename__}")
    if pprint_columns is None:
        # print all columns (exclude private
        pprint_columns = [col_name for col_name in scans[0].__dict__ if not col_name[0] == "_"]

    logger.info(pprint_columns)

    for col in pprint_columns:
        table.add_column(col)

    for scan in scans:
        table.add_row(*[str(getattr(scan, col)) for col in pprint_columns])

    Console().print(table)


list_astro = partial(list_data, model=Astro, pprint_columns=["date", "scan", "source", "obsmode", "size"])
list_manual = partial(list_data, model=Manual, pprint_columns=["name", "date", "size"])
list_tablebt = partial(list_data, model=Tablebt, pprint_columns=["name", "date", "scan", "size"])


def edit_comment(scan=None, filename=None, comment="", session=None):
    if session is None:
        session = get_session()

    populate_scans(session)
    filter = None

    if scan is not None:
        filter = {"scan", scan}

    if filename is not None:
        filter = {"name": filename}

    if filter is None:
        raise KeyError("Please call with either a scan number or a filename")

    scan = session.query(Scan).filter_by(**filter).one()

    scan.comment = comment

    session.commit()
