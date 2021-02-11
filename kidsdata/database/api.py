import logging
import os
from datetime import datetime
from typing import List

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, Session

from kidsdata.database.constants import CALIB_DIR, RE_KIDPAR, DB_URI
from kidsdata.database.helpers import create_param_row, create_kidpar_row, get_closest
from kidsdata.database.models import Scan, Extra, Table, Kidpar

logger = logging.getLogger(__name__)

global_session = None


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


def get_scan(scan, session=None) -> List[str]:  # TODO
    """Get filename of corresponding scan number.

    Parameters
    ----------
    scan : the scan number to retrieve
    session : optional, sqlalchemy session to query

    Returns
    -------
    file_path : the full path of the file
    """
    if session is None:
        session = get_session()

    file_paths = [file_path for file_path, in session.query(Scan.file_path).filter_by(scan=scan)]

    return file_paths


def get_extra(start, end, session=None) -> List[str]:  # TODO
    """Get path for extra scans (typically skydips or dome observation) between two timestamp.

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

    file_paths = [
        file_path
        for file_path, in session.query(Extra.file_path).filter(start < Extra.date).filter(Extra.date < end).all()
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
    rows = []
    for Model in [Scan, Extra, Table]:
        rows.extend([file_path for file_path, in session.query(Model.file_path).filter_by(name=filename).all()])

    return rows


def populate_params(session=None):
    """Read the header of each file and construct the parameter database."""

    if session is None:
        session = get_session()

    logging.getLogger("kidsdata").setLevel("ERROR")

    rows = []

    for Model in [Scan, Extra, Table]:
        rows.extend(session.query(Model).filter_by(param_id=None).all())

    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "Scan {task.completed}/{task.total}",
            TimeRemainingColumn(),
            refresh_per_second=1
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
        except AttributeError as ae:
            logger.error(f"Cannot insert kidpar {file_path} into database")

    session.commit()


def get_kidpar(time: datetime, session=None) -> str:
    """Returns the file path of the kidpar valid for the given time"""

    if session is None:
        session = get_session()

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
