import logging
from typing import List

from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from kidsdata.database.helpers import get_session, create_param_rows
from kidsdata.database.models import Scan, Extra, Table

logger = logging.getLogger(__name__)


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

def populate_params(session = None):
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
                create_param_rows(session, row.file_path)
            except Exception as e:
                logger.error(f"Updating params for file {row.file_path} failed for the following reason : ")
                logger.exception(e)

            session.commit()
