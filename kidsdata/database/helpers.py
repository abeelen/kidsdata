from datetime import datetime
from pathlib import Path

import logging

from rich.progress import track
from sqlalchemy.orm import Session

from kidsdata.database.constants import RE_DIR
from kidsdata.database.models import Scan, Extra, Inlab

logger = logging.getLogger(__name__)


def stat_filename(filename):
    stat = filename.stat()
    return {
        "size": stat.st_size,
        "ctime": datetime.fromtimestamp(stat.st_ctime),
        "mtime": datetime.fromtimestamp(stat.st_mtime),
    }


def create_row(filename, re_pattern, extract_func=None):

    stat = filename.stat()
    row = {
        "filename": filename.as_posix(),
        "name": filename.name,
        "size": stat.st_size,
        "ctime": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        # "comment": " " * 128
    }

    row.update(stat_filename(filename))

    if extract_func is not None:
        row.update(extract_func(filename, re_pattern))

    return row


def populate_func(session, dirs, re_pattern=None, extract_func=None, Model=None):

    # One query for all
    _DB = [filename for filename, in session.query(Model.filename)]

    # Filter here
    filenames = []
    for path in dirs:
        for filename in Path(path).glob("**/*"):
            if re_pattern.match(filename.name) and str(filename) not in _DB:
                filenames.append(filename)
                logger.info(f"File matching found : {filename}")

    logger.info(f"Adding {len(filenames)} files to database")
    for filename in filenames:
        row = create_row(filename, re_pattern, extract_func)
        session.add(Model(**row))

    session.commit()



def scan_columns(filename, re_pattern=None):
    date, hour, scan, source, obsmode = re_pattern.match(filename.name).groups()
    dtime = datetime.strptime(" ".join([date, hour]), "%Y%m%d %H%M")
    scan = int(scan)
    return {
        "date": dtime,
        "scan": scan,
        "source": source,
        "obsmode": obsmode,
    }


def extra_columns(filename, re_pattern=None):
    time_data = [int(item) for item in re_pattern.match(filename.name).groups()]
    return {"date": datetime(*time_data)}


def inlab_columns(filename, re_pattern=None):
    hour, minute, scan = re_pattern.match(filename.name).groups()
    _, year, month, day = RE_DIR.match(filename.parent.name).groups()
    dtime = datetime.strptime(" ".join([year, month, day, hour, minute]), "%Y %m %d %H %M")
    return {"date": dtime, "scan": scan}


def populate_kidpar_func(session: Session):
    """Read the header of each file and construct the parameter database."""
    ## WIP
    from ..kids_rawdata import KidsRawData  # To avoid import loop

    for Model in [Scan, Extra, Inlab]:
        rows = session.query(Model).all()

    data_rows = []
    param_rows = {}
    for row in track(rows, description="Creating params rows..."):
        if row["param_id"] is not None:
            # Skip if present
            continue
        filename = row["filename"]
        try:
            kd = KidsRawData(filename)
            hash_param = hash(str(kd.param_c))
            data_row = {"filename": filename, "param_id": hash_param}
            param_row = {"param_id": hash_param}
            param_row.update(kd.param_c)
            data_rows.append(data_row)
            param_rows[hash_param] = param_row
            del kd
        except AssertionError:
            logging.warning("{} failed".format(filename))

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
            NEW_PARAM[key] = MaskedColumn(NEW_PARAM[key].data.astype(_dtype), mask=mask)

        if DB_PARAM is not None:
            DB_PARAM = vstack([DB_PARAM, NEW_PARAM])
            DB_PARAM = unique(DB_PARAM, "param_id")
        else:
            DB_PARAM = NEW_PARAM

        # Update db
        NEW_PARAM = Table(data_rows)
        if "param_id" not in rows.colnames:
            rows.add_column(Column(0, name="param_id", dtype=np.int64))

        rows.add_index("filename")
        idx = rows.loc_indices[NEW_PARAM["filename"]]
        rows["param_id"][idx] = NEW_PARAM["param_id"]

        rows._correct_time()

        DB_PARAM.write(DB_PARAM_FILE, overwrite=True)
        rows.write(rows.filename, overwrite=True)


