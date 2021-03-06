import json
import os
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

import logging

import numpy as np
from astropy.io import fits

from sqlalchemy import create_engine, union_all, case
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, scoped_session, sessionmaker, aliased
from sqlalchemy.orm.exc import NoResultFound

from kidsdata.database.constants import RE_DIR, param_colums_key_mapping
from kidsdata.database.models import Param, Kidpar, Scan

logger = logging.getLogger(__name__)


def one(session, model, **filters):
    return session.query(model).filter_by(**filters).one()


def get_or_create(session, model, create_method="", create_method_kwargs=None, obj=None, **kwargs):
    """
    Simply get an object if already present in the database or create it in the
    other case. See
    http://skien.cc/blog/2014/01/15/sqlalchemy-and-race-conditions-implementing/
    and
    http://skien.cc/blog/2014/02/06/sqlalchemy-and-race-conditions-follow-up/
    for better details on why this function as been upgraded to the provided
    example. Better handling of weird cases in the situation of multiple
    processes using the database at the same time.
    """
    try:
        return one(session, model, **kwargs)
    except NoResultFound:
        kwargs.update(create_method_kwargs or {})
        if obj is None:
            created = getattr(model, create_method, model)(**kwargs)
        else:
            created = obj
        try:
            session.add(created)
            session.commit()
            return created
        except IntegrityError:
            session.rollback()
            return one(session, model, **kwargs)


def create_row(file_path, re_pattern, extract_func=None):
    stat = file_path.stat()
    row = {
        "file_path": file_path.as_posix(),
        "name": file_path.name,
        "size": stat.st_size,
        "ctime": datetime.fromtimestamp(stat.st_ctime),
        "mtime": datetime.fromtimestamp(stat.st_mtime),
        # "comment": " " * 128
    }

    if extract_func is not None:
        row.update(extract_func(file_path, re_pattern))

    return row


def populate_func(session, dirs, re_pattern=None, extract_func=None, Model=None):
    # One query for all
    paths = [file_path for file_path, in session.query(Model.file_path)]

    # Filter here
    filenames = []
    for path in dirs:
        for file_path in Path(path).glob("**/*"):
            if re_pattern.match(file_path.name) and str(file_path) not in paths:
                filenames.append(file_path)
                logger.debug(f"File matching found : {file_path}")

    logger.info(f"Adding {len(filenames)} files to database")
    for file_path in filenames:
        row = create_row(file_path, re_pattern, extract_func)
        session.add(Model(**row))

    session.commit()


def astro_columns(file_path, re_pattern=None):
    """Extract func for Astro table"""
    date, hour, scan, source, obsmode = re_pattern.match(file_path.name).groups()
    dtime = datetime.strptime(" ".join([date, hour]), "%Y%m%d %H%M")
    if scan is not None and scan != "":
        scan = int(scan)
    else:
        scan = -1
    return {
        "date": dtime,
        "scan": scan,
        "source": source,
        "obsmode": obsmode,
    }


def manual_columns(file_path, re_pattern=None):
    """Extract func for Manual table"""
    time_data = [int(item) for item in re_pattern.match(file_path.name).groups()]
    return {"date": datetime(*time_data)}


def tablebt_columns(file_path, re_pattern=None):
    hour, minute, scan = re_pattern.match(file_path.name).groups()
    _, year, month, day = RE_DIR.match(file_path.parent.name).groups()
    dtime = datetime.strptime(" ".join([year, month, day, hour, minute]), "%Y %m %d %H %M")
    return {"date": dtime, "scan": scan}


def create_param_row(session, path):
    """Read a scan file header and create a row in the Param table

    - creates a row in the table Param if not exists already
    - update the foreign key on the row of Scan/Manual/Tablebt table

    IMPORTANT: this function adds rows to the session but does not commit the
    changes to database because it is made for parallelization
    """
    logger.debug(f"populating params for scan {path}")
    from ..kids_rawdata import KidsRawData  # To avoid import loop

    row = session.query(Scan).filter_by(file_path=path).first()

    file_path = row.file_path
    kd = KidsRawData(file_path)
    params = kd.param_c
    params = {
        param_name: int(param_val) if isinstance(param_val, np.int32) else param_val
        for param_name, param_val in params.items()
    }
    parameters = json.dumps(params)
    param_hash = sha256(bytes(parameters, encoding="utf-8")).hexdigest()
    param_ids = session.query(Param.id).filter_by(param_hash=param_hash).all()
    if not param_ids:
        names = kd.names
        dict_values_params = {
            db_name: params.get(key_name, None) for key_name, db_name in param_colums_key_mapping["params"].items()
        }

        # inserting json dumps of DataSc/Sd etc fields as strings
        dict_values_names = {
            db_name: json.dumps(getattr(names, key_name, None))
            for key_name, db_name in param_colums_key_mapping["names"].items()
        }
        new_param_row = Param(**dict_values_names, **dict_values_params, param_hash=param_hash, parameters=parameters)
        param_row = get_or_create(session, Param, obj=new_param_row, param_hash=param_hash)
    else:
        param_row = param_ids[0]

    row.param_id = param_row.id
    del kd


def create_kidpar_row(session, file_path):
    """Creates a Kidpar object (sqlalchmy model object) from a file and returns it

    Raises
    ------
    AttributeError :
    """

    header = fits.getheader(file_path, 1)
    db_start = header.get("DB-START")
    db_end = header.get("DB-END")
    if not db_start and not db_end:
        raise AttributeError("Could not find fields DB-START or DB-END")

    row = {
        "file_path": file_path.as_posix(),
        "name": file_path.name,
        "start": header.get("DB-START"),
        "end": header.get("DB-END"),
    }

    return Kidpar(**row)


# https://stackoverflow.com/questions/42552696/sqlalchemy-nearest-datetime
def get_closest(session, cls, col, the_time):
    greater = session.query(cls).filter(col > the_time).order_by(col.asc()).limit(1).subquery().select()

    lesser = session.query(cls).filter(col <= the_time).order_by(col.desc()).limit(1).subquery().select()

    the_union = union_all(lesser, greater).alias()
    the_alias = aliased(cls, the_union)
    the_diff = getattr(the_alias, col.name) - the_time
    abs_diff = case([(the_diff < timedelta(0), -the_diff)], else_=the_diff)

    return session.query(the_alias).order_by(abs_diff.asc()).first()
