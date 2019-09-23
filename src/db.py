import logging
logging.basicConfig(level=logging.INFO)

from pathlib import Path
from itertools import chain
from src.kids_data import KissRawData
from datetime import datetime
from astropy.table import Table, join # for now
from astropy.utils.console import ProgressBar

BASE_DIRS = [Path('/data/KISS/Raw/nika2c-data3/KISS'), ]
DATABASE_SCAN = None
DATABASE_PARAM = None


def fill_database(dirs=None):

    global DATABASE_SCAN

    if dirs is None:
        dirs = BASE_DIRS

    filenames = chain(*[path.glob('X*_*_S????_*_*') for path in dirs])

    data_rows = []
    for filename in filenames:
        date, hour, scan, source, obsmode = filename.name[1:].split('_')
        dtime = datetime.strptime(' '.join([date, hour]), '%Y%m%d %H%M')
        scan =  int(scan[1:])
        data_rows.append((filename.as_posix(), dtime, scan, source, obsmode))

    DATABASE_SCAN = Table(names=['filename', 'date', 'scan', 'source', 'obsmode'], rows=data_rows)
    DATABASE_SCAN.sort('date')

def auto_fill(func):
    def wrapper(*args, **kwargs):
        if DATABASE_SCAN is None:
            fill_database()
        return func(*args, **kwargs)
    return wrapper

@auto_fill
def extend_database():

    global DATABASE_SCAN, DATABASE_PARAM
    data_rows = []
    param_rows = {}
    for filename in ProgressBar(DATABASE_SCAN['filename']):
        kd = KissRawData(filename)
        hash_param =  hash(str(kd.param_c))
        data_row = {'filename': filename, 'param_id': hash_param}
        param_row = {'param_id': hash_param}
        param_row.update(kd.param_c)
        data_rows.append(data_row)
        param_rows[hash_param] = param_row
        del(kd)

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


@auto_fill
def get_scan(scan=None):

    mask = DATABASE_SCAN['scan'] == scan
    return DATABASE_SCAN[mask]['filename'].data[0]

@auto_fill
def list_scan(**kwargs):

    _database = DATABASE_SCAN

    # Filtering on all possible key from table
    for key in kwargs.keys():
        if key in DATABASE_SCAN.keys():
            if '_gt' in key:
                _database = _database[_database[key.split('_')[0]] > kwargs[key]]
            elif '_lt' in key:
                _database = _database[_database[key.split('_')[0]] < kwargs[key]]
            else:
                _database = _database[_database[key] == kwargs[key]]

    print(_database[['date', 'scan', 'source', 'obsmode']])
