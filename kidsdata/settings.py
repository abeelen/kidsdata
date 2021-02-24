from pathlib import Path
from decouple import config

# only used in default session of database.api
DB_DIR = config("DB_DIR", default=".", cast=Path)
DB_URI = config("DB_URI", default="sqlite:///" + str(DB_DIR.resolve() / "kids_data.db"))
BASE_DIRS = [Path(directory) for directory in config("DATA_DIR", default="/data/CONCERTO/InLab").split(";")]

# other settings
CALIB_DIR = config("CALIB_DIR", default="/data/CONCERTO/Calib/", cast=Path)
CACHE_DIR = config("CACHE_DIR", default="/data/CONCERTO/Cache", cast=Path)

# read_kidsdata settings
NIKA_LIB_PATH = config("NIKA_LIB_PATH", default="/data/CONCERTO/Processing/kid-all-sw/Readdata/C", cast=Path)
READRAW_LIB_PATH = config(
    "READRAW_LIB_PATH", default="/data/CONCERTO/Processing/kid-all-sw/Acquisition/kani/readRaw", cast=Path
)
