from pathlib import Path
from decouple import config

# only used in default session of database.api
DB_DIR = config("DB_DIR",  default=".", cast=Path)
DB_URI = config("DB_URI", default="sqlite:///" + str(DB_DIR / "kids_data.db"))
BASE_DIRS = [Path(directory) for directory in config("DATA_DIR", default="/data/KISS/Raw/nika2c-data3/KISS").split(";")]

# other settings
CALIB_DIR = config("CALIB_DIR", default="/data/KISS/Calib/", cast=Path)
NIKA_LIB_PATH = config("NIKA_LIB_PATH", default="/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/", cast=Path)
CACHE_DIR = config("CACHE_DIR", default="/data/KISS/Cache", cast=Path)
