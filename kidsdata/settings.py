from pathlib import Path
from decouple import AutoConfig

config = AutoConfig(".")

# only used in default session of database.api
DB_URI = config("DB_URI", default="sqlite:///" + str(Path(".").resolve() / "kids_data.db"))
BASE_DIRS = [Path(directory) for directory in config("DATA_DIR", default="/data/CONCERTO/InLab").split(";")]

# other settings
CALIB_DIR = config("CALIB_DIR", default="/data/CONCERTO/Calib/", cast=Path)
CACHE_DIR = config("CACHE_DIR", default="/data/CONCERTO/Cache", cast=Path)

# read_kidsdata settings
READDATA_LIB_PATH = config("READDATA_LIB_PATH", default="/data/CONCERTO/Processing/kid-all-sw/Readdata/C", cast=Path)
READRAW_LIB_PATH = config(
    "READRAW_LIB_PATH", default="/data/CONCERTO/Processing/kid-all-sw/Acquisition/kani/readRaw", cast=Path
)

KISS_CAT_DIR = config(
    "KISS_CAT_DIR",
    default="/data/KISS/NIKA_lib_AB_OB_gui/Acquisition/instrument/kiss_telescope/library/KISS_Source_Position/",
    cast=Path,
)
