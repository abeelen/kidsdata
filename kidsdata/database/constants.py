import os
import re
from pathlib import Path

BASE_DIRS = [Path(directory) for directory in os.getenv("DATA_DIR", "/data/KISS/Raw/nika2c-data3/KISS").split(";")]
CALIB_DIR = Path(os.getenv("CALIB_DIR", "/data/KISS/Calib/"))

RE_DIR = re.compile(r"X(\d*)_(\d{4,4})_(\d{2,2})_(\d{2,2})$")

# Regular scans : X20201110_0208_S1192_Crab_ITERATIVERASTER
RE_SCAN = re.compile(r"^X(\d{8,8})_(\d{4,4})_S(\d{4,4})_([\w|\+]*)_(\w*)$")

# Extra files : X_2020_10_22_12h53m20_AA_man :
RE_EXTRA = re.compile(r"^X_(\d{4,4})_(\d{2,2})_(\d{2,2})_(\d{2,2})h(\d{2,2})m(\d{2,2})_AA_man$")

# CONCERTO Table test : X14_04_Tablebt_scanStarted_10 :
RE_TABLE = re.compile(r"X(\d{2,2})_(\d{2,2})_Tablebt_scanStarted_(\d*)$")

# for kidpar files
RE_KIDPAR = re.compile(r"^e_kidpar")

DB_DIR = Path(os.getenv("DB_DIR", "."))

DB_URI = os.getenv("DB_URI", "sqlite:///" + str(DB_DIR / "kids_data.db"))

# { kd field : mapping }
# mapping being :
#     key : column name of table database
#     value : key name in file header
param_colums_key_mapping = {
    "params": {
        "nomexp": "nomexp",
        "acqfreq": "acqfreq",
        "div_kid": "div_kid"
    },
    "names": {
        "DataSc": "data_sc",
        "DataSd": "data_sd",
        "DataUc": "data_uc",
        "DataUd": "data_ud",
        "RawDataDetector": "raw_data_detector"
    }
}
