#!/bin/env python
# WARNING : Check that black is not srewing those lines
#SBATCH --job-name=process_iterativeraster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=50GB
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0

# Launch with :
# > sbatch template.py *args
# Or
# > sbatch --array=0-19 template.py *args
# To parallelize on 20 task (with 3 CPUs and 50 GB each)
# See help for more options

import os
import logging
import argparse
import numpy as np
import matplotlib as mpl
from pathlib import Path

# For some reason, this has to be done BEFORE importing anything from kidsdata, otherwise, it does not work...
logging.basicConfig(
    level=logging.DEBUG,  # " stream=sys.stdout,
    # format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Prevent verbose output from matplotlib
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import matplotlib.pyplot as plt

# We should be in non interactive mode...
if int(os.getenv("SLURM_ARRAY_TASK_COUNT", 0)) > 0:
    logging.info("Within sbatch...")
    mpl.use("Agg")
    # To avoid download concurrency,
    # but need to be sure that the cache is updated
    from astropy.utils.data import conf

    conf.remote_timeout = 120
    conf.download_cache_lock_attempts = 120

    # from astropy.utils import iers
    # iers.conf.auto_download = False
    # iers.conf.auto_max_age = None

    # Do not work
    # from astropy.config import set_temp_cache
    # set_temp_cache('/tmp')
else:
    plt.ion()


from autologging import logged, TRACE

from kidsdata import KissData, get_scan, list_scan

CALIB_PATH = Path("/data/KISS/Calib")


@logged
def process_scan(scan, output_dir=Path("."), **kwargs):

    try:
        kd = KissData(get_scan(scan))
        # .... Processing....
        return 0
    except (TypeError, ValueError, IndexError) as e:
        print(e)
        return 1


def main(args):

    # Slurm magyc...
    i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    n = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    if int(i) == 0:
        logging.info(f"Output directory {args.output_dir} created")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}

    db = list_scan(output=True, source=args.source, obsmode="ITERATIVERASTER")
    scans = [_["scan"] for _ in db]

    _scans = np.array_split(scans, n)[i]

    logging.info("{} will do {}".format(i, _scans))

    if _scans is None:
        return None

    for scan in _scans:
        if scan is None:
            continue
        logging.info("{} : {}".format(i, scan))
        process_scan(scan, output_dir=args.output_dir, **kwargs)


if __name__ == "__main__":

    # i = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))

    parser = argparse.ArgumentParser(description="Process some source.")
    parser.add_argument("source", type=str, action="store", help="name of the source")
    parser.add_argument("--output_dir", type=str, action="store", help="output directory (default: `source`)")
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir if args.output_dir else args.source)

    main(args)
