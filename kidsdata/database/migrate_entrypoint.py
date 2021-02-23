import os
import sys
from pathlib import Path

import alembic.config


def call_alembic():
    curdir = os.curdir
    os.chdir(Path(__file__).parent)
    alembicArgs = [
        '--raiseerr',
        *sys.argv[1:],
    ]
    alembic.config.main(argv=alembicArgs)
    os.chdir(curdir)
