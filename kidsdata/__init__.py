from ._version import get_versions

from .rta import *
from .kiss_data import KissData
from .inlab_data import InLabData
from .ftsdata import FTSData

__version__ = get_versions()["version"]
del get_versions


# Back compatibility issue, will disapear
def KissRawData(*args, **kwargs):
    from warnings import warn

    warn("Deprecated name, please use KissData")
    return KissData(*args, **kwargs)
