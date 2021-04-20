from ._version import get_versions


from .database import *
from .rta import *

from .kiss_data import KissData
from .ftsdata import FTSData
from .inlab_data import InLabData
from .concerto_data import ConcertoData

__version__ = get_versions()["version"]
del get_versions


# Back compatibility issue, will disapear
def KissRawData(*args, **kwargs):
    from warnings import warn

    warn("Deprecated name, please use KissData")
    return KissData(*args, **kwargs)
