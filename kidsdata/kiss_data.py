from .kiss_rawdata import KissRawData
from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy


# pylint: disable=no-member
class KissData(KissSpectroscopy, KissContinuum):
    pass
