from pathlib import Path
from autologging import logged

import astropy.units as u
from astropy.time import Time

from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy
from .telescope_positions import MBFitsPositions
from .kiss_rawdata import KissRawData


@logged
class ConcertoData(KissContinuum, KissSpectroscopy):
    """Class to deal with OnSky CONCERO data"""

    def __init__(self, *args, mbfits_file=None, **kwargs):
        """
        Parameters
        ----------
        mbfits_file : str
            the path to the MBFits file containing the pointing of this scan
        """
        super().__init__(*args, **kwargs)
        self.meta["MBFITS"] = mbfits_file

    def read_data(self, *args, **kwargs):

        super().read_data(*args, **kwargs)

        # ObsTime are not in UTC (the default) but in GPS  !!
        obstime = Time((self.obstime + 19 * u.s).mjd, format="mjd", scale="tai")

        self._KidsRawData__dataSc["obstime"] = obstime
        self.obstime = obstime
        KissRawData.u_obstime.fget.cache_clear()

        if self.meta["MBFITS"] is not None and Path(self.meta["MBFITS"]).exists():
            self.telescope_positions = MBFitsPositions(self.meta["MBFITS"], "LONGOFF_LATOFF")
