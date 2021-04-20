from pathlib import Path
from autologging import logged

from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy
from .telescope_positions import MBFitsPositions


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

        if self.meta["MBFITS"] is not None and Path(self.meta["MBFITS"]).exists():
            self.telescope_positions = MBFitsPositions(self.meta["MBFITS"], "LONGOFF_LATOFF")
