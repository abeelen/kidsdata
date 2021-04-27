from pathlib import Path
from autologging import logged
import numpy as np

import astropy.units as u
from astropy.time import Time

from .kiss_rawdata import KissRawData
from .kids_rawdata import KidsRawData
from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy
from .telescope_positions import MBFitsPositions
from .settings import MBFITS_DIR, ESO_PROJID


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

        if mbfits_file == "auto":
            kwargs = {
                "scan": self.meta["SCAN"],
                "date": Time(self.meta["OBSDATE"], format="iso", out_subfmt="date").value,
                "projid": ESO_PROJID,
            }
            mbfits_file = MBFITS_DIR / "APEX-{scan}-{date}-{projid}".format(**kwargs)

        self.meta["MBFITS"] = mbfits_file

        if self.meta["MBFITS"] is not None and Path(self.meta["MBFITS"]).exists():
            self.telescope_positions = MBFitsPositions(self.meta["MBFITS"], "OFFSET_AZ_EL")
            self.meta.update(self.telescope_positions.header_to_meta())

    def read_data(self, *args, **kwargs):

        super().read_data(*args, **kwargs)

        # ObsTime are not in UTC (the default) but in GPS  !!
        obstime = Time((self.obstime + 19 * u.s).mjd, format="mjd", scale="tai")

        self._KidsRawData__dataSc["obstime"] = obstime
        self.obstime = obstime
        KissRawData.u_obstime.fget.cache_clear()

    def _lower_nptint(self, nptint, reduce=np.mean):

        assert self.nptint / nptint == self.nptint // nptint

        self.interferograms = (self.continuum[:, :, None] + self.interferograms).reshape(self.ndet, -1, nptint)
        self.continuum = reduce(self.interferograms, axis=-1)
        self.interferograms -= self.continuum[:, :, None]

        for key in self.__dict__.keys():

            if key in ["continuum", "interferograms"]:
                continue

            item = getattr(self, key)

            if not hasattr(item, "shape"):
                continue

            if item.shape == (self.ndet, self.nint, self.nptint):
                setattr(self, key, item.reshape(self.ndet, -1, nptint))
            elif item.shape == (self.nint, self.nptint):
                setattr(self, key, item.reshape(-1, nptint))
            elif item.shape == (self.nint,) or item.shape == (self.ndet, self.nint):
                self.__log.warning("{} need special care".format(key))

        self.nint = self.nint * self.nptint // nptint
        self.nptint = nptint
        KissRawData.u_obstime.fget.cache_clear()
        KissRawData._get_positions.cache_clear()
        KissRawData.mod_mask.fget.cache_clear()
        KidsRawData._get_positions.cache_clear()
        KidsRawData.get_telescope_positions.cache_clear()
