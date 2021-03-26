import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from autologging import logged

from scipy.signal import savgol_filter
from scipy.ndimage.morphology import binary_closing

from astropy.stats import mad_std
from astropy.time import Time

from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy
from .kiss_rawdata import KissRawData
from .kids_rawdata import KidsRawData
from .utils import cpu_count, mad_med
from .utils import _import_from
from .telescope_positions import InLabPositions


N_CPU = cpu_count()

_pool_global = None


def _pool_initializer(*args):
    global _pool_global
    _pool_global = args


def _downsample_to_continuum(ikids, nint=-1, nptint=1024):

    global _pool_global
    (ph_IQ,) = _pool_global

    continuum = np.nanmedian(ph_IQ[ikids].reshape(ikids.shape[0], nint, nptint), axis=2)
    continuum = continuum - np.nanmedian(continuum, axis=1)[:, None]
    continuum = ph_IQ.dtype.type(continuum)

    return continuum


def _to_continuum(ikids):

    global _pool_global
    (ph_IQ,) = _pool_global

    continuum = ph_IQ[ikids].reshape(ikids.shape[0], -1)
    continuum = continuum - np.nanmedian(continuum, axis=1)[:, None]
    continuum = ph_IQ.dtype.type(continuum)

    return continuum


def _to_interferograms(ikids):

    global _pool_global
    (ph_IQ,) = _pool_global

    return ph_IQ.dtype.type(ph_IQ[ikids] - np.nanmedian(ph_IQ[ikids], axis=-1)[:, :, None])


def _ph_unwrap(ikids):

    global _pool_global
    (ph_IQ,) = _pool_global
    return ph_IQ.dtype.type(
        np.unwrap(ph_IQ[ikids].reshape(ikids.shape[0], -1) / 1000).reshape(ikids.shape[0], *ph_IQ.shape[1:]) * 1000
    )


@logged
class InLabData(KissContinuum, KissSpectroscopy):
    """Class dealing with CONCERTO Lab data.

    Methods
    -------
    _fix_table(**kwargs)
        to transform table positions into angles and flags
    _from_phase()
        derive continuum and kidsfreq from phase only data
    _change_nptint(nptint)
        reshape the current data with new nptint
    """

    def read_data(self, *args, delta_pix=540000, **kwargs):

        super().read_data(*args, **kwargs)

        if "ph_IQ" in self.__dict__:
            self.__log.info("Unwrap phIQ")
            with Pool(
                N_CPU,
                initializer=_pool_initializer,
                initargs=(self.ph_IQ,),
            ) as pool:
                ph_IQ = pool.map(_ph_unwrap, np.array_split(np.arange(self.list_detector.shape[0]), N_CPU))

            self.ph_IQ = np.vstack(ph_IQ)

        self.meta["delta_pix"] = delta_pix

        keys = self.__dict__.keys()
        pos_keys = [
            (key, key.replace("tabx", "taby"))
            for key in [key for key in keys if key.endswith("tabx")]
            if key in keys and key.replace("tabx", "taby") in keys
        ]

        if len(pos_keys) == 1:
            self.__log.debug("Initializing InLabPositions")
            lon, lat = pos_keys[0]

            pos = np.array([getattr(self, lon).flatten(), getattr(self, lat).flatten()])
            mjd = getattr(self, "obstime").flatten()

            if pos.shape[1] == self.nint * self.nptint:
                pass
            elif pos.shape[1] == self.nint:
                # Undersamped position, assuming center of block
                self.__log.error('Under sampled position with "tabdiff" should not occur')
                mjd = Time(mjd.mjd.reshape(self.nint, self.nptint).mean(1), scale=mjd.scale, format="mjd")
            else:
                raise ValueError("Do not known how to handle position tabx|y")

            args = (mjd, pos)
            # Do not copy data (reference)
            # pos_keys = {
            #     "tab": InLabPositions(*args, position_key="tab"),
            #     "tabdiff": InLabPositions(*args, position_key="tabdiff"),
            # }
            self.telescope_positions = InLabPositions(*args, delta_pix=self.meta["delta_pix"])

    def _from_phase(self, clean_raw=False):
        # Non moving mirror -> keep everything oversampled as continuum :
        if mad_std(self.laser.mean(0)) < 1:
            self.__log.info("Non moving laser, keeping fully sampled data")
            with Pool(
                N_CPU,
                initializer=_pool_initializer,
                initargs=(self.ph_IQ,),
            ) as pool:
                continuum = pool.map(_to_continuum, np.array_split(np.arange(self.list_detector.shape[0]), N_CPU))

            self.continuum = np.vstack(continuum)

        else:
            self.__log.info("Spectroscopic data, downsampling continuum")

            _this = partial(_downsample_to_continuum, nint=self.nint, nptint=self.nptint)
            with Pool(
                N_CPU,
                initializer=_pool_initializer,
                initargs=(self.ph_IQ,),
            ) as pool:
                continuum = pool.map(_this, np.array_split(np.arange(self.list_detector.shape[0]), N_CPU))

            self.continuum = np.vstack(continuum)

            # interferograms is fully sampled (copy is made here) remove first order continuum
            with Pool(
                N_CPU,
                initializer=_pool_initializer,
                initargs=(self.ph_IQ,),
            ) as pool:
                interferograms = pool.map(
                    _to_interferograms, np.array_split(np.arange(self.list_detector.shape[0]), N_CPU)
                )

            self.interferograms = np.vstack(interferograms)

            # TODO: do the same on all the mask !!
            self.A_masq = np.zeros(self.ph_IQ.shape[1:])

        if clean_raw:
            self._clean_data("_KidsRawData__dataSd")

    def _change_nptint(self, nptint):

        if self.nptint % nptint != 0 and nptint % self.nptint != 0:
            self.__log.error("Not a multiple of the original nptint")
            return None

        nptint_ratio = nptint / self.nptint
        nint = int(self.nint / nptint_ratio)

        nint_max = int(nint * nptint_ratio)
        if self.nint % nint_max != 0:
            self.__log.warning("{} blocs truncated".format(self.nint % nint_max))

        for key in self.__dict__.keys():
            item = getattr(self, key)

            if not hasattr(item, "shape"):
                continue

            if item.shape == (self.ndet, self.nint, self.nptint):
                setattr(self, key, item[:, 0:nint_max, :].reshape(self.ndet, -1, nptint))
            elif item.shape == (self.nint, self.nptint):
                setattr(self, key, item[0:nint_max].reshape(-1, nptint))
            elif item.shape == (self.nint,) or item.shape == (self.ndet, self.nint):
                self.__log.warning("{} need special care".format(key))

        self.nint = self.nint * self.nptint // nptint
        self.nptint = nptint
        self.__log.info("Clearing Cache")
        KissSpectroscopy.opds.cache_clear()
        KissSpectroscopy.laser.fget.cache_clear()
        KissSpectroscopy.laser_directions.fget.cache_clear()
        KissRawData.mod_mask.fget.cache_clear()
        KissRawData.fmod.fget.cache_clear()
        KissRawData.get_object_altaz.cache_clear()
        KissRawData._pdiff_Az.fget.cache_clear()
        KissRawData._pdiff_El.fget.cache_clear()
        KidsRawData.get_telescope_positions.cache_clear()
        self.__log.info("You probably need to run _fix_table()")
