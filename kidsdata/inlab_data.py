import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from autologging import logged

from scipy.signal import savgol_filter

from astropy.stats import mad_std

from .kiss_continuum import KissContinuum
from .kiss_spectroscopy import KissSpectroscopy
from .kiss_rawdata import KissRawData
from .utils import cpu_count, mad_med
from .utils import _import_from


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


def _to_kidfreq(ikids):

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

    def read_data(self, *args, **kwargs):

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

    # Fix table positions
    def _fix_table(self, delta_pix=540000, speed_sigma_clipping=5, min_speed=1, plot=False, savgol_args=(11, 3)):

        ## Andrea Catalano Priv. Comm 20200113 : 30 arcsec == 4.5 mm
        # Scan X15_16_Tablebt_scanStarted_12 with Lxy = 90/120, we have (masked) delta_x = 180000.0, masked_delta_y = 240000.0,
        # so we have micron and Lxy is to be understood has half the full map
        # ((4.5*u.mm) / (30*u.arcsec)).to(u.micron / u.deg) = <Quantity 540000. micron / deg>
        # So with delta_pix = 540000 with should have proper degree in _tabdiff_Az and _tabdiff_El

        tab_keys = [
            key
            for key in self.names.DataSc + self.names.DataSd + self.names.DataUc + self.names.DataUd
            if "tabx" in key or "taby" in key
        ]
        tab_prefix = set([key.split("_")[0] for key in tab_keys])
        assert len(tab_prefix) == 1
        tab_prefix = tab_prefix.pop()

        tabx = getattr(self, "{}_tabx".format(tab_prefix)).flatten()
        taby = getattr(self, "{}_taby".format(tab_prefix)).flatten()
        mask = np.zeros_like(tabx, dtype=bool)

        # Mask anomalous speed :
        if speed_sigma_clipping is not None:

            savgol_kwargs = {"deriv": 1}
            speed = np.sqrt(
                savgol_filter(tabx, *savgol_args, **savgol_kwargs) ** 2
                + savgol_filter(taby, *savgol_args, **savgol_kwargs) ** 2
            )
            # speed = np.sqrt(np.diff(tabx, append=0)**2 + np.diff(taby, append=0)**2)
            # Flag small speed which sometimes could represent most of the data
            mask |= speed < min_speed
            med_speed, mad_speed = mad_med(speed[~mask])
            if mad_speed > 0:
                mask |= np.abs(speed - med_speed) > speed_sigma_clipping * mad_speed

            from scipy.ndimage.morphology import binary_closing

            # Remove singletons, and doublons
            mask = binary_closing(mask, iterations=2)

        if mad_std(tabx) == 0 and mad_std(taby) == 0:
            self.__log.info("Non moving planet")
            mask = np.zeros_like(tabx, dtype=bool)
            delta_pix = 1
        elif delta_pix is None:
            # Normalization of the tab position between -0.5, 0.5
            delta_x = tabx[~mask].max() - tabx[~mask].min()
            delta_y = taby[~mask].max() - taby[~mask].min()
            delta_pix = np.max([delta_x, delta_y])

        tabdiff_Az = (tabx - tabx[~mask].mean()) / delta_pix
        tabdiff_El = (taby - taby[~mask].mean()) / delta_pix

        self._tabdiff_Az = tabdiff_Az
        self._tabdiff_El = tabdiff_El
        self._KidsRawData__position_keys["tabdiff"] = ("_tabdiff_Az", "_tabdiff_El")

        self.mask_tel = mask

        self.__log.warning("Masking {:3.0f} % of the data from speed".format(np.mean(mask) * 100))

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(tabdiff_Az, tabdiff_El)
            ax.scatter(tabdiff_Az[~mask], tabdiff_El[~mask])

        # fix_table & shift  before!!
        if (
            hasattr(self, "continuum")
            and self.continuum is not None
            and self.mask_tel.shape[0] != self.continuum.shape[1]
        ):
            self._tabdiff_Az = (
                np.ma.array(self._tabdiff_Az, mask=self.mask_tel).reshape(-1, self.nptint).mean(axis=1).data
            )
            self._tabdiff_El = (
                np.ma.array(self._tabdiff_El, mask=self.mask_tel).reshape(-1, self.nptint).mean(axis=1).data
            )
            self.mask_tel = (
                self.mask_tel.reshape(-1, self.nptint).mean(axis=1) > 0.8
            )  # Allow for 80% flagged positional data

        return delta_pix

    def _from_phase(self):
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

            # As well as telescope positions :cube = self.interferograms_cube(ikid=[20], coord='tabdiff', flatfield=None, cm_func=None, baseline=2, cdelt=(0.02, 0.25), weights=None, bins=150)

            # fix_table & shift  before!!
            if self.mask_tel.shape[0] != self.continuum.shape[1]:
                self._tabdiff_Az = (
                    np.ma.array(self._tabdiff_Az, mask=self.mask_tel).reshape(-1, self.nptint).mean(axis=1).data
                )
                self._tabdiff_El = (
                    np.ma.array(self._tabdiff_El, mask=self.mask_tel).reshape(-1, self.nptint).mean(axis=1).data
                )
                self.mask_tel = (
                    self.mask_tel.reshape(-1, self.nptint).mean(axis=1) > 0.8
                )  # Allow for 80% flagged positional data

            # kidfreq is fully sampled (copy is made here) remove first order continuum
            with Pool(
                N_CPU,
                initializer=_pool_initializer,
                initargs=(self.ph_IQ,),
            ) as pool:
                kidfreq = pool.map(_to_kidfreq, np.array_split(np.arange(self.list_detector.shape[0]), N_CPU))

            self.kidfreq = np.vstack(kidfreq)

            self.A_masq = np.zeros(self.ph_IQ.shape[1:])

    def _change_nptint(self, nptint):

        if self.nptint % nptint == 0 or nptint % self.nptint == 0:
            self.__log.error("Not a multiple of the original nptint")
            return None

        # Scans are not well recorderd nptint is 1024, should be 512:
        for key in self.__dict__.keys():
            item = getattr(self, key)

            if not hasattr(item, "shape"):
                continue

            if item.shape == (self.ndet, self.nint, self.nptint):
                setattr(self, key, item.reshape(self.ndet, -1, nptint))
            elif item.shape == (self.nint, self.nptint):
                setattr(self, key, item.reshape(-1, nptint))
        self.nint = self.nint * self.nptint // nptint
        self.nptint = nptint
