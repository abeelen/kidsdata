import numpy as np
import warnings
from copy import deepcopy

from scipy.signal import fftconvolve, medfilt

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits, registry
from astropy.wcs import WCS
from astropy.nddata import NDDataArray, StdDevUncertainty, InverseVariance
from astropy.nddata.ccddata import _known_uncertainties
from astropy.nddata.ccddata import _unc_name_to_cls, _unc_cls_to_name, _uncertainty_unit_equivalent_to_parent


def forman(M):
    """Return Forman window.

    The Forman window is defined in (E-4) [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    numpy.bartlett, numpy.blackman, numpy.hamming, numpy.kaiser, numpy.hanning

    References
    ----------
    ..[1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
          Photometer, 2005MsT..........1S
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return (1 - ((n - M / 2) / M) ** 2) ** 2


class FTSData(NDDataArray):
    """Class to handle OPD or spectral FTS cubes.

    Parameters
    ----------
    data : `~numpy.ndarray` or `FTSData`
        The actual data contained in this `FTSData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.

    uncertainty : `~astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.

    mask : `~numpy.ndarray`-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.

    hits : `~numpy.ndarray`-like, optional
        Hit map for the data, given as a int Numpy array or any object that
        can be converted to a int Numpy array with a shape
        matching that of the data.

    flags : `~numpy.ndarray`-like or `~astropy.nddata.FlagCollection`, optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type (or an object which can be converted
        to a Numpy array) with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.

    wcs : `~astropy.wcs.WCS`, optional
        WCS-object containing the world coordinate system for the data.

    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    unit : `~astropy.units.UnitBase` instance or str, optional
        The units of the data.

    """

    __opd_idx = None
    __freq_idx = None
    hits = None

    def __init__(self, *args, hits=None, **kwargs):

        # Initialize with the parent...
        super().__init__(*args, **kwargs)

        # Additionnal data
        if hits is not None:
            self.hits = np.array(hits).astype(int)

        # Set Internal indexes on the wcs object
        if self.wcs is not None:
            opd_idx = np.argwhere("opd" == np.char.lower(self.wcs.wcs.ctype)).squeeze()
            self.__opd_idx = opd_idx.item() if opd_idx.size == 1 else None

            freq_idx = np.argwhere("freq" == np.char.lower(self.wcs.wcs.ctype)).squeeze()
            self.__freq_idx = freq_idx.item() if freq_idx.size == 1 else None

    @property
    def __is_opd(self):
        return self.__opd_idx is not None

    @property
    def __is_freq(self):
        return self.__freq_idx is not None

    @property
    def opd_axis(self):
        if self.__is_opd:
            return self.wcs.sub([self.__opd_idx + 1]).pixel_to_world(np.arange(self.shape[0]))

    @property
    def spectral_axis(self):
        if self.__is_freq:
            return self.wcs.sub([self.__freq_idx + 1]).pixel_to_world(np.arange(self.shape[0]))

    @property
    def _is_doublesided(self):
        """Return True is the cube is double sided, also enforce positive increments."""
        return (np.sum(self.wcs.sub([self.__opd_idx + 1]).all_pix2world([0, self.shape[0] - 1], 0)) == 0) & (
            self.wcs.wcs.cdelt[self.__opd_idx] > 0
        )

    @property
    def _is_onesided(self):
        """Return True is the cube is one sided, also enforce positive increments."""
        return (np.sum(self.wcs.sub([self.__opd_idx + 1]).all_pix2world(0, 0)) == 0) & (
            self.wcs.wcs.cdelt[self.__opd_idx] > 0
        )

    # from CCDData
    def _slice_wcs(self, item):
        """
        Override the WCS slicing behaviour so that the wcs attribute continues
        to be an `astropy.wcs.WCS`.
        """
        if self.wcs is None:
            return None

        try:
            return self.wcs[item]
        except Exception as err:
            self._handle_wcs_slicing_error(err, item)

    def _extract_doublesided(self):
        """Return the largest doublesided OPD cube from the data.

        Returns
        -------
        output : FTSData
            A doublesided interferograms cube
        """

        assert self.__is_opd, "Intput should be OPD cube"

        opd_wcs = self.wcs.sub([self.__opd_idx + 1])
        opds = opd_wcs.all_pix2world(np.arange(self.data.shape[0]), 0)[0]

        _maxopd = np.min([-opds.min(), opds.max()])

        signed = np.sign(opd_wcs.wcs.cdelt[0])
        slice_idx = opd_wcs.all_world2pix([-signed * _maxopd, signed * _maxopd], 0)[0].astype(int)
        slice_idx += [0, 1]  # Inclusive end
        _slice = slice(*slice_idx)

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[self.__opd_idx] -= _slice.start

        meta = deepcopy(self.meta)
        meta["HISTORY"] = "extract_doublesided"

        mask = self.mask[_slice] if self.mask is not None else None
        hits = self.hits[_slice] if self.hits is not None else None

        result = self.__class__(self.data[_slice], wcs=wcs, mask=mask, meta=meta, hits=hits)
        return result

    def _to_onesided(self):
        """Return a onesided OPD cube from the data.

        Returns
        -------
        output : FTSData
            A onesided interferograms cube
        """
        zpd_idx = self.wcs.sub([self.__opd_idx + 1]).world_to_pixel(0 * self.wcs.wcs.cunit[self.__opd_idx]).astype(int)

        extrema_opd = np.abs(self.wcs.sub([self.__opd_idx + 1]).pixel_to_world([0, self.shape[0] - 1]))

        if extrema_opd[1] >= extrema_opd[0]:
            # Positive single sided : longer right hand side...
            # Or doublesided
            extract_slice = slice(zpd_idx, None)
            os_slice = slice(0, zpd_idx + 1)
            db_slice = slice(zpd_idx, None, -1)
        elif extrema_opd[1] < extrema_opd[0]:
            # Negative single sided : longer left hand side...
            # Or double sided
            extract_slice = slice(zpd_idx, None, -1)
            os_slice = slice(0, self.data.shape[0] - zpd_idx)
            db_slice = slice(zpd_idx, None)

        # TODO: self.mask ??
        # Extract the longest part
        onesided_itg = self.data[extract_slice].copy()
        onesided_hits = self.hits[extract_slice].copy() if self.hits is not None else None

        # Take the mean with the other half on the double sided part
        onesided_itg[os_slice] += self.data[db_slice]
        onesided_itg[os_slice] /= 2

        if onesided_hits is not None:
            onesided_hits[os_slice] += self.hits[db_slice]
            onesided_hits[os_slice] /= 2

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[self.__opd_idx] = 1

        output = FTSData(onesided_itg, wcs=wcs, meta=self.meta, hits=onesided_hits)
        return output

    def __invert_doublesided(self, apodization_function=None):
        """Invert a doublesided interferograms cube.

        Parameters
        ----------
        apodization_function : func
            Apodization function to be used on the interferograms (default: None)

        Returns
        -------
        output : FTSData
            The corresponding spectral cube

        Notes
        -----
        Choice can be made among the function available in numpy at [1]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        assert self.__is_opd, "Intput should be OPD cube"
        assert self._is_doublesided, "Not a doublesided interferogram cube"

        cdelt_opd = self.wcs.wcs.cdelt[self.__opd_idx]
        cunit_opd = u.Unit(self.wcs.wcs.cunit[self.__opd_idx])
        naxis_opd = self.shape[0]
        # freq = np.fft.fftfreq(naxis_opd, d=cdelt_opd * cunit_opd) * cst.c

        if apodization_function is None:
            apodization_function = np.ones

        _cube = np.ma.array(self.data, mask=self.mask).filled(0) * np.expand_dims(
            apodization_function(naxis_opd), tuple(np.arange(1, self.ndim))
        )

        # Spencer 2005 Eq 2.29, direct fft
        spectra = np.fft.fft(np.fft.ifftshift(_cube, axes=0), axis=0)
        # Factor of 2 because we used the fourier transform
        spectra *= (4 * cdelt_opd * cunit_opd).decompose().value
        spectra = np.fft.fftshift(spectra, axes=0)
        # freq = np.fft.fftshift(freq)

        # Build new wcs
        wcs = deepcopy(self.wcs)
        wcs.wcs.ctype[self.__opd_idx] = "FREQ"
        wcs.wcs.cunit[self.__opd_idx] = "Hz"
        # TODO: (cst.c / (cdelt_opd * cunit_opd) / (naxis_opd-1)).to(u.Hz).value give the 1/2L resolution, but fails in the tests
        wcs.wcs.cdelt[self.__opd_idx] = (cst.c / (cdelt_opd * cunit_opd) / naxis_opd).to(u.Hz).value
        wcs.wcs.crpix[self.__opd_idx] = (naxis_opd - 1) / 2 + 1
        wcs.wcs.crval[self.__opd_idx] = 0

        # TODO: Estimate uncertainty/hits
        output = FTSData(spectra, meta=self.meta, wcs=wcs)
        return output

    def __invert_onesided(self, apodization_function=None):
        """Invert a onesided interferograms cube.

        Parameters
        ----------
        apodization_function : func
            Apodization function to be used on the interferograms (default: None)

        Returns
        -------
        output : FTSData
            The corresponding spectral cube

        Notes
        -----
        Choice can be made among the function available in numpy at [1]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        .. [1] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        assert self.__is_opd, "Intput should be OPD cube"
        assert self._is_onesided, "Not a one sided interferogram cube"

        cdelt_opd = self.wcs.wcs.cdelt[self.__opd_idx]
        cunit_opd = u.Unit(self.wcs.wcs.cunit[self.__opd_idx])
        naxis_opd = self.shape[0]

        if apodization_function is None:
            apodization_function = np.ones

        _cube = np.ma.array(self.data, mask=self.mask).filled(0) * np.expand_dims(
            apodization_function(2 * naxis_opd)[naxis_opd:], tuple(np.arange(1, self.ndim))
        )

        # Spencer 2005 Eq 2.29, direct fft
        # Trick is to use the unnormalized irfft
        output_shape = 2 * naxis_opd - 1
        spectra = np.fft.irfft(_cube, n=output_shape, axis=0) * output_shape
        # Factor of 2 because we used the fourier transform
        spectra *= (4 * cdelt_opd * cunit_opd).decompose().value
        spectra = np.fft.fftshift(spectra, axes=0)

        # Build new wcs
        wcs = deepcopy(self.wcs)
        wcs.wcs.ctype[self.__opd_idx] = "FREQ"
        wcs.wcs.cunit[self.__opd_idx] = "Hz"
        # (cst.c / (cdelt_opd * cunit_opd) / (output_shape-1)).to(u.Hz).value give the 1/2L resolution, but fails in the tests
        wcs.wcs.cdelt[self.__opd_idx] = (cst.c / (cdelt_opd * cunit_opd) / output_shape).to(u.Hz).value
        wcs.wcs.crpix[self.__opd_idx] = naxis_opd
        wcs.wcs.crval[self.__opd_idx] = 0

        # TODO: Estimate uncertainty/hits
        output = FTSData(spectra, meta=self.meta, wcs=wcs)

        return output

    def _get_phase_correction_function(
        self,
        niter=1,
        doublesided_apodization=None,
        medfilt_size=None,
        deg=None,
        fitting_func="polynomial",
        pcf_apodization=None,
        plot=False,
        **kwargs
    ):
        """Compute the phase correction function for the current cube

        This follow the description in [1]_ with some additionnal features.

        Parameters
        ----------
        niter : [int], optional
            number of iterations, by default 1
        doublesided_apodization : [function], optional
            apodization function for the double sided inversion, by default None, but see Notes
        medfilt_size : [int], optional
            size of the median filtering window to be applied (before polynomial fitting), by default None
        deg : [int], optional
            the polynomial degree to fit to the phase, by default None
        fitting_func : [str], ("polynomial"|"chebysev"), optional
            fitting function class, either polynomial or chebyshev, by default, "polynomial"
        pcf_apodization : [function], optional
            apodization function for the phase correction function, by default None
        plot : bool, optional
            diagnostic plots, by default False

        Returns
        -------
        array_like (cube shape)
            the phase correction function to be used as convolution kernel for the interferograms

        Notes
        -----
        Choice of apodization function can be made among the function available in numpy at [2]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.


        References
        ----------
        .. [1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
               Photometer, 2005MsT..........1S
        """
        if pcf_apodization is None:
            pcf_apodization = np.ones

        # Working copy
        itg = deepcopy(self._extract_doublesided())

        # Reference iterferogram
        itg_ma = np.ma.array(itg.data, mask=itg.mask, copy=True).filled(0)

        # Null starting phase (take only the upper part)
        phase = np.zeros(((itg.shape[0] - 1) // 2 + 1, *itg.shape[1:]))

        # Loop Here
        for i in range(niter):

            cube = itg._FTSData__invert_doublesided(apodization_function=doublesided_apodization)

            # Spencer 2.39 , well actually phases are -pi/pi so arctan2 or angle
            _phase = np.angle(cube.data[(itg.shape[0] - 1) // 2 :])

            # Replace bad phase :
            _phase[np.isnan(_phase)] = 0

            if plot:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(ncols=4)
                (freq,) = cube.wcs.sub([self.__opd_idx + 1]).all_pix2world(np.arange(cube.shape[0]), 0)
                axes[1].plot(freq, cube.data[:, :, 0])
                axes[2].plot(freq, _phase[:, :, 0])

            if medfilt_size is not None:
                # Median filtering of the phases
                _phase = medfilt(_phase, kernel_size=(medfilt_size, *(1,) * (len(itg.shape) - 1)))

            if deg is not None:

                if fitting_func == "polynomial":
                    polyfit, polyval = np.polynomial.polynomial.polyfit, np.polynomial.polynomial.polyval
                elif fitting_func == "chebychev":
                    polyfit, polyval = np.polynomial.chebyshev.chebfit, np.polynomial.chebyshev.chebval
                else:
                    raise ValueError('fitting_func should be in ("polynomial"|"chebychev")')

                # polynomial fit on the phase, weighted by the intensity
                p = []
                idx = np.linspace(0, 1, _phase.shape[0])
                # np.polynomail.polynomial.polyfit do not accept a (`M`, `K`) array for the weights, so need to loop....
                for spec, weight in zip(
                    _phase.reshape(_phase.shape[0], -1).T,
                    np.abs(cube.data[(itg.shape[0] - 1) // 2 :]).reshape(_phase.shape[0], -1).T,
                ):
                    p.append(polyfit(idx, spec, deg, w=weight))

                p = np.asarray(p).T

                # evaluate the polynomal all at once :
                _phase = polyval(idx, p).T.reshape(_phase.shape)

                # Wrap back the phases to -pi pi, uncessary, but just in case
                _phase = (_phase + np.pi) % (2 * np.pi) - np.pi
                """
fit data also incorporates smoothing in the
out of band region to ensure zero phase and derivative discontinuities and zero amplitude at
zero and Nyquist frequency.
                """

            if plot:
                axes[2].plot(freq, _phase[:, :, 0], linestyle="--")

            phase += _phase

            # Spencer 3.30
            # Using rfft leads pure real pcf and strangely could lead to wrong results
            # phase_correction_function = np.fft.irfft(np.exp(-1j * phase), axis=0, n=2*(phase.shape[0]-1)+1)
            phase_correction_function = np.fft.ifft(
                np.exp(-1j * np.fft.fftshift(np.concatenate([-phase[:0:-1], phase]), axes=0)), axis=0
            )

            # Apodization of the PCF along the first axis
            phase_correction_function = (
                np.fft.fftshift(phase_correction_function, axes=0).T
                * pcf_apodization(phase_correction_function.shape[0])
            ).T

            if plot:
                (x,) = itg.wcs.sub([3]).all_pix2world(np.arange(itg.shape[0]), 0)
                axes[3].plot(x, phase_correction_function[:, :, 0])
                axes[3].set_xlim(-1, 1)
                axes[0].plot(x, itg.data[:, :, 0])
                axes[0].set_xlim(-1, 1)

            # Correct the initial dataset with the current phase for the next iteration
            corrected_itg = fftconvolve(itg_ma, phase_correction_function, mode="same", axes=0).real
            itg.data[:] = corrected_itg

        return phase_correction_function

    def to_spectra(self, onesided_apodization=None, **kwargs):
        """Invert an interferograms cube using the (enhanced) Forman method.

        This follow the description in [1]_.

        Parameters
        ----------
        onesided_apodization : [function], optional
            apodization function to be used on the one sided interferograms, by default None
        niter : [int], optional
            number of iterations, by default 1
        doublesided_apodization : [function], optional
            apodization function for the double sided inversion, by default None, but see Notes
        medfilt_size : [int], optional
            size of the median filtering window to be applied (before polynomial fitting), by default None
        deg : [int], optional
            the polynomial degree to fit to the phase, by default None
        pcf_apodization : [function], optional
            apodization function for the phase correction function, by default None

        Returns
        -------
        output : FTSData
            The corresponding spectral cube

        Notes
        -----
        Choice of apodization function can be made among the function available in numpy at [2]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        References
        ----------
        .. [1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
               Photometer, 2005MsT..........1S
        .. [2] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """

        phase_correction_function = self._get_phase_correction_function(**kwargs)

        # Convolved the interferograms and hits
        itg = np.ma.array(self.data, mask=self.mask).filled(0)
        corrected_itg = fftconvolve(itg, phase_correction_function, mode="same", axes=0).real

        corrected_hits = None
        if self.hits is not None:
            hits = np.ma.array(self.hits, mask=self.mask).filled(0)
            corrected_hits = fftconvolve(hits, phase_correction_function, mode="same", axes=0).real

        corrected = FTSData(corrected_itg, wcs=self.wcs, hits=corrected_hits)

        onesided = corrected._to_onesided()

        return onesided.__invert_onesided(apodization_function=onesided_apodization)

    def to_hdu(
        self,
        hdu_mask="MASK",
        hdu_uncertainty="UNCERT",
        hdu_hits="HITS",
        hdu_flags=None,
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
    ):
        """Creates an HDUList object from a FTSData object.

        Parameters
        ----------
        hdu_mask, hdu_uncertainty, hdu_flags, hdu_hits : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Flags are not supported at this time. If ``None`` this attribute
            is not appended.
            Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty, ``'HITS'`` for hits and
            ``None`` for flags.

        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.

        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.

            .. versionadded:: 3.1

        Raises
        -------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        NotImplementedError
            Saving flags is not supported.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
        if isinstance(self.meta, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.meta.copy()
        else:
            header = fits.Header(self.meta)
        if self.unit is not None and self.unit is not u.dimensionless_unscaled:
            header["bunit"] = self.unit.to_string()
        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header(relax=wcs_relax)
            header.extend(wcs_header, useblanks=False, update=True)
        hdus = [fits.PrimaryHDU(self.data, header)]

        if hdu_mask and self.mask is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.mask, "shape"):
                raise ValueError("only a numpy.ndarray mask can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(self.mask.astype(np.uint8), name=hdu_mask)
            hdus.append(hduMask)

        if hdu_uncertainty and self.uncertainty is not None:
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError("only uncertainties of type {} can be saved.".format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header()
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if hasattr(self.uncertainty, "unit") and self.uncertainty.unit is not None and self.unit is not None:
                if not _uncertainty_unit_equivalent_to_parent(uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        "saving uncertainties with a unit that is not "
                        "equivalent to the unit from the data unit is not "
                        "supported."
                    )

            hduUncert = fits.ImageHDU(self.uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            hdus.append(hduUncert)

        if hdu_hits and self.hits is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.hits, "shape"):
                raise ValueError("only a numpy.ndarray hits can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduHits = fits.ImageHDU(self.hits.astype(np.uint16), name=hdu_hits)
            hdus.append(hduHits)

        if hdu_flags and self.flags:
            raise NotImplementedError("adding the flags to a HDU is not " "supported at this time.")

        hdulist = fits.HDUList(hdus)

        return hdulist

    @classmethod
    def from_array(cls, opd, data, hits=None, mask=None, **kwargs):
        """Construct FTS data from arrays.

        Parameters
        ----------
        opd : array_like or Quantity (M,)
            the optical path difference, by default 'mm'
        data : array_like (M, *)
            the corresponding data, first dimension must match opd
        hits : array_like, optionnal
            the corresponding hits
        mask : array_like, optionnal
            the corresponding mask

        Returns
        -------
        data : FTSData
            the corresponding FTSData objects
        """

        naxis = len(data.shape)
        wcs = WCS(naxis=naxis)

        if not isinstance(opd, u.Quantity):
            opd = u.Quantity(opd, "mm")

        zpd_idx = np.argmin(np.abs(opd))

        if opd[zpd_idx] != 0:
            print("Shifting opd by {} for 0".format(opd[zpd_idx]))
            opd -= opd[zpd_idx]

        dpd = np.diff(opd)
        np.testing.assert_almost_equal(
            np.median(dpd).to(dpd.unit).value, dpd.value, err_msg="Problem on opd differences"
        )

        wcs.wcs.ctype[naxis - 1] = "OPD"
        wcs.wcs.cunit[naxis - 1] = opd.unit

        wcs.wcs.crpix[naxis - 1] = zpd_idx + 1
        wcs.wcs.crval[naxis - 1] = opd[zpd_idx].value
        wcs.wcs.cdelt[naxis - 1] = np.median(dpd).value

        if mask is None:
            mask = False

        return cls(data, wcs=wcs, hits=hits, mask=mask | np.isnan(data), **kwargs)


def fits_ftsdata_writer(
    fts_data,
    filename,
    hdu_mask="MASK",
    hdu_uncertainty="UNCERT",
    hdu_hits="HITS",
    hdu_flags=None,
    key_uncertainty_type="UTYPE",
    **kwd
):
    """
    Write CCDData object to FITS file.

    Parameters
    ----------
    filename : str
        Name of file.

    hdu_mask, hdu_uncertainty, hdu_hits, hdu_flags : str or None, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty ``'HITS'`` for hits and
        ``None`` for flags.

    key_uncertainty_type : str, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.

        .. versionadded:: 3.1

    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`

    Raises
    -------
    ValueError
        - If ``self.mask`` is set but not a `numpy.ndarray`.
        - If ``self.uncertainty`` is set but not a
          `~astropy.nddata.StdDevUncertainty`.
        - If ``self.uncertainty`` is set but has another unit then
          ``self.data``.

    NotImplementedError
        Saving flags is not supported.
    """
    hdu = fts_data.to_hdu(
        hdu_mask=hdu_mask,
        hdu_uncertainty=hdu_uncertainty,
        hdu_hits=hdu_hits,
        key_uncertainty_type=key_uncertainty_type,
        hdu_flags=hdu_flags,
    )
    hdu.writeto(filename, **kwd)


with registry.delay_doc_updates(FTSData):
    #    registry.register_reader('fits', CCDData, fits_ccddata_reader)
    registry.register_writer("fits", FTSData, fits_ftsdata_writer)
    registry.register_identifier("fits", FTSData, fits.connect.is_fits)
