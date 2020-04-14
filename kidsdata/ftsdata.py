import numpy as np
from copy import deepcopy

from scipy.signal import fftconvolve

import astropy.units as u
import astropy.constants as cst
from astropy.io import fits, registry
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
    """Class to handle OPD or spectral cubes."""

    def __init__(self, *args, hits=None, **kwargs):
        # Initialize with the parent...
        super().__init__(*args, **kwargs)
        self.hits = hits

    @property
    def _is_opd(self):
        return self.wcs.sub([3]).wcs.ctype[0] == "opd"

    @property
    def _is_doublesided(self):
        """Test if the cube is doublesided, enforce positive increments."""
        return (np.sum(self.wcs.sub([3]).all_pix2world([0, self.shape[0] - 1], 0)) == 0) & (self.wcs.wcs.cdelt[2] > 0)

    @property
    def _is_onesided(self):
        """Test if the cube is onesided, enforce positive increments."""
        return (np.sum(self.wcs.sub([3]).all_pix2world(0, 0)) == 0) & (self.wcs.wcs.cdelt[2] > 0)

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

        assert self._is_opd, "Intput should be OPD cube"

        opd_wcs = self.wcs.sub([3])
        opds = opd_wcs.all_pix2world(np.arange(self.data.shape[0]), 0)[0]

        _maxopd = np.min([-opds.min(), opds.max()])

        signed = np.sign(opd_wcs.wcs.cdelt[0])
        slice_idx = opd_wcs.all_world2pix([-signed * _maxopd, signed * _maxopd], 0)[0].astype(int)
        slice_idx += [0, 1]  # Inclusive end
        _slice = slice(*slice_idx)

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[2] -= _slice.start

        meta = deepcopy(self.meta)
        meta["HISTORY"] = "extract_doublesided"

        result = self.__class__(self.data[_slice], wcs=wcs, mask=self.mask[_slice], meta=meta, hits=self.hits[_slice])
        return result

    def _to_onesided(self):
        """Return a onesided OPD cube from the data.

        Returns
        -------
        output : FTSData
            A onesided interferograms cube
        """
        zpd_idx = self.wcs.sub([3]).world_to_pixel(0 * self.wcs.wcs.cunit[2]).astype(int)

        # Extract the positive part
        onesided_itg = self.data[zpd_idx:].copy()
        onesided_hits = self.hits[zpd_idx:].copy()

        # Take the mean with the other half of the double sided part, this assume positive onesided
        onesided_itg[: zpd_idx + 1] += self.data[zpd_idx::-1]
        onesided_itg[: zpd_idx + 1] /= 2

        onesided_hits[: zpd_idx + 1] += self.hits[zpd_idx::-1]
        onesided_hits[: zpd_idx + 1] /= 2

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[2] = 1

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
        assert self._is_opd, "Intput should be OPD cube"
        assert self._is_doublesided, "Not a doublesided interferogram cube"

        cdelt_opd = self.wcs.wcs.cdelt[2]
        cunit_opd = u.Unit(self.wcs.wcs.cunit[2])
        naxis_opd = self.shape[0]
        # freq = np.fft.fftfreq(naxis_opd, d=cdelt_opd * cunit_opd) * cst.c

        if apodization_function is None:
            apodization_function = np.ones

        _cube = (
            np.ma.array(self.data, mask=self.mask).filled(0)
            * apodization_function(naxis_opd)[:, np.newaxis, np.newaxis]
        )

        # Spencer 2005 Eq 2.29, direct fft
        # TODO: Check normalization here....
        spectra = (
            np.fft.fft(np.fft.ifftshift(_cube, axes=0), axis=0) * (2 * cdelt_opd * cunit_opd / cst.c).to(1 / u.Hz).value
        )

        spectra = np.fft.fftshift(spectra, axes=0)
        # freq = np.fft.fftshift(freq)

        # Build new wcs
        wcs = deepcopy(self.wcs)
        wcs.wcs.ctype[2] = "FREQ"
        wcs.wcs.cunit[2] = "Hz"
        wcs.wcs.cdelt[2] = (cst.c / (cdelt_opd * cunit_opd) / naxis_opd).to(u.Hz).value
        wcs.wcs.crpix[2] = (naxis_opd - 1) / 2 + 1
        wcs.wcs.crval[2] = 0

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
        assert self._is_opd, "Intput should be OPD cube"
        assert self._is_onesided, "Not a one sided interferogram cube"

        cdelt_opd = self.wcs.wcs.cdelt[2]
        cunit_opd = u.Unit(self.wcs.wcs.cunit[2])
        naxis_opd = self.shape[0]

        if apodization_function is None:
            apodization_function = np.ones

        _cube = (
            np.ma.array(self.data, mask=self.mask).filled(0)
            * apodization_function(2 * naxis_opd)[naxis_opd:, np.newaxis, np.newaxis]
        )

        # Spencer 2005 Eq 2.29, direct fft
        # Trick is to use the unnormalized irfft
        output_shape = 2 * naxis_opd - 1
        spectra = (
            np.fft.irfft(_cube, n=output_shape, axis=0)
            * output_shape
            * (2 * cdelt_opd * cunit_opd / cst.c).to(1 / u.Hz).value
        )
        spectra = np.fft.fftshift(spectra, axes=0)

        # Build new wcs
        wcs = deepcopy(self.wcs)
        wcs.wcs.ctype[2] = "FREQ"
        wcs.wcs.cunit[2] = "Hz"
        wcs.wcs.cdelt[2] = (cst.c / (cdelt_opd * cunit_opd) / output_shape).to(u.Hz).value
        wcs.wcs.crpix[2] = naxis_opd
        wcs.wcs.crval[2] = 0

        # TODO: Estimate uncertainty/hits
        output = FTSData(spectra, meta=self.meta, wcs=wcs)

        return output

    def to_spectra(self, deg=None, pcf_apodization=None, doublesided_apodization=None, onesided_apodization=None):
        """Invert an interferograms cube using the (enhanced) Forman method.

        This follow the description in [1]_.

        Parameters
        ----------
        deg : int
            Degree of the fitting polynomial to the phase (default: None)
        pcf_apodization : func
            Apodization function to be used on the Phase Correction Function (default: None)
        doublesided_apodization : func
            Apodization function to be used on the double sided interferograms (default: None)
        onesided_apodization : func
            Apodization function to be used on the one sided interferograms (default: None)

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
        if pcf_apodization is None:
            pcf_apodization = np.ones

        doublesided = self._extract_doublesided()
        doublesided_cube = doublesided.__invert_doublesided(apodization_function=doublesided_apodization)

        # Spencer 2.39 !
        phase = np.fft.ifftshift(
            np.arctan(doublesided_cube.data.imag / doublesided_cube.data.real), axes=0
        )  # -pi/2 pi/

        # TODO: Enhance Format fit this phase with low-order polynomial

        # Spencer 3.30
        phase_correction_function = np.fft.ifft(np.exp(-1j * phase), axis=0)
        phase_correction_function *= np.fft.ifftshift(pcf_apodization(phase.shape[0]))[:, np.newaxis, np.newaxis]

        # Turn it into a convolution kernel
        phase_correction_function = np.fft.fftshift(phase_correction_function, axes=0)

        # Convolved the interferograms and hits
        itg = np.ma.array(self.data, mask=self.mask).filled(0)
        hits = np.ma.array(self.hits, mask=self.mask).filled(0)

        corrected_itg = fftconvolve(itg, phase_correction_function, mode="same", axes=0).real
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
