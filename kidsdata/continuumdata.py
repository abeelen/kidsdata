import numpy as np
from copy import deepcopy

import astropy.units as u
from astropy.io import fits, registry
from astropy.nddata import NDDataRef
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty

from astropy.nddata.ccddata import _known_uncertainties, _unc_cls_to_name, _uncertainty_unit_equivalent_to_parent


class ContinuumData(NDDataRef):
    def __init__(self, *args, **kwargs):

        if "meta" not in kwargs:
            kwargs["meta"] = kwargs.pop("header", None)
        if "header" in kwargs:
            raise ValueError("can't have both header and meta.")

        # Arbitrary unit by default
        if "unit" not in kwargs:
            kwargs["unit"] = "adu"

        # Remove hit attribute if given and pass it to the setter.
        self.hits = kwargs.pop("hits") if "hits" in kwargs else None
        super().__init__(*args, **kwargs)

    @property
    def header(self):
        return self._meta

    @header.setter
    def header(self, value):
        self.meta = value

    @property
    def hits(self):
        return self._hits

    @hits.setter
    def hits(self, value):
        self._hits = value

    @property
    def snr(self):
        if isinstance(self.uncertainty, InverseVariance):
            return self.data * np.sqrt(self.uncertainty.array)
        elif isinstance(self.uncertainty, StdDevUncertainty):
            return self.data / self.uncertainty.array
        elif isinstance(self.uncertainty, VarianceUncertainty):
            return self.data / np.sqrt(self.uncertainty.array)
        else:
            raise ValueError("Unknown uncertainty type")

    def _arithmetic(self, operation, operand, *args, **kwargs):
        # take all args and kwargs to allow arithmetic on the other properties
        # to work like before.
        # do the arithmetics on the flags (pop the relevant kwargs, if any!!!)
        if self.hits is not None and operand.hits is not None:
            result_hits = operation(self.hits, operand.hits)
            # np.logical_or is just a suggestion you can do what you want
        else:
            if self.hits is not None:
                result_hits = deepcopy(self.hits)
            else:
                result_hits = deepcopy(operand.hits)

        # Let the superclass do all the other attributes note that
        # this returns the result and a dictionary containing other attributes
        result, kwargs = super()._arithmetic(operation, operand, *args, **kwargs)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the processed flags
        kwargs["hits"] = result_hits
        return result, kwargs  # these must be returned

    def _slice(self, item):
        # slice all normal attributes
        kwargs = super()._slice(item)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the sliced flags
        kwargs["hits"] = self.hits[item]
        return kwargs  # these must be returned

    # from astropy.nddata.ccddata
    def _insert_in_metadata_fits_safe(self, key, value):
        """
        Insert key/value pair into metadata in a way that FITS can serialize.

        Parameters
        ----------
        key : str
            Key to be inserted in dictionary.

        value : str or None
            Value to be inserted.

        Notes
        -----
        This addresses a shortcoming of the FITS standard. There are length
        restrictions on both the ``key`` (8 characters) and ``value`` (72
        characters) in the FITS standard. There is a convention for handling
        long keywords and a convention for handling long values, but the
        two conventions cannot be used at the same time.

        This addresses that case by checking the length of the ``key`` and
        ``value`` and, if necessary, shortening the key.
        """

        if len(key) > 8 and len(value) > 72:
            short_name = key[:8]
            self.meta["HIERARCH {}".format(key.upper())] = (short_name, f"Shortened name for {key}")
            self.meta[short_name] = value
        else:
            self.meta[key] = value

    def to_hdu(
        self,
        hdu_data="DATA",
        hdu_mask="MASK",
        hdu_uncertainty="UNCERT",
        hdu_hits="HITS",
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
    ):
        """Creates an HDUList object from a ContinuumData object.
        Parameters
        ----------
        hdu_data, hdu_mask, hdu_uncertainty, hdu_hits : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Default is ``'DATA'`` for data, ``'MASK'`` for mask, ``'UNCERT'``
            for uncertainty and ``HITS`` for hits.
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

        Raises
        -------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
        if isinstance(self.header, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.header.copy()
        else:
            # Because _insert_in_metadata_fits_safe is written as a method
            # we need to create a dummy CCDData instance to hold the FITS
            # header we are constructing. This probably indicates that
            # _insert_in_metadata_fits_safe should be rewritten in a more
            # sensible way...
            dummy_data = ContinuumData([1], meta=fits.Header(), unit="")
            for k, v in self.header.items():
                dummy_data._insert_in_metadata_fits_safe(k, str(v))
            header = dummy_data.header
        if self.unit is not u.dimensionless_unscaled:
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

        hdus = [fits.ImageHDU(self.data, header, name=hdu_data)]

        if hdu_mask and self.mask is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.mask, "shape"):
                raise ValueError("only a numpy.ndarray mask can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(self.mask.astype(np.uint8), header, name=hdu_mask)
            hdus.append(hduMask)

        if hdu_uncertainty and self.uncertainty is not None:
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError("only uncertainties of type {} can be saved.".format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header(header)
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if hasattr(self.uncertainty, "unit") and self.uncertainty.unit is not None:
                if not _uncertainty_unit_equivalent_to_parent(uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        "saving uncertainties with a unit that is not "
                        "equivalent to the unit from the data unit is not "
                        "supported."
                    )

            hduUncert = fits.ImageHDU(self.uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            hdus.append(hduUncert)

        if hdu_hits and self.hits is not None:
            # Always assuming that the hits is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.hits, "shape"):
                raise ValueError("only a numpy.ndarray hits can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduHits = fits.ImageHDU(self.hits.astype(np.uint8), header, name=hdu_hits)
            hdus.append(hduHits)

        hdulist = fits.HDUList(hdus)

        return hdulist


def fits_continuumdata_reader(
    filename,
    hdu=0,
    unit=None,
    hdu_uncertainty="UNCERT",
    hdu_mask="MASK",
    hdu_flags=None,
    key_uncertainty_type="UTYPE",
    **kwd,
):
    raise NotImplementedError("Reading the continuum data is not supported at this time.")


def fits_continuumdata_writer(
    c_data, filename, hdu_mask="MASK", hdu_uncertainty="UNCERT", hdu_hits="HITS", key_uncertainty_type="UTYPE", **kwd
):
    """
    Write ContinuumData object to FITS file.
    Parameters
    ----------
    filename : str
        Name of file.
    hdu_mask, hdu_uncertainty, hdu_hits : str or None, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty and
        ``HITS`` for flags.
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
    hdu = [fits.PrimaryHDU(None, header=fits.Header(c_data.header))]
    hdu += c_data.to_hdu(
        hdu_mask=hdu_mask, hdu_uncertainty=hdu_uncertainty, key_uncertainty_type=key_uncertainty_type, hdu_hits=hdu_hits
    )
    hdu = fits.HDUList(hdu)

    hdu.writeto(filename, **kwd)


with registry.delay_doc_updates(ContinuumData):
    registry.register_reader("fits", ContinuumData, fits_continuumdata_reader)
    registry.register_writer("fits", ContinuumData, fits_continuumdata_writer)
    registry.register_identifier("fits", ContinuumData, fits.connect.is_fits)
