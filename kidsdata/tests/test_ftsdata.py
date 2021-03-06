import pytest

from functools import partial

import numpy as np
import numpy.testing as npt

import astropy.units as u
import astropy.constants as cst
from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty

from kidsdata.ftsdata import FTSData


def empty_ds(n_itg=1024, n_pix=1):
    """Double sided"""
    data = np.zeros((n_itg + 1, n_pix, 1))
    hits = np.ones_like(data)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["OLON-SFL", "OLAT-SFL", "OPD"]
    wcs.wcs.crpix[2] = n_itg // 2 + 1
    wcs.wcs.cunit[2] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


@pytest.fixture(name="empty_ds")
def empty_ds_fixture():
    return empty_ds(n_itg=1024, n_pix=1)


def empty_ds_2d(n_itg=1024, n_pix=1):
    """Double sided"""
    data = np.zeros((n_itg + 1, n_pix))
    hits = np.ones_like(data)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["PIX", "OPD"]
    wcs.wcs.crpix[1] = n_itg // 2 + 1
    wcs.wcs.cunit[1] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


@pytest.fixture(name="empty_ds_2d")
def empty_ds_2d_ficture():
    return empty_ds_2d(n_itg=1024, n_pix=1)


def empty_ds_1d(n_itg=1024):
    """Double sided"""
    data = np.zeros((n_itg + 1))
    hits = np.ones_like(data)

    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ["OPD"]
    wcs.wcs.crpix[0] = n_itg // 2 + 1
    wcs.wcs.cunit[0] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


@pytest.fixture(name="empty_ds_1d")
def empty_ds_1d_ficture():
    return empty_ds_1d(n_itg=1024)


def empty_os(n_itg=1024, n_pix=1):
    """One sided"""
    data = np.zeros((n_itg, n_pix, 1))
    hits = np.ones_like(data)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["OLON-SFL", "OLAT-SFL", "OPD"]
    wcs.wcs.crpix[2] = 1  # Actual first pixel
    wcs.wcs.cunit[2] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


@pytest.fixture(name="empty_os")
def empty_os_fixture():
    return empty_os(n_itg=1024, n_pix=1)


def empty_ss(n_itg=1024, n_pix=1):
    """Single sided"""
    # The longest part will be n_itg
    data = np.zeros((n_itg + n_itg // 4 + 1, n_pix, 1))
    hits = np.ones_like(data)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["OLON-SFL", "OLAT-SFL", "OPD"]
    wcs.wcs.crpix[2] = n_itg // 2 + 1  # One forth of the dataset
    wcs.wcs.cunit[2] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


def empty_ss_low(n_itg=1024, n_pix=1):
    """Single sided Low"""
    # The longest part will be n_itg
    data = np.zeros((n_itg + n_itg // 4 + 1, n_pix, 1))
    hits = np.ones_like(data)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["OLON-SFL", "OLAT-SFL", "OPD"]
    wcs.wcs.crpix[2] = n_itg * 3 // 4 + 1  # One forth of the dataset
    wcs.wcs.cunit[2] = "mm"

    return FTSData(data, wcs=wcs, hits=hits)


@pytest.fixture(name="empty_ss")
def empty_ss_fixture():
    return empty_ss(n_itg=1024, n_pix=1)


def rect(x, center, sigma):
    output = (np.abs(x - center) <= sigma / 2).astype(np.float)
    return output


def rect_itg(x, center_x, sigma_x):
    shift = np.cos(2 * np.pi * u.Quantity(center_x * x).decompose().value)
    rect_ft = np.sinc(u.Quantity(sigma_x * x).decompose().value) * np.abs(sigma_x)
    # Factor 2 comes from the MPI and fourier approximation
    return shift * rect_ft / 2


def gaussian(x, center, sigma):
    output = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return output.decompose()


def gaussian_itg(x, center_x, sigma_x):
    shift = np.cos(2 * np.pi * u.Quantity(center_x * x).decompose().value)
    gaussian_ft = (
        np.exp(-(2 * np.pi ** 2 * u.Quantity(sigma_x * x).decompose().value ** 2)) * np.sqrt(2 * np.pi) * sigma_x
    )
    # Factor 2 comes from the MPI and fourier approximation
    return shift * gaussian_ft / 2


def ils(freq, opd_max):
    r"""Compute the normalized ILS.

    Parameters
    ----------
    freq : astropy.units.Quantity
        the frequency axis
    opd_max : astropy.units.Quantity
        the maximum optical path difference

    Returns
    -------
    array_like
        the ILS to convolve the corresponding spectra

    Notes
    -----
    .. math:: FT( rect(a x) / a ) = sinc\left(\frac{\xi]{a}\right)

    with :math:`a = \frac{1}{2 opd_max}` and :math:`\sinc(x) = \frac{\sin(\pi x)}{\pi x}`
    """
    # Normalized to one rectangle function
    wn = freq / cst.c
    return np.sinc(u.Quantity(wn * 2 * opd_max).decompose().value)


def fixture_itg(
    central_freq=120 * u.GHz,
    sigma_freq=10 * u.GHz,
    n_itg=2048,
    n_pix=1,
    shifts=None,
    cdelt_opd=0.5 * u.mm,
    func_itg=gaussian_itg,
    func=gaussian,
    sided="double",
    plot=False,
):

    if sided == "double":
        data = empty_ds(n_itg=n_itg, n_pix=n_pix)
    elif sided == "double_2d":
        data = empty_ds_2d(n_itg=n_itg, n_pix=n_pix)
    elif sided == "double_1d":
        data = empty_ds_1d(n_itg=n_itg)
    elif sided == "one":
        data = empty_os(n_itg=n_itg, n_pix=n_pix)
    elif sided == "single":
        data = empty_ss(n_itg=n_itg, n_pix=n_pix)
    elif sided == "single_low":
        data = empty_ss_low(n_itg=n_itg, n_pix=n_pix)

    data.wcs.wcs.cdelt[data._opd_idx] = cdelt_opd.to(data.wcs.wcs.cunit[data._opd_idx]).value

    # temporary fix :
    # data.wcs.wcs.ctype[2] = "opd"

    opd_wcs = data.wcs.sub([data._opd_idx + 1])
    cdelt_opd = opd_wcs.wcs.cdelt[0]
    cunit_opd = u.Unit(opd_wcs.wcs.cunit[0])
    naxis_opd = data.shape[0]

    shape = data.shape

    # OPD
    opd_pix = np.repeat(np.arange(naxis_opd), np.product(shape[1:]))
    (x,) = opd_wcs.all_pix2world(opd_pix, 0) * cunit_opd
    x = x.reshape(shape)

    # Add shift in x
    if shifts is not None:
        x += shifts[np.newaxis, :, np.newaxis]

    central_x = (central_freq / cst.c).decompose()
    sigma_x = (sigma_freq / cst.c).decompose()

    # Should actually be in the fft part
    # norm = (2 * cdelt_opd * cunit_opd).decompose()
    itg = func_itg(x, central_x, sigma_x)  # * norm

    data.data[:] = itg

    if plot:
        import matplotlib.pyplot as plt

        if sided == "double":
            spec = data._FTSData__invert_doublesided()
        elif sided == "one":
            spec = data._FTSData__invert_onesided()
        elif sided == "single" or sided == "single_low":
            spec = data.to_spectra()

        (freq,) = spec.wcs.sub([3]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
        # sigma = (freq / cst.c).decompose()

        # temporary fix :
        # coeff = (2 * cdelt_opd * cunit_opd / cst.c).to(1 / u.Hz).value

        plt.clf()
        plt.plot(freq, spec.data[:, 0, 0])  # / coeff)
        plt.plot(freq, func(freq, central_freq, sigma_freq))
        plt.axvline(central_freq.to(u.Hz).value)

    data.meta["central_freq"] = central_freq
    data.meta["sigma_freq"] = sigma_freq
    data.meta["shifts"] = shifts

    return data


# Define all tests as functions to all easy debugging
gaussian_ds = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="double")
gaussian_ds_2d = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="double_2d")
gaussian_ds_1d = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="double_1d")

rect_ds = partial(fixture_itg, func_itg=rect_itg, func=rect, sided="double")
gaussian_os = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="one")
rect_os = partial(fixture_itg, func_itg=rect_itg, func=rect, sided="one")
gaussian_ss = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="single")
rect_ss = partial(fixture_itg, func_itg=rect_itg, func=rect, sided="single")

gaussian_ss_low = partial(fixture_itg, func_itg=gaussian_itg, func=gaussian, sided="single_low")


gaussian_ds_shift = partial(
    fixture_itg,
    n_pix=3,
    cdelt_opd=0.3 * u.mm,
    shifts=[0, 0.15, 0.3] * u.mm,
    func_itg=gaussian_itg,
    func=gaussian,
    sided="double",
)
gaussian_ss_shift = partial(
    fixture_itg,
    n_pix=3,
    cdelt_opd=0.3 * u.mm,
    shifts=[0, 0.15, 0.3] * u.mm,
    func_itg=gaussian_itg,
    func=gaussian,
    sided="single",
)

rect_ds_shift = partial(
    fixture_itg,
    n_pix=3,
    cdelt_opd=0.3 * u.mm,
    shifts=[0, 0.15, 0.3] * u.mm,
    func_itg=rect_itg,
    func=rect,
    sided="double",
)
rect_ss_shift = partial(
    fixture_itg,
    n_pix=3,
    cdelt_opd=0.3 * u.mm,
    shifts=[0, 0.15, 0.3] * u.mm,
    func_itg=rect_itg,
    func=rect,
    sided="single",
)

# Redefined them as fixture for pytest
@pytest.fixture(name="gaussian_ds")
def fixture_gaussian_ds():
    return gaussian_ds()


@pytest.fixture(name="gaussian_ds_2d")
def fixture_gaussian_ds_2d():
    return gaussian_ds_2d()


@pytest.fixture(name="gaussian_ds_1d")
def fixture_gaussian_ds_1d():
    return gaussian_ds_1d()


@pytest.fixture(name="rect_ds")
def fixture_rect_ds():
    return rect_ds()


@pytest.fixture(name="gaussian_os")
def fixture_gaussian_os():
    return gaussian_os()


@pytest.fixture(name="rect_os")
def fixture_rect_os():
    return rect_os()


@pytest.fixture(name="gaussian_ss")
def fixture_gaussian_ss():
    return gaussian_ss()


@pytest.fixture(name="gaussian_ss_low")
def fixture_gaussian_ss_low():
    return gaussian_ss_low()


@pytest.fixture(name="rect_ss")
def fixture_rect_ss():
    return rect_ss()


@pytest.fixture(name="gaussian_ds_shift")
def fixture_gaussian_ds_shift():
    return gaussian_ds_shift()


@pytest.fixture(name="gaussian_ss_shift")
def fixture_gaussian_ss_shift():
    return gaussian_ss_shift()


@pytest.fixture(name="rect_ds_shift")
def fixture_rect_ds_shift():
    return rect_ds_shift()


@pytest.fixture(name="rect_ss_shift")
def fixture_rect_ss_shift():
    return rect_ss_shift()


def test_ftsdata_prop(empty_ds):
    data = empty_ds
    assert data._is_opd
    assert data._is_doublesided
    assert not data._is_onesided


def test_ftsdata_prop2(empty_os):
    data = empty_os
    assert data._is_opd
    assert not data._is_doublesided
    assert data._is_onesided


def test_extract_doublesided(gaussian_ds):
    data = gaussian_ds
    data_ds = data._extract_doublesided()

    assert np.all(data.data == data_ds.data)
    assert np.all(data.wcs.wcs.cdelt == data_ds.wcs.wcs.cdelt)
    assert np.all(data.wcs.wcs.crpix == data_ds.wcs.wcs.crpix)
    assert np.all(data.wcs.wcs.crval == data_ds.wcs.wcs.crval)


def test_extract_doublesided2(gaussian_ss):
    data = gaussian_ss
    data_ds = data._extract_doublesided()

    assert data_ds.shape[0] - 1 == (data.shape[0] - 1) * 4 // 5
    assert np.all(data_ds.data == data.data[: (data.shape[0] - 1) * 4 // 5 + 1])


def test_to_spectra_ds(gaussian_ds):
    data = gaussian_ds
    # The polynomial has to be forced when using half phases, because of numerical noise in the doublesided case
    npt.assert_almost_equal(data._FTSData__invert_doublesided().data, data.to_spectra(deg=0).data)

    # Chebychev polynomials should give the same results at order 0
    npt.assert_almost_equal(
        data._FTSData__invert_doublesided().data, data.to_spectra(deg=0, fitting_func="chebychev").data
    )


# def test_to_spectra(gaussian_ds):
#    data = gaussian_ds
#    import matplotlib.pyplot as plt
#    plt.ion()
#    plt.plot(data._FTSData__invert_doublesided().data[:, 0, 0])
#    plt.plot(data.to_spectra(deg=0).data[:, 0, 0])
#    plt.plot(data.to_spectra(deg=1).data[:, 0, 0])

#    data.wcs.wcs.crpix[2] -= 0.5
#    plt.plot(data.to_spectra(deg=1).data[:, 0, 0])


def test_to_spectra_os(gaussian_os):
    data = gaussian_os
    # There is actually no phase correction done in the to_spectra case...
    npt.assert_almost_equal(data._FTSData__invert_onesided().data, data.to_spectra(deg=0).data)
    npt.assert_almost_equal(data._FTSData__invert_onesided().data, data.to_spectra(deg=None).data)


def test_to_onesided(gaussian_ds):
    data = gaussian_ds
    data_os = data._to_onesided()
    assert data_os.shape[0] - 1 == (data.shape[0] - 1) // 2
    assert np.all(data_os.data == data.data[(data.shape[0] - 1) // 2 :])


def test_singlesided_to_onesided(gaussian_ss):
    data = gaussian_ss
    data_os = data._to_onesided()
    zpd_idx = data.wcs.sub([3]).world_to_pixel(0 * u.mm)
    assert data_os.shape[0] == np.abs(np.array([-1, data.shape[0]]) - zpd_idx).max().astype(int)


def test_singlesided_low_to_onesided(gaussian_ss):
    data = gaussian_ss
    data_os = data._to_onesided()
    zpd_idx = data.wcs.sub([3]).world_to_pixel(0 * u.mm)
    assert data_os.shape[0] == np.abs(np.array([-1, data.shape[0]]) - zpd_idx).max().astype(int)


@pytest.mark.parametrize("fixture", [gaussian_ds, gaussian_ds_2d, gaussian_ds_1d])
def test_invert_doublesided(fixture):
    data = fixture()
    central_freq = data.meta["central_freq"]
    sigma_freq = data.meta["sigma_freq"]

    spec = data._FTSData__invert_doublesided()
    _spec = spec.data.reshape(spec.data.shape[0], -1)[:, 0]
    npt.assert_almost_equal(_spec.imag, 0)

    (freq,) = spec.wcs.sub([data._opd_idx + 1]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
    _should = gaussian(freq, central_freq, sigma_freq) + gaussian(-freq, central_freq, sigma_freq)

    npt.assert_almost_equal(_spec.real, _should)


def test_invert_oneided(gaussian_os):
    data = gaussian_os
    central_freq = data.meta["central_freq"]
    sigma_freq = data.meta["sigma_freq"]

    spec = data._FTSData__invert_onesided()
    _spec = spec.data[:, 0, 0]
    npt.assert_almost_equal(_spec.imag, 0)

    (freq,) = spec.wcs.sub([3]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
    _should = gaussian(freq, central_freq, sigma_freq) + gaussian(-freq, central_freq, sigma_freq)

    npt.assert_almost_equal(_spec.real, _should)


def test_get_phase_correction_function(gaussian_ds_shift):
    data = gaussian_ds_shift
    # Ugly way to retrieve the frequency axis of the inversed cube
    dummy = data._extract_doublesided()
    (freq,) = dummy._FTSData__invert_doublesided().wcs.sub([3]).all_pix2world(np.arange(dummy.shape[0]), 0)

    # Those pcf should be simple shifts function
    shifts = np.exp(-2j * np.pi * (freq * u.Hz * (data.meta["shifts"][:, np.newaxis] / cst.c).to(1 / u.Hz)).T.value)
    _pcf = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(shifts, axes=0), axis=0), axes=0)

    pcf = data._get_phase_correction_function(niter=1, deg=1)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # Testing higher degree polynomials on simple shifts
    pcf = data._get_phase_correction_function(niter=1, deg=3)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # Test many iterations, will also work now with amplitude weighting
    pcf = data._get_phase_correction_function(niter=10, deg=1)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # Test many iterations, will succeed if also apodized
    pcf = data._get_phase_correction_function(niter=10, deg=1, doublesided_apodization=np.hanning)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # Chebychev polynomials should give the same results at order 1
    pcf = data._get_phase_correction_function(niter=10, deg=1, fitting_func="chebychev")
    assert np.allclose(pcf[:, :, 0], _pcf)

    with pytest.raises(ValueError):
        data._get_phase_correction_function(niter=10, deg=1, fitting_func="toto")


# Works, niter=10, no problem
# self = fixture_itg(n_pix=3, cdelt_opd=0.1*u.mm, shifts=[0, 0.1, 0.2]*u.mm)
# self._get_phase_correction_function(niter=3, deg=1, plot=True)
# self = fixture_itg(cdelt_opd=0.3*u.mm, n_pix=3, shifts=[0, 0.1, 0.2]*u.mm, func_itg=gaussian_itg, func=gaussian, sided='double')
# self._get_phase_correction_function(niter=1, deg=1, plot=True)

"""
import matplotlib.pyplot as plt

plt.ion()
itg = fixture_itg(
    n_itg=1024, cdelt_opd=0.3 * u.mm, central_freq=120 * u.GHz, sigma_freq=10 * u.GHz, func=rect, func_itg=rect_itg
)
(opd,) = itg.wcs.sub([3]).all_pix2world(np.arange(itg.shape[0]), 0) * u.Unit(itg.wcs.wcs.cunit[2])
itg_max = np.max(itg.wcs.sub([3]).all_pix2world([0, itg.shape[0] - 1], 0)) * u.Unit(itg.wcs.wcs.cunit[2])
cube = itg.to_spectra()
(freq,) = cube.wcs.sub([3]).all_pix2world(np.arange(cube.shape[0]), 0) * u.Unit(cube.wcs.wcs.cunit[2])
expected_unc = rect(freq, itg.meta["central_freq"], itg.meta["sigma_freq"])
from scipy.signal import fftconvolve

_ils = ils(freq, itg_max)
expected = fftconvolve(expected_unc, _ils, mode="same")

plt.close("all")
plt.plot(freq, cube.data[:, 0, 0], label="cube")
plt.plot(freq, expected_unc, label="expected unc")
plt.plot(freq, expected, label="expected")
plt.plot(freq, _ils, label="ils")

plt.legend()
"""
