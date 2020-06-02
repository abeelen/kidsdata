import pytest

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


@pytest.fixture(name="empty_ss")
def empty_ss_fixture():
    return empty_ss(n_itg=1024, n_pix=1)


def rect(x, center, sigma):
    output = (np.abs(x - center) <= sigma / 2).astype(np.float)
    return output


def rect_itg(x, center_x, sigma_x):
    shift = np.cos(2 * np.pi * u.Quantity(center_x * x).decompose().value)
    return shift * np.sinc(u.Quantity(sigma_x * x).decompose().value) * np.abs(sigma_x)


def gaussian(x, center, sigma):
    output = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return output.decompose()


def gaussian_itg(x, center_x, sigma_x):
    shift = np.cos(2 * np.pi * u.Quantity(center_x * x).decompose().value)
    return (
        shift
        * np.exp(-(2 * np.pi ** 2 * u.Quantity(sigma_x * x).decompose().value ** 2))
        * np.sqrt(2 * np.pi)
        * sigma_x
    )


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
    elif sided == "one":
        data = empty_os(n_itg=n_itg, n_pix=n_pix)
    elif sided == "single":
        data = empty_ss(n_itg=n_itg, n_pix=n_pix)

    data.wcs.wcs.cdelt[2] = cdelt_opd.to(data.wcs.wcs.cunit[2]).value

    # temporary fix :
    # data.wcs.wcs.ctype[2] = "opd"

    opd_wcs = data.wcs.sub([3])
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

    norm = (2 * cdelt_opd * cunit_opd).decompose()
    itg = func_itg(x, central_x, sigma_x) * norm

    data.data[:] = itg

    if plot:
        import matplotlib.pyplot as plt

        if sided == "double":
            spec = data._FTSData__invert_doublesided()
        elif sided == "one":
            spec = data._FTSData__invert_onesided()
        elif sided == "single":
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


@pytest.fixture
def gaussian_ds():
    return fixture_itg(func_itg=gaussian_itg, func=gaussian, sided="double")


@pytest.fixture
def rect_ds():
    return fixture_itg(func_itg=rect_itg, func=rect, sided="double")


@pytest.fixture
def gaussian_os():
    return fixture_itg(func_itg=gaussian_itg, func=gaussian, sided="one")


@pytest.fixture
def rect_os():
    return fixture_itg(func_itg=rect_itg, func=rect, sided="one")


@pytest.fixture
def gaussian_ss():
    return fixture_itg(func_itg=gaussian_itg, func=gaussian, sided="single")


@pytest.fixture
def rect_ss():
    return fixture_itg(func_itg=rect_itg, func=rect, sided="single")


@pytest.fixture
def gaussian_ds_shift():
    return fixture_itg(
        n_pix=3,
        cdelt_opd=0.3 * u.mm,
        shifts=[0, 0.15, 0.3] * u.mm,
        func_itg=gaussian_itg,
        func=gaussian,
        sided="double",
    )


@pytest.fixture
def gaussian_ss_shift():
    return fixture_itg(
        n_pix=3,
        cdelt_opd=0.3 * u.mm,
        shifts=[0, 0.15, 0.3] * u.mm,
        func_itg=gaussian_itg,
        func=gaussian,
        sided="single",
    )


@pytest.fixture
def rect_ds_shift():
    return fixture_itg(
        n_pix=3, cdelt_opd=0.3 * u.mm, shifts=[0, 0.15, 0.3] * u.mm, func_itg=rect_itg, func=rect, sided="double"
    )


@pytest.fixture
def rect_ss_shift():
    return fixture_itg(
        n_pix=3, cdelt_opd=0.3 * u.mm, shifts=[0, 0.15, 0.3] * u.mm, func_itg=rect_itg, func=rect, sided="single"
    )


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
    npt.assert_almost_equal(data._FTSData__invert_doublesided().data, data.to_spectra(deg=0).data)


# def test_to_spectra(gaussian_ds):
#    data = gaussian_ds
#    import matplotlib.pyplot as plt
#    plt.ion()
#    plt.plot(data._FTSData__invert_doublesided().data[:, 0, 0])
#    plt.plot(data.to_spectra(deg=0).data[:, 0, 0])
#    plt.plot(data.to_spectra(deg=1).data[:, 0, 0])

#    data.wcs.wcs.crpix[2] -= 0.5
#    plt.plot(data.to_spectra(deg=1).data[:, 0, 0])


def test_to_spectra_ss(gaussian_os):
    data = gaussian_os
    npt.assert_almost_equal(data._FTSData__invert_onesided().data, data.to_spectra(deg=0).data)
    npt.assert_almost_equal(data._FTSData__invert_onesided().data, data.to_spectra(deg=None).data)


def test_to_onesided(gaussian_ds):
    data = gaussian_ds
    data_os = data._to_onesided()
    assert data_os.shape[0] - 1 == (data.shape[0] - 1) // 2
    assert np.all(data_os.data == data.data[(data.shape[0] - 1) // 2 :])


def test_invert_doublesided(gaussian_ds):
    data = gaussian_ds
    central_freq = data.meta["central_freq"]
    sigma_freq = data.meta["sigma_freq"]

    spec = data._FTSData__invert_doublesided()
    _spec = spec.data[:, 0, 0]
    npt.assert_almost_equal(_spec.imag, 0)

    (freq,) = spec.wcs.sub([3]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
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

    pcf = data._get_phase_correction_function(niter=1, deg=1, real_clip=1e-6)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # Test many iterations, will fail if no apodization
    pcf = data._get_phase_correction_function(niter=10, deg=1, real_clip=1e-6)
    assert not np.allclose(pcf[:, :, 0], _pcf)

    # Or too low real_clipping :
    pcf = data._get_phase_correction_function(niter=10, deg=1, real_clip=1e-10)
    assert not np.allclose(pcf[:, :, 0], _pcf)

    # Test many iterations, will succeed if apodized
    pcf = data._get_phase_correction_function(niter=10, deg=1, real_clip=1e-6, doublesided_apodization=np.hanning)
    assert np.allclose(pcf[:, :, 0], _pcf)

    # but still diverge with too low clipping
    pcf = data._get_phase_correction_function(niter=10, deg=1, real_clip=1e-10, doublesided_apodization=np.hanning)
    assert not np.allclose(pcf[:, :, 0], _pcf)


# Works, niter=10, no problem
# self = fixture_itg(n_pix=3, cdelt_opd=0.1*u.mm, shifts=[0, 0.1, 0.2]*u.mm)
# self._get_phase_correction_function(niter=3, deg=1, plot=True)
# self = fixture_itg(cdelt_opd=0.3*u.mm, n_pix=3, shifts=[0, 0.1, 0.2]*u.mm, func_itg=gaussian_itg, func=gaussian, sided='double')
# self._get_phase_correction_function(niter=1, deg=1, plot=True)
