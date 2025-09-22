"""Tests for masks at API level."""

import pytest


def test_mask_angle(piv):
    # check if the method runs
    piv.velocimetry.mask.angle(inplace=True)


def test_mask_minmax(piv):
    # check if the method runs
    mask1 = piv.velocimetry.mask.minmax(inplace=False)
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    mask2 = piv_mean.velocimetry.mask.minmax(inplace=True)
    piv.velocimetry.mask([mask1, mask2])


def test_mask_corr(piv):
    # check if the method runs
    piv.velocimetry.mask.corr(inplace=True, tolerance=0.3)


def test_mask_count(piv):
    piv.velocimetry.mask.count(inplace=True)


def test_mask_rolling(piv):
    # check if the method runs
    piv.velocimetry.mask.rolling(inplace=True, tolerance=0.4)


def test_mask_outliers(piv):
    # check if the method runs
    piv.velocimetry.mask.outliers(inplace=True, mode="or")


def test_mask_variance(piv):
    # check if the method runs
    piv.velocimetry.mask.variance(inplace=True, tolerance=1.0, mode="or")


def test_mask_window_nan(piv):
    # check if the method runs
    # first do filter that creates some missings
    piv.velocimetry.mask.minmax(s_max=0.6, inplace=True)
    piv.velocimetry.mask.window_nan(inplace=True)


def test_mask_window_mean(piv):
    # check if the method runs
    # piv = piv.isel(time=1)
    piv.velocimetry.mask.window_mean(inplace=True)


def test_mask(piv):
    # make two masks, one which works only with time, and one without, combine.
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    mask1 = piv.velocimetry.mask.angle()
    mask2 = piv_mean.velocimetry.mask.window_mean()
    # combine masks and test inplace=False/True
    piv.velocimetry.mask([mask1, mask2])
    piv.velocimetry.mask([mask1, mask2], inplace=True)


# also test if error messages appear correctly
def test_error_no_time(piv):
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    # now test if an error is raised when no time is present
    with pytest.raises(AssertionError, match='This mask requires dimension "time"'):
        piv_mean.velocimetry.mask.variance()


def test_error_single_time_step(piv):
    piv_sel = piv.isel(time=slice(0, 1))
    # now test if an error is raised when no time is present
    with pytest.raises(AssertionError, match="This mask requires multiple timesteps"):
        piv_sel.velocimetry.mask.variance()
