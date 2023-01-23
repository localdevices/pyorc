import copy

import pytest
import numpy as np
import matplotlib.pyplot as plt


def test_filter_temporal_angle(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_temporal_angle()



def test_filter_temporal_velocity(piv):
    # check if the method runs
    piv.velocimetry.filter_temporal_velocity()


def test_filter_temporal_corr(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_temporal_corr()


def test_filter_temporal_neighbour(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_temporal_neighbour()



def test_filter_temporal_std(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_temporal_std()




def test_filter_temporal(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_temporal()


def test_filter_spatial_nan(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_spatial_nan()



def test_filter_spatial_median(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_spatial_median()



def test_filter_spatial(piv):
    # check if the method runs DEPRECATED
    piv.velocimetry.filter_spatial()


def test_replace_outliers(piv):
    # currently only works time-reduced
    piv.mean(dim="time").velocimetry.replace_outliers()


@pytest.mark.parametrize(
    "distance, nr_points",
    [
        (None, 39),
        (0.1, 50),
        (0.3, 17)
    ]
)
def test_get_transect(piv, cross_section, distance, nr_points):
    x, y, z = cross_section["x"], cross_section["y"], cross_section["z"]
    ds_points = piv.velocimetry.get_transect(x, y, z, crs=32735, rolling=4, distance=distance)
    # check if the angle is computed correctly
    assert(np.isclose(ds_points["v_dir"][0].values, -4.63115779))
    assert(len(ds_points.points)) == nr_points


@pytest.mark.parametrize(
    "mode",
    [
        "local",
        "camera",
        "geographical"
    ]
)
@pytest.mark.parametrize(
    "method",
    [
        "quiver",
        "pcolormesh",
        "scatter",
        "streamplot",
    ]
)
def test_plot(piv, mode, method):
    plot = True
    if method == "streamplot":
        if mode != "local":
            # skipping the test, because streamplot only works local
            plot = False
    if plot:
        piv.mean(dim="time", keep_attrs=True).velocimetry.plot(method=method, mode=mode)
    plt.close("all")


