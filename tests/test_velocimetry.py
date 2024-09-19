import copy

import pytest
import numpy as np
import matplotlib.pyplot as plt


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
    assert(np.isclose(ds_points["v_dir"][0].values, -4.41938864))
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
    if mode == "geographical":
        try:
            import cartopy
        except:
            print("Cartopy is missing, skipping cartopy dependent test")
            plot = False
    if method == "streamplot":
        if mode != "local":
            # skipping the test, because streamplot only works local
            plot = False
    if plot:
        piv.mean(dim="time", keep_attrs=True).velocimetry.plot(method=method, mode=mode)
        plt.show(block=False)
    plt.close("all")
