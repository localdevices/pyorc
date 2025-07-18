import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    # Try to import cartopy
    import cartopy  # noqa: F401

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


@pytest.mark.parametrize(("distance", "nr_points"), [(None, 36), (0.1, 50), (0.3, 17)])
def test_get_transect(piv, cross_section, distance, nr_points):
    x, y, z = (cross_section["x"], cross_section["y"], cross_section["z"])
    ds_points = piv.velocimetry.get_transect(x, y, z, crs=32735, rolling=4, distance=distance)
    # check if the angle is computed correctly
    assert np.isclose(ds_points["v_dir"][0].values, -4.41938864)
    assert (len(ds_points.points)) == nr_points


@pytest.mark.parametrize("mode", ["local", "camera", "geographical"])
@pytest.mark.parametrize(
    "method",
    [
        "quiver",
        "pcolormesh",
        "scatter",
        "streamplot",
    ],
)
def test_plot(piv, mode, method):
    if mode == "geographical" and not CARTOPY_AVAILABLE:
        pytest.importorskip("cartopy", "Cartopy is required for geographical plotting")
    plot = True
    if method == "streamplot":
        if mode != "local":
            # skipping the test, because streamplot only works local
            plot = False
    if plot:
        piv.mean(dim="time", keep_attrs=True).velocimetry.plot(method=method, mode=mode, add_colorbar=True)
    plt.close("all")
