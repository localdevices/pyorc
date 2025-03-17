import numpy as np
import pytest

from pyorc import CrossSection

try:
    import cartopy  # noqa: F401

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def test_get_river_flow(piv_transect):
    # fill method is already tested in get_q, so choose default only
    piv_transect.transect.get_q()
    piv_transect.transect.get_river_flow()
    # we allow for 0.001 m3/s deviation for differences in versions of libs
    assert np.allclose(
        # piv_transect.river_flow.values, [0.0821733, 0.08626413, 0.09137767, 0.09649121, 0.10058204], atol=0.001
        piv_transect.river_flow.values,
        [0.09134876, 0.09304505, 0.09516542, 0.09728579, 0.09898209],
        atol=0.001,
    )


@pytest.mark.parametrize(
    "fill_method",
    [
        "zeros",
        "log_interp",
        "log_fit",
        "interpolate",
    ],
)
def test_get_q(piv_transect, fill_method):
    piv_transect.load()
    piv_transect.transect.get_q(fill_method=fill_method)
    # assert if filled values are more complete than non-filled


def test_get_wetted_perspective(piv_transect):
    piv_transect.load()
    f = piv_transect.transect.get_q()
    f.transect.get_wetted_perspective(h=0.0)


def test_get_cross_section(piv_transect):
    piv_transect.load()
    assert isinstance(piv_transect.transect.cross_section, CrossSection)


@pytest.mark.parametrize(
    "mode",
    ["local", "camera", "geographical"],
)
@pytest.mark.parametrize(
    "method",
    [
        "quiver",
        # "scatter",
    ],
)
def test_plot(piv_transect, mode, method):
    if mode == "geographical" and not CARTOPY_AVAILABLE:
        pytest.importorskip("cartopy", "Cartopy is required for geographical plotting")
    piv_transect.transect.get_q()
    piv_transect.isel(quantile=2).transect.plot(method=method, mode=mode, add_text=True)
