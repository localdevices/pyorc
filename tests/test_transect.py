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
        piv_transect.river_flow.values,
        [0.07832181, 0.07917073, 0.08023188, 0.08129303, 0.08214195],
        atol=0.001,
    )
    v_surf = piv_transect.transect.get_v_surf()
    assert np.allclose(
        v_surf.values,
        [0.14660551, 0.14832331, 0.15047056, 0.15261781, 0.15433561],
        atol=0.001,
    )

    # also get bulk and average velocities and check if the values are logical
    v_bulk = piv_transect.transect.get_v_bulk()
    assert np.allclose(piv_transect.river_flow.values / piv_transect.transect.wetted_surface, v_bulk.values)


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
