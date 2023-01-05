import pytest
import numpy as np
import matplotlib.pyplot as plt

def test_get_river_flow(piv_transect):
    # fill method is already tested in get_q, so choose default only
    piv_transect.transect.get_q()
    piv_transect.transect.get_river_flow()
    # because we only have PIV for one time step, all quantiles will have the same values
    assert(np.allclose(piv_transect.river_flow.values, [0.07360698, 0.0772654 , 0.08183843, 0.08641146, 0.09006988]))


@pytest.mark.parametrize(
    "fill_method",
    [
        "zeros",
        "log_interp",
        "log_fit",
        "interpolate",
    ]
)
def test_get_q(piv_transect, fill_method):
    f = piv_transect.transect.get_q(fill_method=fill_method)
    # assert if filled values are more complete than non-filled


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
        "scatter",
    ]
)
def test_plot(piv_transect, mode, method):
    piv_transect.transect.get_q()
    piv_transect.isel(quantile=2).transect.plot(method=method, mode=mode)
    plt.close("all")
