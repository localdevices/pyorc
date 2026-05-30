import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from pyorc.helpers import to_geotiff


@pytest.fixture()
def transform():
    """Return a simple affine transform for a 4x5 grid with 1m resolution."""
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)


@pytest.fixture()
def data_2d():
    """Return a simple 4-row, 5-column 2D float32 array."""
    return np.random.rand(4, 5).astype(np.float32)


@pytest.fixture()
def data_rgb():
    """Return a simple 4-row, 5-column, 3-band uint8 array mimicking an RGB frame."""
    return np.random.randint(0, 256, (4, 5, 3), dtype=np.uint8)


@pytest.fixture()
def crs():
    return CRS.from_epsg(32636)


@pytest.mark.parametrize(
    ("data_fixture", "crs_fixture", "fn"),
    [
        ("data_2d", None, "out_2d.tif"),
        ("data_2d", "crs", "out_2d_crs.tif"),
        ("data_rgb", "crs", "out_rgb_crs.tif"),
    ],
)
def test_to_geotiff(tmp_path, transform, data_fixture, crs_fixture, fn, request):
    data = request.getfixturevalue(data_fixture)
    crs_value = request.getfixturevalue(crs_fixture) if crs_fixture else None
    fn_path = str(tmp_path / fn)
    to_geotiff(fn=fn_path, data=data, transform=transform, crs=crs_value)

    with rasterio.open(fn_path) as ds:
        if data.ndim == 2:
            assert ds.count == 1
        else:
            assert ds.count == data.shape[2]
        assert ds.width == data.shape[1]
        assert ds.height == data.shape[0]
        assert ds.transform == transform
        written = ds.read().flatten()
        transposed_array = np.transpose(np.atleast_3d(data), (2, 0, 1))
        np.testing.assert_array_equal(written, transposed_array.flatten())
