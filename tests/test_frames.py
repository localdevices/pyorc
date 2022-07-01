import pytest
import pyorc
import numpy as np


@pytest.mark.parametrize(
    "frames, resolution, dims, shape",
    [
        (pytest.lazy_fixture("frames_grayscale"), 0.1, 3, (90, 110)),
        (pytest.lazy_fixture("frames_grayscale"), 0.01, 3, (840, 1100)),
        (pytest.lazy_fixture("frames_grayscale"), 0.005, 3, (1680, 2190)),
        (pytest.lazy_fixture("frames_rgb"), 0.1, 4, (90, 110, 3)),
    ]
)
def test_project(frames, resolution, dims, shape):
# def test_project(frames_grayscale, resolution=0.01, dims=3, shape=(840, 1100)):
    frames_proj = frames.frames.project(resolution=resolution)
    # check amount of time steps is equal
    assert(len(frames_proj.time) == len(frames.time))
    # check if the amount of dims is as expected (different for rgb)
    assert(len(frames_proj.dims) == dims), f"Expected nr of dims is {dims}, but {len(frames_proj.dims)} found"
    # check shape of x, y grids
    assert(frames_proj.isel(time=0).shape == shape), f"Projected frames shape {frames_proj.isel(time=0).shape} do not have expected shape {shape}"


@pytest.mark.parametrize(
    "frames, samples",
    [
        (pytest.lazy_fixture("frames_grayscale"), 2),
        (pytest.lazy_fixture("frames_proj"), 2),
    ]
)
def test_normalize(frames, samples):
    frames_norm = frames.frames.normalize(samples=samples)
    assert(frames_norm[0, 0, 0].values.dtype == "uint8"), f'dtype of result is {frames_norm[0, 0, 0].values.dtype}, expected "uint8"'


def test_edge_detect(frames_proj):
    frames_edge = frames_proj.frames.edge_detect()
    assert(frames_edge.shape == frames_proj.shape)
    assert(frames_edge[0, 0, 0].values.dtype == "float32"), f'dtype of result is {frames_edge[0, 0, 0].values.dtype}, expected "float32"'
    assert(np.allclose(frames_edge.values.flatten()[-4:], [-3.6447144, -6.9251404, -5.4156494, -3.6206055]))



def test_reduce_rolling(frames_grayscale, samples=1):
    frames_reduced = frames_grayscale.frames.reduce_rolling(samples=samples)
    assert(frames_reduced.shape == frames_grayscale.shape)


@pytest.mark.parametrize(
    "frames",
    [
        pytest.lazy_fixture("frames_grayscale"),
        pytest.lazy_fixture("frames_rgb"),
        pytest.lazy_fixture("frames_proj"),
    ]
)
@pytest.mark.parametrize(
    "idx",
    [0, -1]
)
def test_plot(frames, idx):
    frames[idx].plot()

