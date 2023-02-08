import pytest
import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.parametrize(
    "frames, resolution, dims, shape",
    [
        (pytest.lazy_fixture("frames_grayscale"), 0.1, 3, (83, 97)),
        (pytest.lazy_fixture("frames_grayscale"), 0.01, 3, (821, 965)),
        (pytest.lazy_fixture("frames_grayscale"), 0.005, 3, (1642, 1930)),
        (pytest.lazy_fixture("frames_rgb"), 0.1, 4, (83, 97, 3)),
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
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # import matplotlib.pyplot as plt
    # plt.imshow(frames_proj[0])
    # plt.show()


@pytest.mark.parametrize(
    "frames, samples",
    [
        (pytest.lazy_fixture("frames_grayscale"), 2),
        (pytest.lazy_fixture("frames_grayscale_shift"), 2),
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
    assert(np.allclose(frames_edge.values.flatten()[-4:], [6.328125, 2.34375, 0.0625, 0.40625]))



def test_reduce_rolling(frames_grayscale, samples=1):
    frames_reduced = frames_grayscale.frames.reduce_rolling(samples=samples)
    assert(frames_reduced.shape == frames_grayscale.shape)


@pytest.mark.parametrize(
    "frames",
    [
        pytest.lazy_fixture("frames_grayscale"),
        pytest.lazy_fixture("frames_rgb"),
    ]
)
@pytest.mark.parametrize(
    "idx",
    [0, -1]
)
def test_plot(frames, idx):
    frames[idx].frames.plot()
    frames[idx].frames.plot(mode="camera")
    plt.close("all")

@pytest.mark.parametrize(
    "idx",
    [0, -1]
)
def test_plot_proj(frames_proj, idx):
    frames_proj[idx].frames.plot()
    frames_proj[idx].frames.plot(mode="geographical")
    plt.close("all")


@pytest.mark.parametrize(
    "window_size, result",
    [
        (5, [0.04290744, np.nan, np.nan, 0.07388695]),
        (10, [0.25739092, 0.22005227, 0.30368298, 0.33715272]),
        (15, [0.27460322, 0.3151794, 0.26885322, 0.25401443])
    ]
)
def test_get_piv(frames_proj, window_size, result):
    piv = frames_proj.frames.get_piv(window_size=window_size)
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    # check if results are stable
    assert(np.allclose(piv_mean["v_x"].values.flatten()[-4:], result, equal_nan=True))


@pytest.mark.parametrize(
    "frames",
    [
        pytest.lazy_fixture("frames_grayscale"),
        pytest.lazy_fixture("frames_rgb"),
        pytest.lazy_fixture("frames_proj"),
    ]
)
def test_to_ani(frames, ani_mp4):
    frames.frames.to_ani(ani_mp4, progress_bar=False)


@pytest.mark.parametrize(
    "frames",
    [
        pytest.lazy_fixture("frames_grayscale"),
        pytest.lazy_fixture("frames_rgb_stabilize"),
        pytest.lazy_fixture("frames_proj"),
    ]
)
def test_to_video(frames, ani_mp4):
    # only store the first 3 frames
    frames[0:3].frames.to_video(ani_mp4)