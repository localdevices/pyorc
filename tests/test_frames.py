import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize(
    ("frames", "resolution", "method", "dims", "shape", "kwargs"),
    [
        ("frames_grayscale", 0.25, "numpy", 3, (19, 15), {}),
        ("frames_grayscale", 0.25, "numpy", 3, (19, 15), {"reducer": "mean"}),
        ("frames_rgb", 0.25, "numpy", 4, (19, 15, 3), {"reducer": "mean"}),
        ("frames_rgb", 0.25, "numpy", 4, (19, 15, 3), {}),
        ("frames_grayscale", 0.25, "cv", 3, (19, 15), {}),
        ("frames_grayscale", 0.01, "cv", 3, (475, 371), {}),
        ("frames_grayscale", 0.05, "cv", 3, (95, 74), {}),
        ("frames_rgb", 0.25, "cv", 4, (19, 15, 3), {}),
    ],
)
def test_project(frames, resolution, method, dims, shape, kwargs, request):
    frames = request.getfixturevalue(frames)
    frames_proj = frames.frames.project(resolution=resolution, method=method, **kwargs)
    # check amount of time steps is equal
    assert len(frames_proj.time) == len(frames.time)
    # check if the amount of dims is as expected (different for rgb)
    assert len(frames_proj.dims) == dims, f"Expected nr of dims is {dims}, but {len(frames_proj.dims)} found"
    # check shape of x, y grids
    assert (
        frames_proj.isel(time=0).shape == shape
    ), f"Projected frames shape {frames_proj.isel(time=0).shape} do not have expected shape {shape}"


@pytest.mark.parametrize(
    ("frames", "samples"),
    [
        ("frames_grayscale", 2),
        ("frames_grayscale_shift", 2),
        ("frames_proj", 2),
    ],
)
def test_normalize(frames, samples, request):
    frames = request.getfixturevalue(frames)
    frames_norm = frames.frames.normalize(samples=samples)
    assert (
        frames_norm[0, 0, 0].values.dtype == "uint8"
    ), f'dtype of result is {frames_norm[0, 0, 0].values.dtype}, expected "uint8"'


def test_smooth(frames_grayscale):
    frames_smooth = frames_grayscale.frames.smooth()
    assert frames_smooth.shape == frames_grayscale.shape
    assert (
        frames_smooth[0, 0, 0].values.dtype == "float32"
    ), f'dtype of result is {frames_smooth[0, 0, 0].values.dtype}, expected "float32"'
    assert np.allclose(frames_smooth.values.flatten()[-4:], [158.125, 153.5, 151.375, 151.0])


def test_edge_detect(frames_proj):
    frames_edge = frames_proj.frames.edge_detect()
    assert frames_edge.shape == frames_proj.shape
    assert (
        frames_edge[0, 0, 0].values.dtype == "float32"
    ), f'dtype of result is {frames_edge[0, 0, 0].values.dtype}, expected "float32"'
    # assert(np.allclose(frames_edge.values.flatten()[-4:], [-1.3828125, -4.3359375,  1.71875  ,  7.234375 ]))
    assert np.allclose(frames_edge.values.flatten()[-4:], [-6.0390625, 0.8671875, 6.4765625, 4.40625])


def test_reduce_rolling(frames_grayscale, samples=1):
    frames_reduced = frames_grayscale.frames.reduce_rolling(samples=samples)
    assert frames_reduced.shape == frames_grayscale.shape


@pytest.mark.parametrize(
    "frames",
    [
        "frames_grayscale",
        "frames_rgb",
    ],
)
@pytest.mark.parametrize("idx", [0, -1])
def test_plot(frames, idx, request):
    frames = request.getfixturevalue(frames)
    frames[idx].frames.plot()
    frames[idx].frames.plot(mode="camera")
    plt.close("all")


@pytest.mark.parametrize("idx", [0, -1])
def test_plot_proj(frames_proj, idx):
    frames_proj[idx].frames.plot()
    plt.show(block=False)
    plt.close("all")
    try:
        frames_proj[idx].frames.plot(mode="geographical")
        plt.show(block=False)
        plt.close("all")
    except ImportError:
        print("Cartopy is missing, skipping cartopy dependent test")


@pytest.mark.parametrize(
    ("window_size", "engine", "result"),
    [
        # (5, [np.nan, np.nan, np.nan, 0.06877007]),
        (10, "openpiv", [0.11740075, 0.09619355, 0.16204849, 0.14154269]),
        # (10, "ffpiv", [0.11740075, 0.09619355, 0.16204849, 0.14154269]),
        # (15, [0.21774408, 0.21398547, 0.25068682, 0.26456946])
    ],
)
def test_get_piv(frames_proj, window_size, engine, result):
    piv = frames_proj.frames.get_piv(window_size=window_size, engine=engine)
    print(piv["v_x"].shape)
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    # check if results are stable
    assert np.allclose(piv_mean["v_x"].values.flatten()[-4:], result, equal_nan=True)


@pytest.mark.parametrize(
    "window_size",
    [
        26,
    ],
)
def test_compare_piv(frames_proj, window_size):
    frames_proj.load()
    piv = frames_proj.frames.get_piv(window_size=window_size, engine="openpiv")
    piv.load()
    u1, v1 = piv["v_x"].mean(dim="time").values, piv["v_y"].mean(dim="time").values
    piv2 = frames_proj.frames.get_piv(window_size=window_size, engine="numba")
    piv2.load()
    u2, v2 = piv2["v_x"].mean(dim="time").values, piv2["v_y"].mean(dim="time").values
    assert np.allclose(u2, u1, atol=1e-5, rtol=1e-4), "too large differences between `u` of openpiv and ffpiv"
    assert np.allclose(v2, v1, atol=1e-5, rtol=1e-4), "too large differences between `v` of openpiv and ffpiv"


@pytest.mark.parametrize(
    "frames",
    [
        "frames_grayscale",
        "frames_rgb",
        "frames_proj",
    ],
)
def test_to_ani(frames, ani_mp4, request):
    frames = request.getfixturevalue(frames)
    frames.frames.to_ani(ani_mp4, progress_bar=False)


@pytest.mark.parametrize(
    "frames",
    [
        "frames_grayscale",
        "frames_rgb_stabilize",
        "frames_proj",
    ],
)
def test_to_video(frames, ani_mp4, request):
    frames = request.getfixturevalue(frames)
    # only store the first 3 frames
    frames[0:3].frames.to_video(ani_mp4)
