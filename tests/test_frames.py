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


def test_range(frames_grayscale):
    frames_range = frames_grayscale.frames.range()
    assert len(frames_range.shape) == 2, "shape of result is not 2D"
    assert frames_range.dtype == frames_grayscale.dtype, f'dtype of result is {frames_range.dtype}, expected "uint8"'
    assert np.allclose(frames_range.values.flatten()[-4:], [22, 27, 22, 31]), "last 4 values are not as expected"


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
    assert np.allclose(frames_edge.values.flatten()[-4:], [-5.6953125, 4.0703125, 8.0625, 4.3125])


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
    plt.close("all")
    frames[idx].frames.plot(mode="camera")


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
    ("window_size", "engine", "ensemble_corr", "result"),
    [
        (10, "openpiv", [0.08245023, 0.06594574, 0.11719926, 0.10809214]),
        (10, "numba", True, [0.08245023, 0.06594574, 0.11719926, 0.10809214]),
        (10, "numba", False, [0.08245023, 0.06594574, 0.11719926, 0.10809214]),
    ],
)
def test_get_piv(frames_proj, window_size, engine, corr_mean, result):
    piv = frames_proj.frames.get_piv(window_size=window_size, corr_mean=corr_mean, engine=engine)
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
