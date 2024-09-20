import pytest
import pyorc
import numpy as np


def test_camera_config(vid_cam_config):
    assert(isinstance(vid_cam_config.camera_config, pyorc.CameraConfig)), "Video camera_config property is not a pyorc.CameraConfig object"


def test_end_frame(vid):
    assert(vid.end_frame == 2)

def test_start_frame(vid):
    assert(vid.start_frame == 0)


def test_h_a_none(vid):
    assert(vid.h_a is None)


def test_h_a_float(vid_cam_config):
    assert(vid_cam_config.h_a == 0.)


def test_fps(vid):
    print(vid.fps)
    assert(vid.fps == 30.)


@pytest.mark.parametrize(
    "video, method, result",
    [
        ("vid_cam_config", "grayscale", [85, 71, 65, 80]),
        ("vid_cam_config_stabilize", "grayscale", [60, 78, 70, 76]),
        ("vid_cam_config", "rgb", [84, 91, 57, 70]),
        ("vid_cam_config", "hsv", [36, 95, 91, 36])
    ]
)
def test_get_frame(video, method, result, request):
    video = request.getfixturevalue(video)
    frame = video.get_frame(1, method=method)
    assert(np.allclose(frame.flatten()[0:4], result))


@pytest.mark.parametrize(
    "video, method",
    [
        ("vid_cam_config_nonlazy", None),
        ("vid_cam_config", "grayscale"),
        ("vid_cam_config", "rgb"),
        ("vid_cam_config", "hsv")
    ]
)
def test_get_frames(video, method, request):
    video = request.getfixturevalue(video)
    # check if the right amount of frames is extracted
    if method:
        frames = video.get_frames(method=method)
    else:
        frames = video.get_frames()
    assert(len(frames) == video.end_frame - video.start_frame + 1)
    # check if the time difference is well taken from the fps of the video
    assert(np.allclose(np.diff(frames.time.values), [1./video.fps]))
