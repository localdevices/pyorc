import platform

import numpy as np
import pytest

import pyorc


def get_tolerance():
    """Tolerance for checking returned image intensity values.

    Tolerance is needed because arm / aarch64 processors use slightly different decoding methods.
    """
    if platform.machine().startswith("arm") or platform.machine() == "aarch64":
        return 5  # More lenient on ARM
    return 0


def test_camera_config(vid_cam_config):
    assert isinstance(
        vid_cam_config.camera_config, pyorc.CameraConfig
    ), "Video camera_config property is not a pyorc.CameraConfig object"


def test_end_frame(vid):
    assert vid.end_frame == 2


def test_start_frame(vid):
    assert vid.start_frame == 0


def test_h_a_none(vid):
    assert vid.h_a is None


def test_h_a_float(vid_cam_config):
    assert vid_cam_config.h_a == 0.0


def test_fps(vid):
    print(vid.fps)
    assert vid.fps == 30.0


@pytest.mark.parametrize(
    ("video", "method", "result"),
    [
        ("vid_cam_config", "grayscale", [85, 71, 65, 80]),
        ("vid_cam_config_stabilize", "grayscale", [5, 88, 78, 73]),
        ("vid_cam_config", "rgb", [84, 91, 57, 70]),
        ("vid_cam_config", "hsv", [36, 95, 91, 36]),
    ],
)
def test_get_frame(video, method, result, request):
    tolerance = get_tolerance()
    video = request.getfixturevalue(video)
    frame = video.get_frame(1, method=method)
    assert np.allclose(frame.flatten()[0:4], result, atol=tolerance)


@pytest.mark.parametrize(
    ("video", "method"),
    [
        ("vid_cam_config_nonlazy", None),
        ("vid_cam_config", "grayscale"),
        ("vid_cam_config", "rgb"),
        ("vid_cam_config", "hsv"),
        ("vid_cam_config", "hue"),
        ("vid_cam_config", "sat"),
        ("vid_cam_config", "val"),
    ],
)
def test_get_frames(video, method, request):
    video = request.getfixturevalue(video)
    # check if the right amount of frames is extracted
    if method:
        frames = video.get_frames(method=method)
    else:
        frames = video.get_frames()
    assert len(frames) == video.end_frame - video.start_frame + 1
    # check if the time difference is well taken from the fps of the video
    assert np.allclose(np.diff(frames.time.values), [1.0 / video.fps])
