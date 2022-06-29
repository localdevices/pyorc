import pyorc
import numpy as np

def test_camera_config(vid_cam_config):
    assert(isinstance(vid_cam_config.camera_config, pyorc.CameraConfig)), "Video camera_config property is not a pyorc.CameraConfig object"


def test_end_frame(vid):
    assert(vid.end_frame == 1)


def test_start_frame(vid):
    assert(vid.start_frame == 0)


def test_h_a_none(vid):
    assert(vid.h_a is None)


def test_h_a_float(vid_cam_config):
    assert(vid_cam_config.h_a == 0.)


def test_fps(vid):
    print(vid.fps)
    assert(vid.fps == 30.)


def test_get_frame_grayscale(vid_cam_config):
    grays = vid_cam_config.get_frame(1, method="grayscale")
    # test if the first 4 numbers are as expected
    assert(np.allclose(grays.flatten()[0:4], [77.33333333, 63.33333333, 57.33333333, 72.33333333]))


def test_get_frame_rgb(vid_cam_config):
    rgb = vid_cam_config.get_frame(2, method="rgb")
    # test if the first 4 numbers are as expected
    assert(np.allclose(rgb.flatten()[0:4], [84, 91, 57, 70]))


def test_get_frames(vid_cam_config):
    # check if the right amount of frames is extracted
    frames = vid_cam_config.get_frames()
    assert(len(frames) == vid_cam_config.end_frame - vid_cam_config.start_frame)
    # check if the time difference is well taken from the fps of the video
    assert(np.allclose(np.diff(frames.time.values), [1./vid_cam_config.fps]))


