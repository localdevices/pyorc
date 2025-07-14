"""pyorc: free and open-source image-based surface velocity and discharge."""

__version__ = "0.8.8"

from .api import CameraConfig, CrossSection, Frames, Transect, Velocimetry, Video, get_camera_config, load_camera_config  # noqa
from .project import *  # noqa
from . import cli, service, sample_data  # noqa

__all__ = [
    "CameraConfig",
    "load_camera_config",
    "get_camera_config",
    "Video",
    "Frames",
    "Velocimetry",
    "Transect",
    "CrossSection",
    "service",
    "cli",
    "sample_data",
]
