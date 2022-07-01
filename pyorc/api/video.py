import copy
import cv2
import dask
import dask.array as da
import json

import numpy as np
import os
import warnings
import xarray as xr

from .. import cv, const
from .cameraconfig import load_camera_config, get_camera_config, CameraConfig


class Video:  #(cv2.VideoCapture)
    def __repr__(self):
        template = """
Filename: {:s}
FPS: {:f}
start frame: {:d}
end frame: {:d}
Camera configuration: {:s}
        """.format

        return template(self.fn, self.fps, self.start_frame, self.end_frame, self.camera_config.__repr__() if hasattr(self, "camera_config") else "none")

    def __init__(
            self,
            fn,
            camera_config=None,
            h_a=None,
            start_frame=None,
            end_frame=None,
            *args,
            **kwargs
    ):
        """
        Video class, inheriting parts from cv2.VideoCapture. Contains a camera configuration to it, and a start and end
        frame to read from the video. Several methods read frames into memory or into a xr.DataArray with attributes.
        These can then be processed with other pyorc API functionalities.

        :param fn: str, locally stored video file
        :param camera_config: CameraConfig object, containing all information about the camera, lens parameters, lens
            position, ground control points with GPS coordinates, and all referencing information (see CameraConfig),
            needed to reproject frames on a horizontal geographically referenced plane.
        :param h_a: float, actual height [m], measured in local vertical reference during the video (e.g. a staff gauge
            in view of the camera)
        :param start_frame: int, first frame to use in analysis
        :param end_frame: int, last frame to use in analysis
        :param args: list or tuple, arguments to pass to cv2.VideoCapture on initialization.
        :param kwargs: dict, keyword arguments to pass to cv2.VideoCapture on initialization.
        """
        assert(isinstance(start_frame, (int, type(None)))), 'start_frame must be of type "int"'
        assert(isinstance(end_frame, (int, type(None)))), 'end_frame must be of type "int"'
        if camera_config is not None:
            # check if h_a is supplied, if so, then also z_0 and h_ref must be available
            if h_a is not None:
                assert(isinstance(camera_config.gcps["z_0"], float)),\
                    "h_a was supplied, but camera config's gcps do not contain z_0, this is needed for dynamic " \
                    "reprojection. You can supplying z_0 and h_ref in the camera_config's gcps upon making a camera " \
                    "configuration. "
                assert (isinstance(camera_config.gcps["h_ref"], float)),\
                    "h_a was supplied, but camera config's gcps do not contain h_ref, this is needed for dynamic " \
                    "reprojection. You can supplying z_0 and h_ref in the camera_config's gcps upon making a camera " \
                    "configuration. "
        # super().__init__(*args, **kwargs)
        cap = cv2.VideoCapture(fn)
        # explicitly open file for reading
        # self.open(fn)
        # set end and start frame
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame is not None:
            if (start_frame > self.frame_count and self.frame_count > 0):
                raise ValueError("Start frame is larger than total amount of frames")
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame < start_frame:
                raise ValueError(
                    f"Start frame {start_frame} is larger than end frame {end_frame}"
                )
        else:
            end_frame = self.frame_count
        self.end_frame = end_frame
        self.start_frame = start_frame
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_number = 0
        # set other properties
        # if h_a is not None:
        self.h_a = h_a
        # make camera config part of the vidoe object
        if camera_config is not None:
            self.camera_config = camera_config
        self.fn = fn
        self._stills = {}  # here all stills are stored lazily
        # nothing to be done at this stage, release file for now.
        # self.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        del cap

    @property
    def camera_config(self):
        """

        :return: CameraConfig object
        """
        return self._camera_config

    @camera_config.setter
    def camera_config(self, camera_config_input):
        """
        Set camera config as a serializable object from either a filename, json string or a dict
        :param camera_config_input: str, dict, CameraConfig object, filename string, or json string containing camera
            configuration.
        """
        try:
            if isinstance(camera_config_input, str):
                if os.path.isfile(camera_config_input):
                    # assume string is a file
                    self._camera_config = load_camera_config(camera_config_input)
                else:  # Try to read CameraConfig from string
                    self._camera_config = get_camera_config(camera_config_input)
            elif isinstance(camera_config_input, CameraConfig):
                # set CameraConfig as is
                self._camera_config = camera_config_input
            elif isinstance(camera_config_input, dict):
                # Create CameraConfig from dict
                self._camera_config = CameraConfig(**camera_config_input)
        except:
            raise IOError("Could not recognise input as a CameraConfig file, string, dictionary or CameraConfig object.")

    @property
    def end_frame(self):
        """

        :return: int, last frame considered in analysis
        """
        return self._end_frame

    @end_frame.setter
    def end_frame(self, end_frame=None):
        if end_frame is None:
            self._end_frame = self.frame_count
        else:
            self._end_frame = min(self.frame_count, end_frame)

    @property
    def h_a(self):
        """

        :return: Actual water level [m] during video
        """
        return self._h_a

    @h_a.setter
    def h_a(self, h_a):
        if h_a is not None:
            assert(isinstance(h_a, float)), f"The actual water level must be a float, you supplied a {type(h_a)}"
            if h_a > 10:
                warnings.warn(f"Your water level is {h_a} meters, which is very high for a locally referenced reading. Make sure this value is correct and expressed in unit meters.")
            if h_a < 0:
                warnings.warn("Water level is negative. This can be correct, but may be unlikely, especially if you use a staff gauge.")
        self._h_a = h_a

    @property
    def start_frame(self):
        """

        :return: int, first frame considered in analysis
        """
        return self._start_frame

    @start_frame.setter
    def start_frame(self, start_frame=None):
        if start_frame is None:
            self._start_frame = 0
        else:
            self._start_frame = start_frame

    @property
    def fps(self):
        """

        :return: float, frames per second
        """
        return self._fps

    @fps.setter
    def fps(self, fps):
        if (np.isinf(fps)) or (fps <= 0):
            raise ValueError(f"FPS in video is {fps} which is not a valid value. Repair the video file before use")
        self._fps = fps

    @property
    def corners(self):
        """

        :return: list of 4 lists (int) with [column, row] locations of area of interest in video objective
        """
        return self._corners

    @corners.setter
    def corners(self, corners):
        self._corners = corners

    def get_frame(self, n, method="grayscale", lens_corr=False):
        """
        Retrieve one frame. Frame will be corrected for lens distortion if lens parameters are given.

        :param n: int, frame number to retrieve
        :param method: str, can be "rgb", "grayscale", or "hsv", default: "grayscale"
        :param lens_corr: bool, optional, if set to True, lens parameters will be used to undistort image
        :return: np.ndarray containing frame
        """
        assert(n >= 0), "frame number cannot be negative"
        assert(n <= self.end_frame - self.start_frame), "frame number is larger than the different between the start and end frame"
        assert(method in ["grayscale", "rgb", "hsv"]), f'method must be "grayscale", "rgb" or "hsv", method is "{method}"'
        cap = cv2.VideoCapture(self.fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n + self.start_frame)
        try:
            ret, img = cap.read()
        except:
            raise IOError(f"Cannot read")
        if ret:
            if lens_corr:
                if self.camera_config.lens_pars is not None:
                    # apply lens distortion correction
                    img = cv.undistort_img(img, **self.camera_config.lens_pars)
            if method == "grayscale":
                # apply gray scaling, contrast- and gamma correction
                # img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #  mean(axis=2)
            elif method == "rgb":
                # turn bgr to rgb for plotting purposes
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif method == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.frame_count = n + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        return img

    def get_frames(self, **kwargs):
        """
        Get a xr.DataArray, containing a dask array of frames, from `start_frame` until `end_frame`, expected to be read
        lazily. The xr.DataArray will contain all coordinate variables and attributes, needed for further processing
        steps.

        :param kwargs: dict, keyword arguments to pass to `get_frame`. Currently only `grayscale` is supported.
        :return: xr.DataArray, containing all requested frames
        """
        assert(hasattr(self, "_camera_config")), "No camera configuration is set, add it to the video using the .camera_config method"
        # assert(hasattr(self, "_h_a")), 'Water level with this video has not been set, please set it with the .h_a method'
        # camera_config may be altered for the frames object, so copy below
        camera_config = copy.deepcopy(self.camera_config)
        get_frame = dask.delayed(self.get_frame, pure=True)  # Lazy version of get_frame
        if not("lens_corr" in kwargs):
            # if not explicitly set by user, check if lens pars are available, and if so, add lens_corr to kwargs
            if hasattr(self.camera_config, "lens_pars"):
                kwargs["lens_corr"] = True
        frames = [get_frame(n=n, **kwargs) for n in range(self.end_frame - self.start_frame)]
        sample = frames[0].compute()
        data_array = [da.from_delayed(
            frame,
            dtype=sample.dtype,
            shape=sample.shape
        ) for frame in frames]
        if "lens_corr" in kwargs:
            if kwargs["lens_corr"]:
                # also correct the control point src
                camera_config.gcps["src"] = cv.undistort_points(
                    camera_config.gcps["src"],
                    sample.shape[0],
                    sample.shape[1],
                    **self.camera_config.lens_pars
                )
                camera_config.corners = cv.undistort_points(
                    camera_config.corners,
                    sample.shape[0],
                    sample.shape[1],
                    **self.camera_config.lens_pars
                )
        time = np.arange(len(data_array))*1/self.fps
        # y needs to be flipped up down to match the order of rows followed by coordinate systems (bottom to top)
        y = np.flipud(np.arange(data_array[0].shape[0]))
        x = np.arange(data_array[0].shape[1])
        # perspective column and row coordinate grids
        xp, yp = np.meshgrid(x, y)
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        if len(sample.shape) == 3:
            coords["rgb"] = np.array([0, 1, 2])
        # make DataArray dimensions and attributes
        dims = tuple(coords.keys())
        attrs = {
            "camera_shape": str([len(y), len(x)]),
            "camera_config": camera_config.to_json(),
            "h_a": json.dumps(self.h_a)
        }
        frames = xr.DataArray(
            da.stack(data_array, axis=0),
            dims=dims,
            coords=coords,
            attrs=attrs
        )
        del coords["time"]
        if len(sample.shape) == 3:
            del coords["rgb"]
        # add coordinate grids (i.e. without time)
        frames = frames.frames._add_xy_coords([xp, yp], coords, const.PERSPECTIVE_ATTRS)
        frames.name = "frames"
        return frames

