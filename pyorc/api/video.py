"""Video module for pyorc."""

import copy
import json
import os
import warnings
from typing import List, Literal, Optional, Union

import cv2
import dask
import dask.array as da
import numpy as np
import xarray as xr

from pyorc import const, cv, helpers

from .cameraconfig import CameraConfig, get_camera_config, load_camera_config


class Video:  # (cv2.VideoCapture)
    """Video class for reading and extracting data from video files."""

    def __repr__(self):
        template = """
Filename: {:s}
FPS: {:f}
start frame: {:d}
end frame: {:d}
Camera configuration: {:s}
        """.format

        return template(
            self.fn,
            self.fps,
            self.start_frame,
            self.end_frame,
            self.camera_config.__repr__() if hasattr(self, "camera_config") else "none",
        )

    def __init__(
        self,
        fn: str,
        camera_config: Optional[Union[str, CameraConfig]] = None,
        h_a: Optional[float] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        freq: int = 1,
        chunksize: int = 20,
        stabilize: Optional[List[List]] = None,
        lazy: bool = True,
        rotation: Optional[int] = None,
        fps: Optional[float] = None,
        progress: bool = True,
    ):
        """Video class, inheriting parts from cv2.VideoCapture.

        Contains a camera configuration to it, and a start and end frame to read from the video. Several methods read
        frames into memory or into a xr.DataArray with attributes. These can then be processed with other pyorc API
        functionalities.

        Parameters
        ----------
        fn : str
            Locally stored video file
        camera_config : pyorc.CameraConfig, optional
            contains all information about the camera, lens parameters, lens position, ground control points with GPS
            coordinates, and all referencing information (see CameraConfig), needed to reproject frames on a horizontal
            geographically referenced plane.
        h_a : float, optional
            actual height [m], measured in local vertical reference during the video (e.g. a staff gauge in view of
            the camera)
        start_frame : int, optional
            first frame to use in analysis (default: 0)
        end_frame : int, optional
            last frame to use in analysis (if not set, last frame available in video will be used)
        freq : int, optional
            Frequency to read frames with. Default is 1, if set to e.g. 2 only each 2nd frame will be read.
        chunksize : int, optional
            Amount of frames to read in one pass, defaults to 20
        stabilize : list of lists, optional
            set of coordinates, that together encapsulate the polygon that defines the mask, separating land from water.
            The mask is used to select region (on land) for rigid point search for stabilization. If not set, then no
            stabilization will be performed
        lazy : bool, optional
            By default set to True, making frames read in organised chunks. If set to False, video is read in memory
            entirely. Likely, we will deprecate this option in the future as we plan to optimize chunked reading.
            Currently if memory size allows, setting lazy to False results in faster processing.
        rotation : int, optional
            can be 0, 90, 180, 270. If provided, images will be forced to rotate along the provided angle.
        fps : float, optional
            hard set for frames per second. Use this with utmost caution and only when you are confident that the video
            metadata is incorrect.
        progress : bool, optional
            Display progress bar while reading video. Default is True.

        """
        assert isinstance(start_frame, (int, type(None))), 'start_frame must be of type "int"'
        assert isinstance(end_frame, (int, type(None))), 'end_frame must be of type "int"'
        self.feats_stats = None
        self.feats_errs = None
        self.ms = None
        self.mask = None
        self.lazy = lazy
        self.progress = progress
        self.stabilize = stabilize
        if camera_config is not None:
            self.camera_config = camera_config
            # if camera_config is not None:
            # check if h_a is supplied, if so, then also z_0 and h_ref must be available
            if h_a is not None:
                assert isinstance(self.camera_config.gcps["z_0"], float), (
                    "h_a was supplied, but camera config's gcps do not contain z_0, this is needed for dynamic "
                    "reprojection. You can supplying z_0 and h_ref in the camera_config's gcps upon making a camera "
                    "configuration. "
                )
                assert isinstance(self.camera_config.gcps["h_ref"], float), (
                    "h_a was supplied, but camera config's gcps do not contain h_ref, this is needed for dynamic "
                    "reprojection. You must supply z_0 and h_ref in the camera_config's gcps upon making a camera "
                    "configuration. "
                )
                # check if h_a is not very far from h_ref
                if np.abs(h_a - self.camera_config.gcps["h_ref"]) > const.WATER_LEVEL_MAX_DIFF:
                    warnings.warn(
                        f"h_a is more than {const.WATER_LEVEL_MAX_DIFF} meters different from h_ref. You may have "
                        "made a mistake in the h_a definition. Check for instance if your h_a is measured with the same"
                        "datum as h_ref. ",
                        stacklevel=2,
                    )
        if not os.path.isfile(fn):
            raise IOError(
                f"Video file {fn} does not exist. If you are on windows and using \\ as file "
                f"separator ensure that the file string is prepended with an r, e.g. "
                f"r'c:\\video_files\\filename.mp4'"
            )

        cap = cv2.VideoCapture(fn)
        # Force to automatically rotate frames
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # explicitly open file for reading
        if self.stabilize is not None:
            # set a gridded mask based on the roi points
            self.set_mask_from_exterior(self.stabilize)
        # set end and start frame
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if start_frame is not None:
            if start_frame > self.frame_count > 0:
                raise ValueError("Start frame is larger than total amount of frames")
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame < start_frame:
                raise ValueError(f"Start frame {start_frame} is larger than end frame {end_frame}")
            # end frame cannot be larger than total amount of available frames
            end_frame = np.minimum(end_frame, self.frame_count)
        else:
            end_frame = self.frame_count
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.rotation = rotation
        # extract times, frame numbers and frames as far as available
        time, frame_number, frames = cv.get_time_frames(
            cap, start_frame, end_frame, lazy=lazy, rotation=self.rotation, method="bgr", fps=fps, progress=progress
        )
        self.frames = frames
        # check if end_frame changed
        if len(frame_number) > 0:
            if frame_number[-1] != end_frame:
                warnings.warn(
                    f"End frame {end_frame} cannot be read from file. End frame is adapted to {frame_number[-1]}",
                    stacklevel=2,
                )
                end_frame = frame_number[-1]
        # apparently we are reading an image instead of a video

        self.end_frame = end_frame
        self.freq = freq
        self.chunksize = chunksize
        self.time = time
        self.frame_number = frame_number
        self.start_frame = start_frame
        if self.stabilize is not None:
            # select the right recipe dependent on the movie being fixed or moving
            self.get_ms(cap)
        if fps is None:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = fps
        # set other properties
        self.h_a = h_a
        # make camera config part of the video object
        self.fn = fn
        self._stills = {}  # here all stills are stored lazily
        # nothing to be done at this stage, release file for now.
        cap.release()
        del cap

    @property
    def lazy(self):
        """Get lazy flag."""
        return self._lazy

    @lazy.setter
    def lazy(self, lazy):
        self._lazy = lazy

    @property
    def mask(self):
        """Get region mask for stabilization."""
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        if mask is None:
            self._mask = None
        else:
            self._mask = mask

    @property
    def camera_config(self):
        """Get `CameraConfig` object attached to `Video` instance."""
        if hasattr(self, "_camera_config"):
            return self._camera_config
        else:
            return None

    @camera_config.setter
    def camera_config(self, camera_config_input: Union[str, dict]):
        """Set camera config as a serializable object from either a filename, json string or a dict."""
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
        except IOError:
            raise IOError(
                "Could not recognise input as a CameraConfig file, string, dictionary or CameraConfig object."
            )

    @property
    def end_frame(self):
        """:return: int, last frame considered in analysis"""
        return self._end_frame

    @end_frame.setter
    def end_frame(self, end_frame: Optional[int] = None):
        # sometimes last frames are not read by OpenCV, hence we skip the last frame always
        if end_frame is None:
            self._end_frame = self.frame_count - 1
        else:
            self._end_frame = end_frame

    @property
    def freq(self):
        """Get video sampling frequency."""
        return self._freq

    @freq.setter
    def freq(self, freq=1):
        self._freq = freq

    @property
    def progress(self):
        """Get progress property."""
        return self._progress

    @progress.setter
    def progress(self, progress=True):
        self._progress = progress

    @property
    def stabilize(self):
        """Get stabilization region coordinates."""
        if self._stabilize is not None:
            return self._stabilize
        elif hasattr(self, "camera_config"):
            if hasattr(self.camera_config, "stabilize"):
                return self.camera_config.stabilize

    @stabilize.setter
    def stabilize(self, coords: Optional[List[List]] = None):
        self._stabilize = coords

    @property
    def h_a(self):
        """:return: Actual water level [m] during video"""
        return self._h_a

    @h_a.setter
    def h_a(self, h_a: float):
        if h_a is not None:
            assert isinstance(h_a, float), f"The actual water level must be a float, you supplied a {type(h_a)}"
            if h_a < 0:
                warnings.warn(
                    "Water level is negative. This can be correct, but may be unlikely, especially if you use a "
                    "staff gauge.",
                    stacklevel=2,
                )
        self._h_a = h_a

    @property
    def start_frame(self):
        """:return: int, first frame considered in analysis"""
        return self._start_frame

    @start_frame.setter
    def start_frame(self, start_frame: Optional[int] = None):
        if start_frame is None:
            self._start_frame = 0
        else:
            self._start_frame = start_frame

    @property
    def frames(self):
        """Get frames of Video instance."""
        return self._frames

    @frames.setter
    def frames(self, frames: Optional[List] = None):
        self._frames = frames

    @property
    def fps(self):
        """:return: float, frames per second"""
        return self._fps

    @fps.setter
    def fps(self, fps: float):
        if (np.isinf(fps)) or (fps <= 0):
            raise ValueError(f"FPS in video is {fps} which is not a valid value. Repair the video file before use")
        self._fps = fps

    @property
    def corners(self):
        """:return: list of 4 lists (int) with [column, row] locations of area of interest in video objective"""
        return self._corners

    @corners.setter
    def corners(self, corners: List[List]):
        self._corners = corners

    @property
    def rotation(self):
        """Get rotation code."""
        if self._rotation is not None:
            return self._rotation
        elif hasattr(self, "camera_config"):
            if hasattr(self.camera_config, "rotation"):
                return helpers.get_rotation_code(self.camera_config.rotation)

    @rotation.setter
    def rotation(self, rotation_code: int):
        """Set rotation from integer in the form of OpenCV rotation codes."""
        self._rotation = helpers.get_rotation_code(rotation_code)

    def get_frame(
        self, n: int, method: Optional[Literal["grayscale", "rgb", "hsv", "bgr"]] = "grayscale"
    ) -> np.ndarray:
        """Retrieve one frame.

        Parameters
        ----------
        n : int
            frame number to retrieve
        method : str
            can be "rgb", "grayscale", or "hsv", default: "grayscale"

        Returns
        -------
        frame : np.ndarray
            2d array (grayscale) or 3d (rgb/hsv) with frame

        """
        assert n >= 0, "frame number cannot be negative"
        assert (
            n - self.start_frame <= self.end_frame - self.start_frame
        ), "frame number is larger than the difference between the start and end frame "
        # assert (method in ["grayscale", "rgb",
        #                    "hsv"]), f'method must be "grayscale", "rgb" or "hsv", method is "{method}"'
        if not os.path.isfile(self.fn):
            raise IOError(f"Video file {self.fn} does not exist.")
        cap = cv2.VideoCapture(self.fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n + self.start_frame)
        ret, img = cv.get_frame(cap, rotation=self.rotation, ms=self.ms[n] if self.ms else None, method=method)
        self.frame_count = n + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        return img

    def get_frames_chunk(
        self, n_start: int, n_end: int, method: Optional[Literal["grayscale", "rgb", "hsv", "bgr"]] = "grayscale"
    ) -> np.ndarray:
        """Retrieve a chunk of frames in one go.

        Parameters
        ----------
        n_start : int
            frame number to initiate retrieval
        n_end : int
            last frame number of retrieval
        method : str
            can be "rgb", "grayscale", or "hsv", default: "grayscale"

        Returns
        -------
        frame : np.ndarray
            2d array (grayscale) or 3d (rgb/hsv) with frame

        """
        assert n_start >= 0, "frame number cannot be negative"
        assert (
            n_start - self.start_frame <= self.end_frame - self.start_frame
        ), "frame number is larger than the difference between the start and end frame "
        # assert (method in ["grayscale", "rgb",
        #                    "hsv"]), f'method must be "grayscale", "rgb" or "hsv", method is "{method}"'
        if not os.path.isfile(self.fn):
            raise IOError(f"Video file {self.fn} does not exist.")
        cap = cv2.VideoCapture(self.fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_start + self.start_frame)
        imgs = []
        for n in range(n_start, n_end):
            ret, img = cv.get_frame(cap, rotation=self.rotation, ms=self.ms[n] if self.ms else None, method=method)
            self.frame_count = n + 1
            imgs.append(img)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
        return np.array(imgs)

    def get_frames(self, method: Optional[Literal["grayscale", "rgb", "hsv", "bgr"]] = "grayscale") -> xr.DataArray:
        """Get a xr.DataArray, containing a dask array of frames, from `start_frame` until `end_frame`.

        The xr.DataArray will contain all coordinate variables and attributes, needed for further processing. It
        may be lazy, so that frames can be read at a later stage.

        Parameters
        ----------
        method: str, optional
            method for color scaling, can be "grayscale" (default) or "rgb"

        Returns
        -------
        frames : xr.DataArray
            containing all requested frames

        """
        assert hasattr(
            self, "_camera_config"
        ), "No camera configuration is set, add it to the video using the .camera_config method"
        # camera_config may be altered for the frames object, so copy below
        camera_config = copy.deepcopy(self.camera_config)
        # frames_chunk = 20
        if self.frames is None:
            # a specific method for collecting frames is requested or lazy access is requested.
            # get_frame = dask.delayed(self.get_frame, pure=True)  # Lazy version of get_frame
            get_frames_chunk = dask.delayed(self.get_frames_chunk, pure=True)  # Lazy version of get_frame
            # get all listed frames
            # frames = [get_frame(n=n, method=method) for n, f_number in enumerate(self.frame_number)]
            # derive video shape
            sample = get_frames_chunk(n_start=0, n_end=1, method=method).compute()[0]
            data_array = []
            for n_start in range(0, len(self.frame_number), self.chunksize):
                n_end = np.minimum(n_start + self.chunksize, len(self.frame_number))
                frame_chunk = get_frames_chunk(n_start=n_start, n_end=n_end, method=method)
                shape = (n_end - n_start, *sample.shape)
                data_array.append(da.from_delayed(frame_chunk, dtype=sample.dtype, shape=shape))

            da_stack = da.concatenate(data_array, axis=0)
        else:
            da_stack = self.frames
            # apply stabilisation
            if self.ms is not None:
                # da_stack = np.array([cv.transform(img, m) for img, m in zip(da_stack, self.ms)])
                da_stack = np.array([cv.transform(cv.color_scale(img, method), m) for img, m in zip(da_stack, self.ms)])
            else:
                # only color transform
                da_stack = np.array([cv.color_scale(img, method) for img in da_stack])
            sample = da_stack[0]

        time = (
            np.array(self.time) * 0.001
        )  # measure in seconds to comply with CF conventions # np.arange(len(data_array))*1/self.fps
        # y needs to be flipped up down to match the order of rows followed by coordinate systems (bottom to top)
        y = np.flipud(np.arange(sample.shape[0]))
        x = np.arange(sample.shape[1])
        # perspective column and row coordinate grids
        xp, yp = np.meshgrid(x, y)
        coords = {"time": time, "y": y, "x": x}
        if len(sample.shape) == 3:
            coords["rgb"] = np.array([0, 1, 2])
        # make DataArray dimensions and attributes
        dims = tuple(coords.keys())
        attrs = {
            "camera_shape": str([len(y), len(x)]),
            "camera_config": camera_config.to_json(),
            "h_a": json.dumps(self.h_a),
            "chunksize": self.chunksize,
        }
        frames = xr.DataArray(
            da_stack,
            dims=dims,
            coords=coords,
            attrs=attrs,
        )[:: self.freq]
        frames = frames.chunk({"time": 20})  # set chunks over time dimension
        del coords["time"]
        if len(sample.shape) == 3:
            del coords["rgb"]
        # add coordinate grids (i.e. without time)
        frames = frames.frames.add_xy_coords({"xp": xp, "yp": yp}, coords, const.PERSPECTIVE_ATTRS)
        frames.name = "frames"
        return frames

    def set_mask_from_exterior(self, exterior: List[List]):
        """Prepare a mask grid with 255 outside of the stabilization polygon and 0 inside.

        Parameters
        ----------
        exterior : list of lists
            coordinates defining the polygon for masking

        Returns
        -------
        self.mask : np.ndarray
            mask for stabilization region

        """
        mask_coords = np.array([exterior], dtype=np.int32)
        mask = np.zeros((self.height, self.width), np.uint8)
        mask = cv2.fillPoly(mask, [mask_coords], 255)
        mask[mask == 0] = 1
        mask[mask == 255] = 0
        mask[mask == 1] = 255
        self.mask = mask

    def get_ms(self, cap: cv2.VideoCapture, split: Optional[int] = 2):
        """Get stabilization transforms for each frame based on analysis of stable points outside of water area.

        Parameters
        ----------
        cap : cv2.VideoCapture
            ref to video file

        split : int
            Amount of quadrants to split the area in to search for stabilization points. In each quadrant, an equal
            number of stabilization points will be searched for .This is in order to ensure stabilization uses points
            in all quadrants and prevent overfitting on only one part of the objective.

        """
        self.ms = cv.get_ms_gftt(
            cap,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            split=split,
            mask=self.mask,
            progress=self.progress,
        )
