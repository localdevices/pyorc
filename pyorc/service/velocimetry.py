import os.path
from pyorc import CameraConfig, Video
import cv2
import logging
import yaml
import matplotlib.pyplot as plt

from typing import Optional, Dict

logger = logging.getLogger(__name__)
from dask.diagnostics import ProgressBar
class VelocityFlowProcessor(object):
    """
    General processor class for processing of videos in to velocities and flow and derivative products

    """

    def __init__(
            self,
            recipe: Dict,
            videofile: str,
            cameraconfig: str,
            output: str,
            stat_fn="stat.yml",
            fn_piv="piv.nc",
            fn_piv_mask="piv_mask.nc",
            logger=logger
    ):
        """
        Initialize processor with settings and files

        Parameters
        ----------
        recipe : dict
            YAML recipe, parsed from CLI
        videofile : str
            path to video
        cameraconfig : str
            path to camera config file
        output : str
            path to output file

        """
        self.update = False  # set to True when checks are needed if data already exists or not
        self.recipe = recipe
        self.output = output
        self.fn_piv = os.path.join(self.output, fn_piv)
        self.fn_piv_mask = os.path.join(self.output, fn_piv)
        self.read = True
        self.write = False
        self.fn_video = videofile
        self.fn_cam_config = cameraconfig
        self.logger = logger
        self.set_status_fn(stat_fn)
        self.get_status()
        # TODO: perform checks, minimum steps required
        self.logger.info("pyorc velocimetry processor initialized")
    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, output):
        self._output = output

    @property
    def read(self):
        return self._read

    @read.setter
    def read(self, read):
        self._read = read

    @property
    def write(self):
        return self._write

    @write.setter
    def write(self, write):
        self._write = write

    @property
    def fn_cam_config(self):
        return self._fn_cam_config

    @fn_cam_config.setter
    def fn_cam_config(self, fn_cam_config):
        self._fn_cam_config = fn_cam_config

    @property
    def video_fn(self):
        return self._video_fn

    @video_fn.setter
    def video_fn(self, video_fn):
        self._video_fn = video_fn


    def set_status_fn(self, fn):
        """
        Prepare expected status file, containing filenames and hashes of existing files if these are already processed

        """
        self.status_fn = os.path.join(self.output, fn)


    def get_status(self):
        """
        sets status of project
        Returns
        -------

        """
        if os.path.isfile(self.status_fn):
            # read file and return dict
            with open(self.status_fn, "r") as f:
                body = f.read()
            self.status = yaml.load(body, Loader=yaml.FullLoader)
        else:
            self.status = {}


    def process(self):
        """
        Single method to perform all processing in logical pre-defined order. Also checks for compulsory steps


        Returns
        -------

        """

        # TODO: read in video file with cam configuration

        # TODO: if present, perform frame processing activities
        #
        #  TODO: .project and check if first frame contains data, if not (only black)  return error

        # TODO: .get_piv and check if first frame contains data

        # TODO .get_transect and check if it contains data,

        #  perform any post processing such as plotting or possibly later other analyses


    # def wrapper
    # when status is update, then a wrapper function should check the consistency of all input data for each function
    # block with previous runs. If not consistent then rerun. Do this automatically if data is not present or
    # -y is provided, do with user intervention if stale file is present or -y is not provided

    def update(self):
        """
        Only updates relevant parts based on stats fn hash checks. If dependencies changed according to hash then
        reprocess.

        Returns
        -------

        """
    def video(self):
        # TODO prepare reader
        # self.video = ...
        # some checks ...
        self.logger.info(f"Video successfully read from {self.video_fn} with start frame and end frame") # TODO add start frame and end frame

        # at the end
        # self.da_frames = self.video.get_frames()
        raise NotImplementedError

    def frames(self):
        write = True  # TODO: get this from recipe
        # TODO: preprocess steps

        # at the end
        # self.da_proj = self.da_frames.frames.project()
        # if present: to_video
        # self.piv = self.da_proj.get_piv
        if write:
            delayed_obj = self.piv.to_netcdf(self.fn_piv, compute=False)
            with ProgressBar():
                delayed_obj.compute()

        raise NotImplementedError



    def mask(self):
        # TODO go through several masking groups
        write = True  # TODO: get this from recipe
        piv_masked = self.piv
        for mask_group in self.recipe.mask:
            print("get masks per step")
            # get masks....

            # piv_masked = piv_masked.mask(masks, inplace=True)
        # finally ...
        self.piv_masked = piv_masked
        self.piv_masked.velocimetry.set_encoding()
        # store results to file
        if write:
            delayed_obj = self.piv_masked.to_netcdf(self.piv_masked_fn, compute=False)
            with ProgressBar():
                delayed_obj.compute()


    def plot(self):
        raise NotImplementedError

