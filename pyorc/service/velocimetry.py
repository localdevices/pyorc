import copy
import functools
import logging
import os.path

import click

import pyorc
import xarray as xr
import yaml
from pyorc.cli import cli_utils

from dask.diagnostics import ProgressBar
from matplotlib.colors import Normalize
from typing import Dict


logger = logging.getLogger(__name__)


def apply_methods(obj, subclass, logger=logger, **kwargs):
    for m, _kwargs in kwargs.items():
        # get the subclass with the expected methods
        cls = getattr(obj, subclass)
        if not(hasattr(cls, m)):
            raise ValueError(f'Method "{m}" for {subclass} does not exist, please check your recipe')
        logger.debug(f"Applying {m} on {subclass} with parameters {_kwargs}")
        meth = getattr(cls, m)
        obj = meth(**_kwargs)

    return obj

def get_masks(obj, **mask_methods):
    masks = []
    for m, _kwargs in mask_methods.items():
        if _kwargs is None:
            # make an empty dict to pass args
            _kwargs = {}
        meth = getattr(obj.velocimetry.mask, m)
        masks.append(meth(**_kwargs))
    return masks



def run_func_hash_io(attrs=[], inputs=[], outputs=[], check=False):
    """
    wrapper function that checks if inputs to function have changed and or output is present or not.
    Runs function if either output is not present or input has changed. If check is False then simply passes everything
    """
    def decorator_func(processor_func):
        @functools.wraps(processor_func)
        def wrapper_func(ref, *args, **kwargs):
            # default, assume running will take place
            run = True
            if check and ref.update:
                for attr, output in zip(attrs, outputs):
                    fn = getattr(ref, output)
                    if os.path.isfile(fn):
                        # TODO: also check if inputs have changed, and if not, run=False
                        run = False
                        break
            if run:
                # apply the wrapped processor function
                processor_func(ref, *args, **kwargs)
            else:
                for attr, output in zip(attrs, outputs):
                    if attr is not None:
                        fn = getattr(ref, output)
                        setattr(ref, attr, xr.open_dataset(fn))
        return wrapper_func
    return decorator_func



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
            update: bool=False,
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
        update : bool, optional
            if set, only update components with changed inputs and configurations

        """
        self.update = update  # set to True when checks are needed if data already exists or not
        self.recipe = recipe
        self.output = output
        self.fn_piv = os.path.join(self.output, fn_piv)
        self.fn_piv_mask = os.path.join(self.output, fn_piv_mask) if "mask" in recipe else self.fn_piv
        self.read = True
        self.write = False
        self.fn_video = videofile
        self.fn_cam_config = cameraconfig
        self.logger = logger
        # self.set_status_fn(stat_fn)
        # self.get_status()
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
    def fn_video(self):
        return self._fn_video

    @fn_video.setter
    def fn_video(self, fn_video):
        self._fn_video = fn_video


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

        self.video(**self.recipe["video"])
        self.frames(**self.recipe["frames"])
        self.velocimetry(**self.recipe["velocimetry"])
        if "mask" in self.recipe:
            self.mask(**self.recipe["mask"])
        else:
            # no masking so use non-masked velocimetry as masked
            self.velocimetry_mask_obj = self.velocimetry_obj
        if "transect" in self.recipe:
            self.transect(**self.recipe["transect"])
        else:
            # no masking so use non-masked velocimetry as masked
            self.velocimetry_mask_obj = self.velocimetry_obj
        self.plot(**self.recipe["plot"])

        # TODO .get_transect and check if it contains data,

        #  perform any post processing such as plotting or possibly later other analyses


    # def wrapper
    # when status is update, then a wrapper function should check the consistency of all input data for each function
    # block with previous runs. If not consistent then rerun. Do this automatically if data is not present or
    # -y is provided, do with user intervention if stale file is present or -y is not provided

    def video(self, **kwargs):
        # TODO prepare reader
        self.video_obj = pyorc.Video(
            self.fn_video,
            camera_config=self.fn_cam_config,
            **kwargs
        )
        # some checks ...
        self.logger.info(f"Video successfully read from {self.fn_video}")
        # at the end
        self.da_frames = self.video_obj.get_frames()
        self.logger.debug(f"{len(self.da_frames)} frames retrieved from video")


    def frames(self, **kwargs):
        # TODO: preprocess steps
        if "project" not in kwargs:
            kwargs["project"] = {}
        # iterate over steps in processing
        self.da_frames = apply_methods(
            self.da_frames,
            "frames",
            logger=self.logger,
            **kwargs
        )
        self.logger.info(f'Frames are preprocessed')


    @run_func_hash_io(attrs=["velocimetry_obj"], check=True, inputs=[], outputs=["fn_piv"])
    def velocimetry(self, method="get_piv", write=False, **kwargs):
        if len(kwargs) > 1:
            raise OverflowError(f"Too many arguments under velocimetry, only one allowed, but {len(kwargs)} given.")
        if len(kwargs) == 0:
            kwargs[method] = {}
        # get velocimetry results
        self.velocimetry_obj = apply_methods(self.da_frames, "frames", logger=self.logger, **kwargs)
        m = list(kwargs.keys())[0]
        parameters = kwargs[m]
        self.logger.info(f"Velocimetry derived with method {m} with parameters {parameters}")
        if write:
            delayed_obj = self.velocimetry_obj.to_netcdf(self.fn_piv, compute=False)
            with ProgressBar():
                delayed_obj.compute()
            self.logger.info(f"Velocimetry written to {self.fn_piv}")
            # Load the velocimetry into memory to prevent re-writes in next steps
            self.velocimetry_obj = xr.open_dataset(self.fn_piv)


    @run_func_hash_io(attrs=["velocimetry_mask_obj"], check=True, inputs=[], outputs=["fn_piv_mask"])
    def mask(self, write=False, **kwargs):
        # TODO go through several masking groups
        self.velocimetry_mask_obj = copy.deepcopy(self.velocimetry_obj)
        for mask_name, mask_grp in kwargs.items():
            self.logger.debug(f"Applying {mask_name}")
            masks = get_masks(self.velocimetry_mask_obj, **mask_grp)
            # apply found masks on velocimetry object
            self.velocimetry_mask_obj.velocimetry.mask(masks, inplace=True)
        self.logger.info(f"Velocimetry masks applied")
        # set the encoding to a good compression level
        self.velocimetry_mask_obj.velocimetry.set_encoding()
        # store results to file
        if write:
            delayed_obj = self.velocimetry_mask_obj.to_netcdf(self.fn_piv_mask, compute=False)
            with ProgressBar():
                delayed_obj.compute()


    def transect(self, write=False, **kwargs):
        for transect_name, transect_grp in kwargs.items():
            if not "shapefile" in kwargs:
                click.UsageError(f'Transect with name "{transect_name}" does not have a "shapefile". Please add "shapefile" in the recipe file')
        # read shapefile
        coords, crs = cli_utils.read_shape(kwargs["shapefile"])
        print(coords)

    def plot(self, **plot_recipes):
        for name, plot_params in plot_recipes.items():
            fn_jpg = os.path.join(self.output, name + ".jpg")
            mode = plot_params["mode"]
            ax = None
            # look for inputs
            if "frames" in plot_params:
                if "frame_number" in plot_params:
                    n = plot_params["frame_number"]
                else:
                    n = 0
                opts = plot_params["frames"] if plot_params["frames"] is not None else {}
                f = self.video_obj.get_frames(method="rgb")[n]
                p = f.frames.plot(ax=ax, mode=mode, **opts)
                # continue with axes of p
                ax = p.axes
            if "velocimetry" in plot_params:
                opts = plot_params["velocimetry"]
                if "vmin" in opts or "vmax" in opts:
                    if "vmin" in opts:
                        vmin = opts["vmin"]
                        del opts["vmin"]
                    else:
                        vmin = None
                    if "vmax" in opts:
                        vmax = opts["vmax"]
                        del opts["vmax"]
                    else:
                        vmax = None
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    opts["norm"] = norm
                reducer = plot_params["reducer"] if "reducer" in plot_params else "mean"
                reducer_params = plot_params["reducer_params"] if "reducer_params" in plot_params else {}
                # TODO: replace by masked obj
                velocimetry_reduced = getattr(
                    self.velocimetry_mask_obj,
                    reducer
                )(
                    dim="time",
                    keep_attrs=True,
                    **reducer_params
                )
                p = velocimetry_reduced.velocimetry.plot(ax=ax, mode=mode, **opts)
                ax = p.axes
            if "transect" in plot_params:
                # TODO add transect plot
                pass
            write_pars = plot_params["write_pars"] if "write_pars" in plot_params else {}
            ax.figure.savefig(fn_jpg, **write_pars)

