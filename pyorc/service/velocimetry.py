import copy
import click
import functools
import logging
import os.path
import subprocess
import xarray as xr
import yaml

from dask.diagnostics import ProgressBar
from matplotlib.colors import Normalize
from typing import Dict

from ..cli import cli_utils
from .. import Video, CameraConfig

__all__ = ["velocity_flow", "velocity_flow_subprocess"]

logger = logging.getLogger(__name__)


def vmin_vmax_to_norm(opts):
    """
    Check if opts contains vmin and/or vmax. If so change that into a norm option which works for all plotting methods

    Parameters
    ----------
    opts : dict
        Dictionary with kwargs to pass to plotting method of pyorc.

    Returns
    -------
    opts : dict
        Dictionary with vmin/vmax replaced by norm if these are provided by user

    """
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
    return opts

def apply_methods(obj, subclass, logger=logger, skip_args=[], **kwargs):
    for m, _kwargs in kwargs.items():
        if m not in skip_args:
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


def run_func_hash_io(
        attrs=[],
        inputs=[],
        configs=[],
        outputs=[],
        write_path=".pyorc",
        check=False,
):
    """
    wrapper function that checks if inputs to function have changed and or output is present or not.
    Runs function if either output is not present or input has changed. If check is False then simply passes everything
    """
    def decorator_func(processor_func):
        @functools.wraps(processor_func)
        def wrapper_func(ref, *args, **kwargs):
            func_name = processor_func.__name__
            # set output path for state files
            path_out = os.path.join(ref.output, write_path)
            if not(os.path.isdir(path_out)):
                os.makedirs(path_out)
            # default, assume running will take place
            run = True
            if check and ref.update:
                run = False  # assume you don't run unless at least one thing changed
                # start with config checks
                fn_recipe = os.path.join(path_out, f"{ref.prefix}{func_name}.yml")
                if not (os.path.isfile(fn_recipe)):
                    run = True
                else:
                    recipe_part = {c: ref.recipe[c] for c in configs if c in ref.recipe}
                    with open(fn_recipe, "r") as f:
                        cfg_ancient = f.read()
                    cfg = yaml.dump(recipe_part, default_flow_style=False, sort_keys=False)
                    if cfg != cfg_ancient:
                        # config has changed
                        ref.logger.debug(f'Configuration of "{func_name}" has changed, requiring rerun')
                        run = True
                if not(run):
                    # if configs are not changed, then go to the file integrity checks
                    for i in inputs + outputs:
                        fn = getattr(ref, i)
                        fn_hash = os.path.join(path_out, f"{os.path.basename(getattr(ref, i))}.hash")
                        if not(os.path.isfile(fn)):
                            run = True
                            break
                        else:
                            # also check if hash file exists
                            if not(os.path.isfile(fn_hash)):
                                run = True
                                break
                            else:
                                hash256 = cli_utils.get_file_hash(fn)
                                with open(fn_hash, "r") as f:
                                    hash256_ancient = f.read()
                                if hash256.hexdigest() != hash256_ancient:
                                    ref.logger.debug(f"File integrity of {fn} has changed, requiring rerun of {func_name}")
                                    run = True
                                    break
            if run:
                # apply the wrapped processor function
                ref.logger.info(
                    f"Running {func_name}")
                processor_func(ref, *args, **kwargs)
                # after run, store configuration file and hashes of in- and outputs
                fn_recipe = os.path.join(path_out, f"{ref.prefix}{func_name}.yml")
                recipe_part = {c: ref.recipe[c] for c in configs if c in ref.recipe}
                with open(fn_recipe, "w") as f:
                    yaml.dump(recipe_part, f, default_flow_style=False, sort_keys=False)
                # after run, store input and output hashes
                for i in inputs + outputs:
                    fn_hash = os.path.join(path_out, f"{os.path.basename(getattr(ref, i))}.hash")
                    # get hash
                    hash256 = cli_utils.get_file_hash(getattr(ref, i))
                    # print(hash256.hexdigest())
                    with open(fn_hash, "w") as f:
                        f.write(hash256.hexdigest())
            else:
                ref.logger.info(f'Configuration, dependencies, input and output files for section "{func_name}" have not changed since last run, skipping...')
                for attr, output in zip(attrs, outputs):
                    if attr is not None:
                        fn = getattr(ref, output)
                        ref.logger.info(f'Results for section "{func_name}" already available, reading from {os.path.abspath(fn)}')
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
            cameraconfig: Dict,
            prefix: str,
            output: str,
            h_a: float=None,
            update: bool=False,
            concurrency=True,
            fn_piv="piv.nc",
            fn_piv_mask="piv_mask.nc",
            fn_transect_template="transect_{:s}.nc",
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
        cameraconfig : dict
            camera config as dict (not yet loaded as CamerConfig object)
        prefix : str
            prefix of produced output files
        output : str
            path to output file
        update : bool, optional
            if set, only update components with changed inputs and configurations
        concurrency : bool, optional
            if set to False, then dask will only run synchronous preventing overuse of memory. This will be slower

        """
        if h_a is not None:
            recipe["video"]["h_a"] = h_a
        # check what projection method is used, use throughout
        self.proj_method = "cv"
        proj = recipe["frames"].get("project")
        if proj:
            if proj.get("method") == "numpy":
                self.proj_method = "numpy"
        self.update = update  # set to True when checks are needed if data already exists or not
        self.recipe = recipe
        self.output = output
        self.concurrency = concurrency
        self.prefix = prefix
        self.fn_piv = os.path.join(self.output, prefix + fn_piv)
        self.fn_piv_mask = os.path.join(self.output, prefix + fn_piv_mask) if "mask" in recipe else self.fn_piv
        self.fn_transect_template = os.path.join(self.output, prefix + fn_transect_template).format if "transect" in recipe else None
        if self.fn_transect_template is not None:
            self.fn_transects = [self.fn_transect_template(t) for t in recipe["transect"]]
        self.read = True
        self.write = False
        self.fn_video = videofile
        self.cam_config = CameraConfig(**cameraconfig)
        self.logger = logger
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

    def process(self):
        """
        Single method to perform all processing in logical pre-defined order. Also checks for compulsory steps


        Returns
        -------

        """
        if not self.concurrency:
            import dask
            # run only synchronous
            dask.config.set(scheduler='synchronous')
        # dask.config.set(pool=Pool(4))
        # dask.config.set(scheduler='processes')
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
        # else:
        #     # no masking so use non-masked velocimetry as masked
        #     self.velocimetry_mask_obj = self.velocimetry_obj
        if "plot" in self.recipe:
            self.plot(**self.recipe["plot"])
        # remove all potentially memory consumptive attributes
        self.da_frames.close()
        self.velocimetry_mask_obj.close()
        self.velocimetry_obj.close()
        delattr(self, "video_obj")
        delattr(self, "velocimetry_obj")
        delattr(self, "velocimetry_mask_obj")
        delattr(self, "da_frames")
        return None
        # TODO .get_transect and check if it contains data,

        #  perform any post processing such as plotting or possibly later other analyses


    # def wrapper
    # when status is update, then a wrapper function should check the consistency of all input data for each function
    # block with previous runs. If not consistent then rerun. Do this automatically if data is not present or
    # -y is provided, do with user intervention if stale file is present or -y is not provided

    def video(self, **kwargs):
        self.video_obj = Video(
            self.fn_video,
            camera_config=self.cam_config,
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
            skip_args=["to_video"],
            **kwargs
        )
        if "to_video" in kwargs:
            kwargs_video = kwargs["to_video"]
            self.logger.info(f"Writing video of processed frames to {kwargs_video['fn']}")
            self.da_frames.frames.to_video(**kwargs_video)
        self.logger.info(f'Frames are preprocessed')


    @run_func_hash_io(attrs=["velocimetry_obj"], check=True, inputs=["fn_video"], configs=["video", "frames", "velocimetry"], outputs=["fn_piv"])
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
            del delayed_obj
            self.logger.info(f"Velocimetry written to {self.fn_piv}")
            # Load the velocimetry into memory to prevent re-writes in next steps
            delattr(self, "velocimetry_obj")
            self.velocimetry_obj = xr.open_dataset(self.fn_piv)


    @run_func_hash_io(attrs=["velocimetry_mask_obj"], check=True, inputs=["fn_piv"], configs=["video", "frames", "velocimetry", "mask"], outputs=["fn_piv_mask"])
    def mask(self, write=False, **kwargs):
        # TODO go through several masking groups
        self.velocimetry_mask_obj = copy.deepcopy(self.velocimetry_obj)
        for mask_name, mask_grp in kwargs.items():
            self.logger.debug(f'Applying "{mask_name}" with parameters {mask_grp}')
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
            del delayed_obj


    @run_func_hash_io(check=False, configs=["transect"], inputs=["fn_piv_mask"])
    def transect(self, write=False, **kwargs):
        self.transects = {}
        # keep integrity of original kwargs
        _kwargs = copy.deepcopy(kwargs)
        for transect_name, transect_grp in _kwargs.items():
            self.logger.debug(f'Processing transect "{transect_name}"')
            # check if there are coordinates provided

            if not ("shapefile" in transect_grp or "geojson" in transect_grp):
                raise click.UsageError(f'Transect with name "{transect_name}" does not have a "shapefile" or '
                                       f'"geojson". Please add "shapefile" in the recipe file')
            # read geojson or shapefile (as alternative
            if "geojson" in transect_grp:
                # read directly from geojson
                coords, crs = cli_utils.read_shape(geojson=transect_grp["geojson"])
            elif "shapefile" in transect_grp:
                coords, crs = cli_utils.read_shape(fn=transect_grp["shapefile"])
            self.logger.debug(f"Coordinates read for transect {transect_name}")
            # check if coords have z coordinates
            if len(coords[0]) == 2:
                raise click.UsageError(
                    f'Transect in {os.path.jabspath(transect_grp["shapefile"])} only contains x, y, but no z-coordinates.'
                )
            x, y, z = zip(*coords)
            self.logger.debug(f"Sampling transect {transect_name}")
            # sample the coordinates
            if not("get_transect" in transect_grp):
                transect_grp["get_transect"] = {}
            if transect_grp["get_transect"] is None:
                transect_grp["get_transect"] = {}
            self.transects[transect_name] = self.velocimetry_mask_obj.velocimetry.get_transect(
                x=x,
                y=y,
                z=z,
                crs=crs,
                **transect_grp["get_transect"]
            )
            if "get_q" in transect_grp:
                if transect_grp["get_q"] is None:
                    transect_grp["get_q"] = {}
                # add q
                self.transects[transect_name] = self.transects[transect_name].transect.get_q(**transect_grp["get_q"])
            if "get_river_flow" in transect_grp:
                if not("get_q" in transect_grp):
                    raise click.UsageError(
                        f'"get_river_flow" found in {transect_name} but no "get_q" found, which is a requirement for "get_river_flow"'
                    )
                if transect_grp["get_river_flow"] is None:
                    transect_grp["get_river_flow"] = {}
                # add q
                self.transects[transect_name].transect.get_river_flow(**transect_grp["get_river_flow"])
            if write:
                # output file
                fn_transect = os.path.abspath(self.fn_transect_template(transect_name))
                self.logger.debug(f'Writing transect "{transect_name}" to {fn_transect}')
                delayed_obj = self.transects[transect_name].to_netcdf(fn_transect, compute=False)
                with ProgressBar():
                    delayed_obj.compute()
                self.logger.info(f'Transect "{transect_name}" written to {fn_transect}')

    @run_func_hash_io(check=False, configs=["video", "frames", "velocimetry", "transect", "plot"], inputs=["fn_video", "fn_piv_mask"], outputs=[])
    def plot(self, **plot_recipes):
        _plot_recipes = copy.deepcopy(plot_recipes)
        for name, plot_params in _plot_recipes.items():
            self.logger.debug(f'Processing plot "{name}"')
            fn_jpg = os.path.join(self.output, self.prefix + name + ".jpg")
            mode = plot_params["mode"]
            ax = None
            # look for inputs
            if "frames" in plot_params:
                if "frame_number" in plot_params:
                    n = plot_params["frame_number"]
                else:
                    n = 0
                opts = plot_params["frames"] if plot_params["frames"] is not None else {}
                f = self.video_obj.get_frames(method="rgb")
                if mode != "camera":
                    f = f.frames.project(method=self.proj_method)[n]
                else:
                    f = f[n]
                p = f.frames.plot(ax=ax, mode=mode, **opts)
                # continue with axes of p
                ax = p.axes
            if "velocimetry" in plot_params:
                opts = plot_params["velocimetry"]
                opts = vmin_vmax_to_norm(opts)
                # select the time reducer. If not defined, choose mean
                reducer = plot_params["reducer"] if "reducer" in plot_params else "mean"
                reducer_params = plot_params["reducer_params"] if "reducer_params" in plot_params else {}
                # reduce the velocimetry over time for plotting purposes
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
                del velocimetry_reduced
            if "transect" in plot_params:
                for transect_name, opts in plot_params["transect"].items():
                    opts = vmin_vmax_to_norm(opts)
                    # read file
                    fn_transect = self.fn_transect_template(transect_name)
                    ds_trans = xr.open_dataset(fn_transect)
                    # default quantile is 2 (50%), otherwise choose from list
                    quantile = 2 if not("quantile") in opts else opts["quantile"]
                    ds_trans_q = ds_trans.isel(quantile=quantile)
                    # add to plot
                    p = ds_trans_q.transect.plot(ax=ax, mode=mode, **opts)
                    ax = p.axes
                    # done with transect, remove from memory
                    ds_trans.close()
                    del ds_trans
            # if mode == "camera":
            #     ax.axis("equal")
            write_pars = plot_params["write_pars"] if "write_pars" in plot_params else {}
            self.logger.debug(f'Writing plot "{name}" to {fn_jpg}')
            ax.figure.savefig(fn_jpg, **write_pars)
            self.logger.info(f'Plot "{name}" written to {fn_jpg}')

def velocity_flow(**kwargs):
    # simply execute the entire process
    processor = VelocityFlowProcessor(**kwargs)
    # process video following the settings
    processor.process()
    # flush the processor itself once done
    del processor


def velocity_flow_subprocess(
    videofile,
    recipe,
    cameraconfig,
    output,
    prefix=None,
    h_a: float = None,
    update: bool = False,
    concurrency=True,
    logger=logging
):
    """
    Writes the requirements to temporary files and runs velocimetry from a separate CLI instance

    Parameters
    ----------
    recipe : dict
        YAML recipe, parsed from CLI
    videofile : str
        path to video
    cameraconfig : dict
        camera config as dict (not yet loaded as CamerConfig object)
    prefix : str
        prefix of produced output files
    output : str
        path to output file
    update : bool, optional
        if set, only update components with changed inputs and configurations
    concurrency : bool, optional
        if set to False, then dask will only run synchronous preventing overuse of memory. This will be slower


    Returns
    -------

    """
    # store recipe in file
    logger.info(f"Launching separate pyorc instance for videofile {videofile}")
    if not(os.path.isdir(output)):
        os.makedirs(output)
    fn_recipe = os.path.join(output, "recipe.yml")
    fn_cam_config = os.path.join(output, "camera_config.json")
    with open(fn_recipe, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
    CameraConfig(**cameraconfig).to_file(fn_cam_config)
    cmd = [
        "pyorc",
        "velocimetry",
        "-V",
        videofile,
        "-c",
        fn_cam_config,
        "-r",
        fn_recipe
    ]
    # add components where needed
    if h_a is not None:
        cmd.append("-h")
        cmd.append(str(h_a))
    if concurrency == False:
        cmd.append("--lowmem")
    if update:
        cmd.append("-u")
    if prefix:
        cmd.append("-p")
        cmd.append(prefix)
    cmd_suffix = [
        "-u",
        "-vvv",
        output
    ]
    cmd = cmd + cmd_suffix
    # call subprocess
    result = subprocess.run(cmd)
    return result
