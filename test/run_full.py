import openpiv.tools
import openpiv.pyprocess
import xarray as xr
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
from rasterio.plot import reshape_as_raster
import OpenRiverCam as ORC
import cv2
from matplotlib.animation import FuncAnimation, FFMpegWriter

# import example data
from example_data import movie


def proj_frames(movie, dst, prefix="proj"):
    lensParameters = movie["camera_config"]["camera_type"]["lensParameters"]
    lensPosition = movie["camera_config"]["lensPosition"]
    src = os.path.join(movie['file']['bucket'], movie['file']['identifier'])

    n = 0  # frame number
    for _t, img in ORC.io.frames(
            src, start_frame=0, grayscale=True, lens_pars=lensParameters
    ):
        # make a filename
        dst_fn = os.path.join(
            dst, "{:s}_{:04d}_{:06d}.tif".format(prefix, n, int(_t * 1000))
        )
        print(f"Putting frame {n} in file {dst_fn}")
        n += 1  # increase frame number by one
        corr_img, transform = ORC.cv.orthorectification(
            img=img,
            lensPosition=lensPosition,
            h_a=movie["h_a"],
            bbox=movie["camera_config"]["aoi_bbox"],
            resolution=movie["camera_config"]["resolution"],
            **movie["camera_config"]["gcps"],
        )
        if len(corr_img.shape) == 3:
            raster = np.int8(reshape_as_raster(corr_img))
        else:
            raster = np.int8(np.expand_dims(corr_img, axis=0))
        # write to temporary file
        ORC.io.to_geotiff(
            dst_fn,
            raster,
            transform,
            crs=f"EPSG:{movie['camera_config']['site']['crs']}",
            compress="JPEG",
        )


def compute_piv(movie, dst, prefix="proj", piv_kwargs={}):
    """
    compute velocities over frame pairs, choosing frame interval, start / end frame.

    :param movie: dict, contains file dictionary and camera_config
    :param prefix: str, prefix of geotiff files assumed to be present in bucket
    :param piv_kwargs: str, arguments passed to piv algorithm, parameters are defined in docstring of
           openpiv.pyprocess.extended_search_area_piv
    :param logger: logger object
    :return: None
    """
    var_names = ["v_x", "v_y", "s2n", "corr"]
    var_attrs = [
        {
            "standard_name": "sea_water_x_velocity",
            "long_name": "Flow element center velocity vector, x-component",
            "units": "m s-1",
            "coordinates": "lon lat",
        },
        {
            "standard_name": "sea_water_y_velocity",
            "long_name": "Flow element center velocity vector, y-component",
            "units": "m s-1",
            "coordinates": "lon lat",
        },
        {
            "standard_name": "ratio",
            "long_name": "signal to noise ratio",
            "units": "",
            "coordinates": "lon lat",
        },
        {
            "standard_name": "correlation_coefficient",
            "long_name": "correlation coefficient between frames",
            "units": "",
            "coordinates": "lon lat",
        },
    ]
    encoding = {var: {"zlib": True} for var in var_names}
    start_time = datetime.strptime(movie["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    resolution = movie["camera_config"]["resolution"]
    camera_config = movie["camera_config"]
    n = 0
    print(
        f"Computing velocities from projected frames in {movie['file']['bucket']}"
    )
    # open file from bucket in memory
    bucket = movie["file"]["bucket"]
    # get files with the right prefix
    fns = glob.glob(os.path.join(dst, "*.tif"))
    fns.sort()
    frame_b = None
    ms = None
    time, v_x, v_y, s2n, corr = [], [], [], [], []

    for n, fn in enumerate(fns):
        # store previous time offset
        _ms = ms
        # determine time offset of frame from filename
        ms = timedelta(milliseconds=int(fn[-10:-4]))
        frame_a = frame_b
        frame_b = ORC.piv.imread(fn)
        # rewind to beginning of file
        if (frame_a is not None) and (frame_b is not None):
            # we have two frames in memory, now estimate velocity
            print(f"Processing frame {n}")
            # determine time difference dt between frames
            dt = (ms - _ms).total_seconds()
            cols, rows, _v_x, _v_y, _s2n, _corr = ORC.piv.piv(
                frame_a,
                frame_b,
                res_x=resolution,
                res_y=resolution,
                dt=dt,
                search_area_size=movie["camera_config"]["aoi_window_size"],
                **piv_kwargs,
            )
            v_x.append(_v_x), v_y.append(_v_y), s2n.append(_s2n), corr.append(_corr)
            time.append(start_time + ms)
    # finally read GeoTiff transform from the first file
    print(f"Retrieving coordinates of grid from {fn}")
    xs, ys, lons, lats = ORC.io.convert_cols_rows(fns[0], cols, rows)

    # prepare local axes
    spacing_x = np.diff(cols[0])[0]
    spacing_y = np.diff(rows[:, 0])[0]
    x = np.linspace(
        resolution / 2 * spacing_x,
        (len(cols[0]) - 0.5) * resolution * spacing_x,
        len(cols[0]),
    )
    y = np.flipud(
        np.linspace(
            resolution / 2 * spacing_y,
            (len(rows[:, 0]) - 0.5) * resolution * spacing_y,
            len(rows[:, 0]),
        )
    )

    # prepare dataset
    dataset = ORC.io.to_dataset(
        [v_x, v_y, s2n, corr],
        var_names,
        x,
        y,
        time=time,
        lat=lats,
        lon=lons,
        xs=xs,
        ys=ys,
        attrs=var_attrs,
    )
    # write to file and to bucket
    dataset.to_netcdf(os.path.join(dst, "velocity.nc"), encoding=encoding)
    print(f"velocity.nc successfully written in {dst}")

def filter_piv(
    movie, dst, filter_temporal_kwargs={}, filter_spatial_kwargs={}
):
    """
    Filters a PIV velocity dataset (derived with compute_piv) with several temporal and spatial filter. This removes
    noise, isolated velocities in space and time, and moving features that are not likely to be water related.
    Input keyword arguments to the filters can be provided in the request, through several dictionaries.

    :param movie: dict, contains file dictionary and camera_config
    :param filter_temporal_kwargs: dict with the following possible kwargs for temporal filtering
        (+default values if not provided):
        kwargs_angle, dict, containing the following keyword args:
            angle_expected=0.5 * np.pi -- expected angle in radians of flow velocity measured from upwards, clock-wise.
                In OpenRiverCam this is always 0.5*pi because we make sure water flows from left to right
            angle_bounds=0.25 * np.pi -- the maximum angular deviation from expected angle allowed. Velocities that are
                outside this bound are masked
        kwargs_std, dict, containing following keyword args:
            tolerance=1.0 -- maximum allowed std/mean ratio in a pixel. Individual time steps outside this ratio are
                filtered out.
        kwargs_velocity, dict, containing following keyword args:
            s_min=0.1 -- minimum velocity expected to be measured by piv in m/s. lower velocities per timestep are
                filtered out
            s_max=5.0 -- maximum velocity expected to be measured by piv in m/s. higher velocities per timestep are
                filtered out
        kwargs_corr, dict, containing following keyword args:
            corr_min=0.4 -- minimum correlation needed to accept a velocity on timestep basis. Le Coz in FUDAA-LSPIV
                suggest 0.4
        kwargs_neighbour, dict, containing followiung keyword args:
            roll=5 -- amount of time steps in rolling window (centred)
            tolerance=0.5 -- Relative acceptable velocity of maximum found within rolling window

    :param filter_spatial_kwargs: dict with the following possible kwargs for spatial filtering
        (+default values if not provided):
        kwargs_nan, dict, containing following keyword args:
            tolerance=0.8 -- float, amount of NaNs in search window measured as a fraction of total amount of values
                [0-1]
            stride=1 --int, stride used to determine relevant neighbours
        kwargs_median, dict, containing following keyword args:
            tolerance=0.7 -- amount of standard deviations tolerance
            stride=1 -- int, stride used to determine relevant neighbours

    :param logger: logging object
    :return:
    """
    print(f"Filtering surface velocities in {movie['file']['bucket']}")
    # open file from bucket in memory
    fn = os.path.join(dst, "velocity.nc")
    print("applying temporal filters")
    ds = ORC.piv.filter_temporal(fn, filter_corr=True, **filter_temporal_kwargs)
    print("applying spatial filters")
    ds = ORC.piv.filter_spatial(ds, **filter_spatial_kwargs)

    encoding = {var: {"zlib": True} for var in ds}
    # write gridded netCDF with filtered velocities netCDF
    ds.to_netcdf(os.path.join(dst, "velocity_filter.nc"), encoding=encoding)
    print(f"velocity_filter.nc successfully written in {dst}")

def compute_q(
    movie, dst, v_corr=0.85, quantile=[0.05, 0.25, 0.5, 0.75, 0.95],
):
    """
    compute velocities over provided bathymetric cross section points, depth integrated velocities and river flow
    over several quantiles.

    :param movie: dict, contains file dictionary and camera_config
    :param v_corr: float (range: 0-1, typically close to 1), correction factor from surface to depth-average
           (default: 0.85)
    :param quantile: float or list of floats (range: 0-1)  (default: 0.5)
    :return: None
    """
    encoding = {}
    # open S3 bucket
    # open file from bucket in memory
    bucket = movie["file"]["bucket"]
    fn = os.path.join(dst, "velocity_filter.nc")
    print(
        f"Extracting cross section from velocities in {fn}"
    )

    # retrieve velocities over cross section only (ds_points has time, points as dimension)
    ds_points = ORC.io.interp_coords(
        fn, *zip(*movie["bathymetry"]["coords"])
    )

    # add the effective velocity perpendicular to cross-section
    ds_points["v_eff"] = ORC.piv.vector_to_scalar(
        ds_points["v_x"], ds_points["v_y"]
    )

    # integrate over depth with vertical correction
    ds_points["q"] = ORC.piv.depth_integrate(
        ds_points["zcoords"],
        ds_points["v_eff"],
        movie["camera_config"]["gcps"]["z_0"],
        movie["h_a"],
        v_corr=v_corr,
    )

    # integrate over the width of the cross-section
    Q = ORC.piv.integrate_flow(ds_points["q"], quantile=quantile)

    # extract a callback from Q
    Q_dict = {
        "discharge_q{:02d}".format(int(float(q) * 100)): float(Q.sel(quantile=q))
        for q in Q["quantile"]
    }

    # overwrite gridded netCDF with cross section netCDF
    dst_q_depth = os.path.join(dst, "q_depth.nc")
    ds_points.to_netcdf(dst_q_depth, encoding=encoding)
    print(f"Interpolated velocities successfully written to {dst_q_depth}")

    dst_Q = os.path.join(dst, "Q.nc")
    # overwrite gridded netCDF with cross section netCDF
    Q.to_netcdf(dst_Q, encoding=encoding)

    print(f"Discharge successfully written in {dst_Q}")
    return Q_dict


def make_video(movie, dst, video_args):
    def init():
        # im_data = openpiv.tools.imread(fns[0])
        im_data = cv2.imread(fns[0])
        im.set_data(np.zeros(im_data.shape))
        _u = ds["v_x"][0].values
        # line.set_data([], [])
        q.set_UVC(np.zeros(_u.shape), np.zeros(_u.shape))
        return ax  # line,

    def animate(i):
        print(f"Rendering frame {i}")
        im_data = cv2.imread(fns[i + 1])
        # im_data = openpiv.tools.imread(fns[i+1])
        _u, _v = ds["v_x"][i].values, ds["v_y"][i].values
        _u[np.isnan(_u)] = ds["v_x"].median(dim="time").values[np.isnan(_u)]
        _v[np.isnan(_v)] = ds["v_y"].median(dim="time").values[np.isnan(_v)]
        im.set_data(im_data)
        q.set_UVC(_u, _v)

        # point.set_data(x[idx_line], y[idx_line])
        return ax

    # construct string for movie file
    movie_fn = os.path.join(dst, "movie.mp4")
    # list all files
    fns = glob.glob(os.path.join(dst, "*.tif"))
    fns.sort()
    fn_velocity = os.path.join(dst, "velocity_filter.nc")
    ds = xr.open_dataset(fn_velocity)
    f = plt.figure(figsize=(16, 9), frameon=False)
    f.set_size_inches(16, 9, True)
    f.patch.set_facecolor("k")
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.subplot(111)

    im_data = cv2.imread(fns[0])
    im = ax.imshow(im_data)
    x, y = ds["x"].values, ds["y"].values
#    _u, _v = ds["v_x"][0].values, ds["v_y"][0].values
    _u, _v = ds["v_x"].median(axis=0).values, ds["v_y"].median(axis=0).values
    # make a local mesh
    xi, yi = np.meshgrid(x / movie["camera_config"]["resolution"], np.flipud(y) / movie["camera_config"]["resolution"])
    q = ax.quiver(xi, yi, _u, _v, color="r", alpha=0.5, scale=75, width=0.0010)
    plt.savefig(movie_fn.split('.')[0] + ".png", dpi=300, bbox_inches="tight")

    anim = FuncAnimation(
        f, animate, init_func=init, frames=len(fns)-1, interval=20, blit=False
    )  # interval=40 defaults to 25fps (40ms per frame)
    anim.save(movie_fn, **video_args)


folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam"
video_args = {
    "fps": 4,
    "extra_args": ["-vcodec", "libx264"],
    #               'extra_args': ['-pix_fmt', 'yuv420p'],
    "dpi": 120,
}
# example data is loaded in dictionary "movie" with a placeholder for the video to investigate, below we adapt the
# settings.

# change the movie file (all frames are parsed under a subfolder called 'frames')
movie['file']['bucket'] = r'/home/hcwinsemius/Media/projects/OpenRiverCam/tutorial' # s3 bucket (or in our case a local folder) in which the movie is located
movie['file']['identifier'] = 'clip_schedule_20210327_113240.mkv'
movie['h_a'] = 1.25  # this value is simply what you read on the staff gauge

movie['camera_config']['resolution'] = 0.005
movie['camera_config']['aoi_window_size'] = 60

dst = os.path.join(movie['file']['bucket'], 'win_size_60_res_05_mean_gray')

# make destination folder if not existing
if not(os.path.isdir(dst)):
    os.makedirs(dst)


# # extract and project frames
proj_frames(movie, dst)
# # compute piv
compute_piv(movie, dst, piv_kwargs={})

filter_piv(movie, dst, filter_temporal_kwargs={"kwargs_corr": {"tolerance": -100}})
#
compute_q(movie, dst)

make_video(movie, dst, video_args)

