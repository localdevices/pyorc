from __future__ import annotations

import cv2
import dask
import numpy as np
import xarray as xr

from rasterio.features import rasterize
from typing import Optional, Any
from flox.xarray import xarray_reduce


from . import helpers, cv

# from . import CameraConfig

__all__ = ["project_numpy", "project_cv"]


def project_cv(
    da: xr.DataArray,
    cc: CameraConfig,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
):
    """
    Projection method that uses pure OpenCV. Reprojection is done in two steps: undistortion and reprojection.
    This method gives incorrect mapping in case of very strong distortion and/or where part of the area of interest is
    outside of the field of view. In these cases, it is strongly recommended to use `project_numpy` instead.

    Parameters
    ----------
    da : xr.DataArray
        Frames time series
    cc : pyorc.CameraConfig
    x : np.ndarray
        x-axis
    y : np.ndarray
        y-axis
    z : float
        vertical level value in real-world coordinates

    Returns
    -------
    da_proj : xr.DataArray
        Reprojected Frames

    """
    da_undistort = xr.apply_ufunc(
        cv.undistort_img,
        da,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        output_dtypes=da.dtype,
        kwargs={
            "camera_matrix": cc.camera_matrix,
            "dist_coeffs": cc.dist_coeffs
        },
        dask='parallelized',
        keep_attrs=True,
        vectorize=True
    )  # .rename({
    # also undistort the src control points
    cc.gcps["src"] = cv.undistort_points(
        cc.gcps["src"],
        cc.camera_matrix,
        cc.dist_coeffs
    )
    h_a = cc.z_to_h(z)
    src = cc.get_bbox(
        camera=True,
        h_a=h_a,
        expand_exterior=False
    ).exterior.coords[0:4]
    dst_xy = cc.get_bbox(expand_exterior=False).exterior.coords[0:4]
    # get geographic coordinates bbox corners
    dst = cv.transform_to_bbox(
        dst_xy,
        cc.bbox,
        cc.resolution
    )
    M = cv.get_M_2D(src, dst)
    da_proj = xr.apply_ufunc(
        cv.get_ortho, da_undistort,
        kwargs={
            "M": M,
            "shape": tuple(np.flipud(cc.shape)),
            "flags": cv2.INTER_AREA
        },
        input_core_dims=[["y", "x"]],
        output_core_dims=[["new_y", "new_x"]],
        dask_gufunc_kwargs={
            "output_sizes": {
                "new_y": len(y),
                "new_x": len(x)
            },
        },
        output_dtypes=[da.dtype],
        vectorize=True,
        exclude_dims=set(("y", "x")),
        dask="parallelized",
        keep_attrs=True
    ).rename({
        "new_y": "y",
        "new_x": "x"
    })
    da_proj["y"] = y
    da_proj["x"] = x
    return da_proj


def project_numpy(
    da: xr.DataArray,
    cc: CameraConfig,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    reducer: Optional[str] = "mean",
):
    """
    Project from FOV pixels directly to target grid, including undistortion and projection.

    Parameters
    ----------
    da : xr.DataArray
        Frames time series
    cc : pyorc.CameraConfig
    x : np.ndarray
        x-axis
    y : np.ndarray
        y-axis
    z : float
        vertical level value in real-world coordinates
    reducer : str, optional
        If set to a valid reducer (like mean, median, max) oversampled target pixels will be reduced by using the set
        reducer. Oversampled target pixels are defined as pixels that have more than one pixels in the original
        Field of View that fit within that pixel. All other pixels are defined with nearest-neighbour. Default is
        "nearest" for nearest neighbour.

    Returns
    -------
    da_proj : xr.DataArray
        Reprojected Frames

    """
    # create coordinate system for target grid
    coords = {
        "time": da.time,
        "y": y,
        "x": x
    }
    if "rgb" in da.coords:
        chunks = (1, None, None, None)
        coords["rgb"] = da.rgb
        shape = (
            len(da),
            len(y),
            len(x),
            3
        )
    else:
        chunks = (1, None, None)
        shape = (
            len(da),
            len(y),
            len(x),
        )
    da_new = xr.DataArray(
        dask.array.zeros(
            shape,
            chunks=chunks,
            dtype=da.dtype
        ) * np.nan,
        coords=coords,
        name="project_frames",
        attrs=da.attrs
    ).stack(group=("y", "x"))  # stack to one dimension for all pixels
    #
    cols, rows = np.meshgrid(np.arange(len(x)), np.arange(len(y)))
    # make a large list of coordinates of target grid.
    xs, ys = helpers.pixel_to_map(cols.flatten(), rows.flatten(), cc.transform)
    # back-project real-world coordinates to camera coordinates
    points_cam = cc.project_points(
        list(zip(xs, ys, np.ones(len(xs)) * z))
    )
    # round cam coordinates to pixels
    points_cam = np.int64(np.round(points_cam))
    # find locations that lie within the camera objective, rest should remain missing value
    idx_in = np.all(
        [
            points_cam[:, 0] > 0,
            points_cam[:, 0] < len(da.x),
            points_cam[:, 1] > 0,
            points_cam[:, 1] < len(da.y),

        ],
        axis=0
    )
    # coerce 2D idxs to 1D idxs
    idx_back = np.array(points_cam[idx_in, 1]) * len(da.x) + np.array(points_cam[idx_in, 0])
    vals = da.stack(group=("y", "x")).isel(group=idx_back)
    # overwrite the values group coordinates
    vals = vals.drop_vars(['group', 'y', 'x'])
    vals["group"] = da_new.group[idx_in]
    da_new[..., idx_in] = vals

    if reducer != "nearest":
        # also fill in the parts that have valid averaged pixels
        coli, rowi = np.meshgrid(
            np.arange(len(da.x)),
            np.arange(len(da.y))
        )
        poly = cc.get_bbox(camera=True, z_a=z)
        mask = xr.DataArray(
            rasterize([poly], out_shape=(cc.height, cc.width)) == 1,
            coords={"y": da.y, "x": da.x},
            name="mask",
            # attrs=da.attrs
        )
        # retrieve only the pixels within mask
        src_pix = list(
            zip(
                coli[mask],
                rowi[mask]
            )
        )
        # orthoproject pixels
        dst_pix = cc.unproject_points(

            src_pix,
            z
        )
        x_pix, y_pix, z_pix = dst_pix.T
        idx_y, idx_x = helpers.map_to_pixel(x_pix, y_pix, cc.transform)
        # ensure no pixels outside of target grid (can be in case of edges)
        idx_inside = np.all([idx_y >= 0, idx_y < len(y), idx_x >= 0, idx_x < len(x)], axis=0)
        idx_x = idx_x[idx_inside]
        idx_y = idx_y[idx_inside]
        # get 1D flat array indexes
        idx = np.array(idx_y) * len(x) + np.array(idx_x)

        # flatten points within mask
        da_point = da.stack(
            points=("y", "x")
        ).where(
            mask.stack(points=("y", "x")),
            drop=True
        )
        # da_point = da.where(mask, drop=True).stack(points=("y", "x"))
        da_point["points_idx"] = "points", np.where(mask.values.flatten())[0]
        # ensure any values that may be outside of target grid are dropped
        da_point = da_point.isel(points=idx_inside)

        # create a data array with relevant indexes
        da_idx = xr.DataArray(
            idx,
            dims=("points"),
            name="group",
            coords={
                "points": da_point.points.values,
            },
        )
        # retrieve unique values from this
        classes = np.unique(da_idx)
        # group unique values and reduce with average
        da_point = xarray_reduce(
            da_point.drop_vars(["y", "x", "points"]),
            da_idx,func=reducer,
            expected_groups=classes,
            engine="numba"
        )
        # replace the nearest by mean values where relevant
        da_point["group"] = da_new.group[classes]
        da_new[..., classes] = da_point
    return da_new.unstack()
