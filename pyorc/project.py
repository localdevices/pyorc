import cv2
import dask
import numpy as np
import xarray as xr

from flox.xarray import xarray_reduce
from scipy.interpolate import RegularGridInterpolator
from pyorc import helpers, cv

__all__ = ["project_numpy", "project_cv"]

def project_cv(da, cc, x, y, z):
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


def project_numpy(da, cc, x, y, z, stride=10, radius=5):
    """
    Projection method that goes from pixels directly to target grid, including undistortion and projection
    using a lookup method across the grid.

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
    stride

    Returns
    -------
    da_proj : xr.DataArray
        Reprojected Frames

    """
    coli, rowi = np.meshgrid(
        np.arange(len(da.x)),
        np.arange(len(da.y))
    )
    src_pix = list(zip(coli[::stride, ::stride].flatten(), rowi[::stride, ::stride].flatten()))
    dst_pix = cc.unproject_points(
        src_pix,
        z
    )
    # x_pix, y_pix, z_pix = list(zip(*dst_pix))
    x_pix, y_pix, z_pix = dst_pix.T
    # reorganise to 2D grid
    x_pix = x_pix.reshape(*coli[::stride, ::stride].shape)
    y_pix = y_pix.reshape(*coli[::stride, ::stride].shape)

    # upscale
    interp_x = RegularGridInterpolator((rowi[::stride, 0], coli[0, ::stride]), x_pix, bounds_error=False,
                                       fill_value=None)
    interp_y = RegularGridInterpolator((rowi[::stride, 0], coli[0, ::stride]), y_pix, bounds_error=False,
                                       fill_value=None)
    x_pix_up = interp_x((rowi, coli))
    y_pix_up = interp_y((rowi, coli))
    idx_y_up, idx_x_up = helpers.map_to_pixel(x_pix_up, y_pix_up, cc.transform)

    # any location outside of the target grid should become a miss
    miss = np.any([idx_x_up >= cc.shape[1], idx_x_up < 0, idx_y_up >= cc.shape[0], idx_y_up < 0], axis=0)

    # flatten to 1D-indexes
    idx = np.array(idx_y_up) * len(x) + np.array(idx_x_up)

    # ensure that indexes outside of area of interest are set to -1.
    idx[miss] = -1

    # reshape indexes to the source grid. Now we know of each pixel in source, where it belongs in target
    # idx = idx.reshape(*coli.shape)

    # turn idx grid into a DataArray
    da_idx = xr.DataArray(
        idx,
        dims=("y", "x"),
        name="group",
        coords={
            "y": da.y.values,
            "x": da.x.values
        },
    )
    # retrieve unique values from this
    classes = np.unique(da_idx)
    # now we simply group the frames by all the indexes and then take the mean of all identified points per index
    da_point = xarray_reduce(da, da_idx, func="mean", expected_groups=classes, engine="numba")
    # remaining problem is that the above indexes may be limited to only a few, and cannot be coerced to the grid that we would like.
    # So we make a lazy array of the new interpolated shape. But now we stack this array over y and x, so that we can paste the
    # interpolated values onto the new array. All to be kept lazy (chunk 1 time step) to prevent memory issues.
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
    ).stack(group=("y", "x"))
    idxs = da_new.group.isel(group=np.unique(da_idx))

    # assign the values to the relevant ids
    da_point["group"] = idxs
    # da_new
    if "rgb" in da_new.coords:
        da_new[:, :, np.unique(da_idx)] = da_point
        # get one sample, and create a mask
        mask = np.int8(helpers.get_enclosed_mask(da_new[0][0].unstack().values))
    else:
        da_new[:, np.unique(da_idx)] = da_point
        # get one sample, and create a mask
        mask = np.int8(helpers.get_enclosed_mask(da_new[0].unstack().values))
    da_new = da_new.unstack()

    # da_fill = da_new
    da_fill = xr.apply_ufunc(
        helpers.mask_fill,
        da_new,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        output_dtypes=da.dtype,
        kwargs={
            "mask": mask,
            "radius": radius
        },
        dask='parallelized',
        keep_attrs=True,
        vectorize=True
    )
    return da_fill