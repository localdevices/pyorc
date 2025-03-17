"""pyorc projection functions."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numba as nb
import numpy as np
import xarray as xr

from . import cv

# from . import CameraConfig

__all__ = ["project_numpy", "project_cv"]


@nb.njit(
    nb.float32[:](nb.float32[:], nb.int64[:], nb.int64),
    nogil=True,
    parallel=False,
    cache=True,
)
def _group_average(data, idx, num_groups):
    """Compute group averages on sampled data, using unique values in idx as group indices.

    Parameters
    ----------
    data : np.ndarray
        1d-array containing values sampled from the original data
    idx : np.ndarray
        1d-array containing indices of the original data that correspond to the values in data.
        idx has the same size as data
    num_groups : int
        the amount of groups to average over. I.e. result of np.unique(idx).size

    """
    # Arrays to hold the sum and count for each group
    group_sums = np.zeros(num_groups, dtype=np.float32)
    group_counts = np.zeros(num_groups, dtype=np.int64)

    # Accumulate sums and counts for each group
    for i in range(len(data)):
        group_sums[idx[i]] += data[i]
        group_counts[idx[i]] += 1

    # Compute the averages
    averages = np.zeros(num_groups, dtype=np.float32)
    for group in range(num_groups):
        # if group_counts[group] > 0:  # Avoid division by zero
        averages[group] = group_sums[group] / group_counts[group]
    return averages


def project_cv(da: xr.DataArray, cc: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, reducer: str):
    """Projection method that uses pure OpenCV.

    Reprojection is done in two steps: undistortion and reprojection.
    This method gives incorrect mapping in case of very strong distortion and/or where part of the area of interest is
    outside of the field of view. In these cases, it is strongly recommended to use `project_numpy` instead.

    Parameters
    ----------
    da : xr.DataArray
        Frames time series
    cc : pyorc.CameraConfig
        pyorc CameraConfig object
    x : np.ndarray
        x-axis of target ortho image
    y : np.ndarray
        y-axis of target ortho image
    z : float
        vertical level value in real-world coordinates
    reducer : str
        not used, only passed to make function commensurate with project_numpy

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
        kwargs={"camera_matrix": cc.camera_matrix, "dist_coeffs": cc.dist_coeffs},
        dask="parallelized",
        keep_attrs=True,
        vectorize=True,
    )  # .rename({
    # also undistort the src control points
    cc.gcps["src"] = cv.undistort_points(cc.gcps["src"], cc.camera_matrix, cc.dist_coeffs)
    h_a = cc.z_to_h(z)
    src = cc.get_bbox(mode="camera", h_a=h_a, expand_exterior=False).exterior.coords[0:4]
    dst_xy = cc.get_bbox(expand_exterior=False).exterior.coords[0:4]
    # get geographic coordinates bbox corners
    dst = cv.transform_to_bbox(dst_xy, cc.bbox, cc.resolution)
    M = cv.get_M_2D(src, dst)
    da_proj = xr.apply_ufunc(
        cv.get_ortho,
        da_undistort,
        kwargs={"M": M, "shape": tuple(np.flipud(cc.shape)), "flags": cv2.INTER_AREA},
        input_core_dims=[["y", "x"]],
        output_core_dims=[["new_y", "new_x"]],
        dask_gufunc_kwargs={
            "output_sizes": {"new_y": len(y), "new_x": len(x)},
        },
        output_dtypes=[da.dtype],
        vectorize=True,
        exclude_dims=set(("y", "x")),
        dask="parallelized",
        keep_attrs=True,
    ).rename({"new_y": "y", "new_x": "x"})
    da_proj["y"] = y
    da_proj["x"] = x
    return da_proj


def img_to_ortho(img, x, y, idx_img, idx_ortho, src_idx=None, uidx=None, norm_idx=None):
    """Project from original image to ortho image using pre-calculated index mapping.

    This function can use nearest neighbour (for undersampled areas)  as well as averages (in oversampled areas).
    If `src_idx` is provided, then the averages are computed using the values in `src_idx` as group indices.

    Parameters
    ----------
    img : np.ndarray
        2d array containing the original image
    x : np.ndarray
        x-axis of target ortho image
    y : np.ndarray
        y-axis of target ortho image
    idx_img : np.ndarray[int]
        indices of the original image that correspond to the values in img. Used for nearest neighbour.
    idx_ortho : np.ndarray[int]
        indixes of the target ortho image that correspond to the values in img. Used for nearest neighbour.
    src_idx : np.ndarray
        1D array of flattened indices of the source image pixels that correspond
        to selected orthographic grid pixels.
    uidx : np.ndarray
        Sorted unique indices of the filtered orthographic grid pixels.
    norm_idx : np.ndarray
        Normalized indices corresponding to their positions in the unique
        filtered orthographic grid pixels. Used for averaging.

    """
    img = np.float32(img.flatten())
    # first make new flattened image
    new_arr = np.zeros((len(y) * len(x)))
    new_arr[idx_ortho] = img[idx_img]
    if src_idx is not None:
        samples_for_mean = img[src_idx]
        # Compute Group Averages
        averages = _group_average(samples_for_mean, norm_idx, len(uidx))
        new_arr[uidx] = averages
    # reshape to original grid
    return new_arr.reshape(len(y), -1)


def project_numpy(
    da: xr.DataArray,
    cc: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    reducer: Optional[str] = "mean",
):
    """Project from FOV pixels directly to target grid, including undistortion and projection.

    Parameters
    ----------
    da : xr.DataArray
        Frames time series
    cc : pyorc.CameraConfig
        pyorc CameraConfig object
    x : np.ndarray
        x-axis of target ortho image
    y : np.ndarray
        y-axis of target ortho image
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
    idx_img_nn, idx_ortho_nn = cc.map_idx_img_ortho(x, y, z)
    if reducer == "mean":
        src_idx_mean, uidx_mean, norm_idx_mean = cc.map_mean_idx_img_ortho(x, y, z)
    else:
        src_idx_mean, uidx_mean, norm_idx_mean = None, None, None
    # da.load()
    da_proj = xr.apply_ufunc(
        img_to_ortho,
        da,
        kwargs={
            "x": x,
            "y": y,
            "idx_img": idx_img_nn,
            "idx_ortho": idx_ortho_nn,
            "src_idx": src_idx_mean,
            "uidx": uidx_mean,
            "norm_idx": norm_idx_mean,
        },
        input_core_dims=[["y", "x"]],
        output_core_dims=[["new_y", "new_x"]],
        dask_gufunc_kwargs={
            "output_sizes": {"new_y": len(y), "new_x": len(x)},
        },
        output_dtypes=[da.dtype],
        vectorize=True,
        exclude_dims=set(("y", "x")),
        dask="parallelized",
        keep_attrs=True,
    ).rename({"new_y": "y", "new_x": "x"})
    da_proj["y"] = y
    da_proj["x"] = x
    return da_proj
