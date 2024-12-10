"""PIV processing wrappers for OpenPIV."""

from typing import List, Optional, Tuple, Union

import numpy as np
import openpiv.pyprocess
import openpiv.tools
import xarray as xr

__all__ = [
    "get_openpiv",
    "piv",
]


def get_openpiv(frames, y, x, dt, **kwargs):
    """Compute time-resolved Particle Image Velocimetry (PIV) using Fast Fourier Transform (FFT) within OpenPIV.

    Calculates velocity using the OpenPIV algorithms by processing sequential frames
    from a dataset and returning the velocity components, signal-to-noise ratio, and
    correlation values. The function shifts frames in time and applies the PIV algorithm
    to compute flow fields over the specified spatial axes.

    Parameters
    ----------
    frames : xarray.Dataset
        The input dataset containing time-dependent frames with coordinates.
    y : array-like
        The spatial coordinates along the y-axis where the outputs should be interpolated.
    x : array-like
        The spatial coordinates along the x-axis where the outputs should be interpolated.
    dt : float
        The time step between consecutive frames (used to go from per-frame to per-second displacement).
    **kwargs : dict
        Additional keyword arguments to be passed to the PIV function.

    Returns
    -------
    xarray.Dataset
        A dataset containing computed velocity components `v_x` and `v_y`,
        signal-to-noise ratios `s2n`, and correlation values `corr`. The dataset
        includes updated x and y coordinates representing the flow field grid.

    """
    # first get rid of coordinates that need to be recalculated
    coords_drop = list(set(frames.coords) - set(frames.dims))
    frames = frames.drop_vars(coords_drop)
    # get frames and shifted frames in time
    frames1 = frames.shift(time=1)[1:].chunk({"time": 1})
    frames2 = frames[1:].chunk({"time": 1})
    # retrieve all data arrays
    v_x, v_y, s2n, corr = xr.apply_ufunc(
        piv,
        frames1,
        frames2,
        dt,
        kwargs=kwargs,
        input_core_dims=[["y", "x"], ["y", "x"], []],
        output_core_dims=[["new_y", "new_x"]] * 4,
        dask_gufunc_kwargs={
            "output_sizes": {"new_y": len(y), "new_x": len(x)},
        },
        output_dtypes=[np.float32] * 4,
        vectorize=True,
        keep_attrs=True,
        dask="parallelized",
    )
    # merge all DataArrays in one Dataset
    ds = xr.merge([v_x.rename("v_x"), v_y.rename("v_y"), s2n.rename("s2n"), corr.rename("corr")]).rename(
        {"new_x": "x", "new_y": "y"}
    )
    # add y and x-axis values
    ds["y"] = y
    ds["x"] = x
    return ds


def piv(
    frame_a,
    frame_b,
    dt,
    res_x=0.01,
    res_y=0.01,
    search_area_size=30,
    window_size=None,
    overlap=None,
    **kwargs,
):
    """Perform PIV analysis on two sequential frames following keyword arguments from openpiv.

    This function also computes the correlations per interrogation window, so that poorly correlated values can be
    filtered out. Furthermore, the resolution is used to convert pixel per second velocity estimates, into meter per
    second velocity estimates. The centre of search area columns and rows are also returned so that a georeferenced
    grid can be written from the results.

    Note: Typical openpiv kwargs are for instance
    window_size=60, overlap=30, search_area_size=60, dt=1./25

    Parameters
    ----------
    frame_a: np.ndarray (2D)
        first frame
    frame_b: np.ndarray (2D)
        second frame
    dt : float
        time resolution in seconds.
    res_x: float, optional
        resolution of x-dir pixels in a user-defined unit per pixel (e.g. m pixel-1) Default: 0.01
    res_y: float, optional
        resolution of y-dir pixels in a user-defined unit per pixel (e.g. m pixel-1) Default: 0.01
    search_area_size: int, optional
        length of subsetted matrix to search for correlations (default: 30)
    window_size: int, optional
        size of interrogation window in amount of pixels. If not set, it is set equal to search_area_size
        (default: None).
    overlap: int, optional
        length of overlap between interrogation windows. If not set, this defaults to 50% of the window_size parameter
        (default: None).
    **kwargs: dict
        keyword arguments related to openpiv. See openpiv manual for further information

    Returns
    -------
    v_x: np.ndarray(2D)
        raw x-dir velocities [m s-1] in interrogation windows (requires filtering to get valid velocities)
    v_y: np.ndarray (2D)
        raw y-dir velocities [m s-1] in interrogation windows (requires filtering to get valid velocities)
    s2n: np.ndarray (2D)
        signal to noise ratio, measured as maximum correlation found divided by the mean correlation
        (method="peak2mean") or second to maximum correlation (method="peak2peak") found within search area
    corr: np.ndarray (2D)
        correlation values in interrogation windows

    """
    window_size = search_area_size if window_size is None else window_size
    overlap = int(round(window_size) / 2) if overlap is None else overlap
    # modified version of extended_search_area_piv to accomodate exporting corr
    v_x, v_y, s2n, corr = extended_search_area_piv(
        frame_a, frame_b, dt=dt, search_area_size=search_area_size, overlap=overlap, window_size=window_size, **kwargs
    )
    return v_x * res_x, v_y * res_y, s2n, corr


def extended_search_area_piv(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    window_size: int,
    overlap: int = 0,
    dt: float = 1.0,
    search_area_size: Optional[Union[Tuple[int, int], List[int], int]] = None,
    correlation_method: str = "circular",
    subpixel_method: str = "gaussian",
    sig2noise_method: Optional[str] = "peak2mean",
    width: int = 2,
    normalized_correlation: bool = True,
    use_vectorized: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform PIV cross-correlation analysis.

    Extended area search can be used to increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. For Cython implementation see
    openpiv.process.extended_search_area_piv

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.

    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the first frame.

    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the second frame.

    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].

    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].

    dt : float
        the time delay separating the two frames [default: 1.0].

    correlation_method : string
        one of the two methods implemented: 'circular' or 'linear',
        default: 'circular', it's faster, without zero-padding
        'linear' requires also normalized_correlation = True (see below)

    subpixel_method : string
         one of the following methods to estimate subpixel location of the
         peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.

    sig2noise_method : string
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)

    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

    search_area_size : int
       the size of the interrogation window in the second frame,
       default is the same interrogation window size and it is a
       fallback to the simplest FFT based PIV

    normalized_correlation: bool
        if True, then the image intensity will be modified by removing
        the mean, dividing by the standard deviation and
        the correlation map will be normalized. It's slower but could be
        more robust

    use_vectorized : bool
        If set, vectorization is used to speed up analysis.

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    sig2noise : 2d np.ndarray ( optional: only if sig2noise_method != None )
        a two dimensional array the signal to noise ratio for each
        window pair.

    corr : 2d np.ndarray
        a two dimensional array with the maximum correlation values found in each interrogation window.

    The implementation of the one-step direct correlation with different
    size of the interrogation window and the search area. The increased
    size of the search areas cope with the problem of loss of pairs due
    to in-plane motion, allowing for a smaller interrogation window size,
    without increasing the number of outlier vectors.

    See:

    Particle-Imaging Techniques for Experimental Fluid Mechanics

    Annual Review of Fluid Mechanics
    Vol. 23: 261-304 (Volume publication date January 1991)
    DOI: 10.1146/annurev.fl.23.010191.001401

    originally implemented in process.pyx in Cython and converted to
    a NumPy vectorized solution in pyprocess.py

    """
    if search_area_size is not None:
        if isinstance(search_area_size, tuple) == False and isinstance(search_area_size, list) == False:
            search_area_size = [search_area_size, search_area_size]
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    # check the inputs for validity
    search_area_size = window_size if search_area_size is None else search_area_size

    if overlap[0] >= window_size[0] or overlap[1] >= window_size[1]:
        raise ValueError("Overlap has to be smaller than the window_size")

    if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
        raise ValueError("Search size cannot be smaller than the window_size")

    if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
        raise ValueError("window size cannot be larger than the image")

    # get field shape
    n_rows, n_cols = openpiv.pyprocess.get_field_shape(frame_a.shape, search_area_size, overlap)

    # We implement the new vectorized code
    aa = openpiv.pyprocess.sliding_window_array(frame_a, search_area_size, overlap)
    bb = openpiv.pyprocess.sliding_window_array(frame_b, search_area_size, overlap)

    # for the case of extended seearch, the window size is smaller than
    # the search_area_size. In order to keep it all vectorized the
    # approach is to use the interrogation window in both
    # frames of the same size of search_area_asize,
    # but mask out the region around
    # the interrogation window in the frame A

    if search_area_size > window_size:
        # before masking with zeros we need to remove
        # edges

        aa = openpiv.pyprocess.normalize_intensity(aa)
        bb = openpiv.pyprocess.normalize_intensity(bb)

        mask = np.zeros((search_area_size[0], search_area_size[1])).astype(aa.dtype)
        pady = int((search_area_size[0] - window_size[0]) / 2)
        padx = int((search_area_size[1] - window_size[1]) / 2)
        mask[slice(pady, search_area_size[0] - pady), slice(padx, search_area_size[1] - padx)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = openpiv.pyprocess.fft_correlate_images(
        aa, bb, correlation_method=correlation_method, normalized_correlation=normalized_correlation
    )
    if use_vectorized == True:
        u, v = openpiv.pyprocess.vectorized_correlation_to_displacements(
            corr, n_rows, n_cols, subpixel_method=subpixel_method
        )
    else:
        u, v = openpiv.pyprocess.correlation_to_displacement(corr, n_rows, n_cols, subpixel_method=subpixel_method)

    # return output depending if user wanted sig2noise information
    sig2noise = np.zeros_like(u) * np.nan
    if sig2noise_method is not None:
        if use_vectorized == True:
            sig2noise = openpiv.pyprocess.vectorized_sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
        else:
            sig2noise = openpiv.pyprocess.sig2noise_ratio(corr, sig2noise_method=sig2noise_method, width=width)

    sig2noise = sig2noise.reshape(n_rows, n_cols)
    # extended code for exporting the maximum found value for corr
    corr = corr.max(axis=-1).max(axis=-1).reshape((n_rows, n_cols))

    return u / dt, v / dt, sig2noise, corr
