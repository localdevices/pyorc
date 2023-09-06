import openpiv.tools
import openpiv.pyprocess
import numpy as np
import xarray as xr


def piv(
    frame_a, frame_b, dt, res_x=0.01, res_y=0.01, search_area_size=30, correlation=True, window_size=None, overlap=None,
        **kwargs
):
    """PIV analysis following keyword arguments from openpiv. This function also computes the correlations per
    interrogation window, so that poorly correlated values can be filtered out. Furthermore, the resolution is used to convert
    pixel per second velocity estimates, into meter per second velocity estimates. The centre of search area columns
    and rows are also returned so that a georeferenced grid can be written from the results.

    Note: Typical openpiv kwargs are for instance
    window_size=60, overlap=30, search_area_size=60, dt=1./25

    Parameters
    ----------
    frame_a: np.ndarray (2D)
        first frame
    frame_b: np.ndarray (2D)
        second frame
    res_x: float, optional
        resolution of x-dir pixels in a user-defined unit per pixel (e.g. m pixel-1) Default: 0.01
    res_y: float, optional
        resolution of y-dir pixels in a user-defined unit per pixel (e.g. m pixel-1) Default: 0.01
    search_area_size: int, optional
        length of subsetted matrix to search for correlations (default: 30)
    correlation: bool, optional
        if True (default), the best found correlation coefficient is also returned for each interrogation
    window_size: int, optional
        size of interrogation window in amount of pixels. If not set, it is set equal to search_area_size (default: None).
    overlap: int, optional
        length of overlap between interrogation windows. If not set, this defaults to 50% of the window_size parameter (default: None).
    **kwargs: keyword arguments related to openpiv. See openpiv manual for further information

    Returns
    -------
    cols: np.ndarray (1D)
        col number of centre of interrogation windows
    rows: np.ndarray (1D)
        row number of centre of interrogation windows
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
    # if isinstance(frame_a, xr.core.dataarray.DataArray):
    #     frame_a = frame_a.values
    # if isinstance(frame_b, xr.core.dataarray.DataArray):
    #     frame_b = frame_b.values
    window_size = search_area_size if window_size is None else window_size
    overlap = int(round(window_size)/2) if overlap is None else overlap
    # v_x, v_y, s2n = openpiv.pyprocess.extended_search_area_piv(
    #     frame_a, frame_b, dt=dt, search_area_size=search_area_size, overlap=overlap, window_size=window_size, **kwargs
    # )
    # modified version of extended_search_area_piv to accomodate exporting corr
    v_x, v_y, s2n, corr = extended_search_area_piv(
        frame_a,
        frame_b,
        dt=dt,
        search_area_size=search_area_size,
        overlap=overlap,
        window_size=window_size,
        **kwargs
    )
    return v_x * res_x, v_y * res_y, s2n, corr


def get_piv_size(**kwargs):
    return openpiv.pyprocess.get_coordinates(**kwargs)


def piv_corr(
    frame_a,
    frame_b,
    search_area_size,
    overlap,
    window_size=None,
    correlation_method="circular",
    normalized_correlation=True,
):
    """Estimate the maximum correlation in piv analyses over two frames. Function taken from openpiv library.
    This is a temporary fix. If correlation can be exported from openpiv, then this function can be removed.

    Parameters
    ----------
    frame_a: np.ndarray (2D)
        first frame
    frame_b: np.ndarray (2D)
        second frame
    overlap: int, optional
        length of overlap between interrogation windows. If not set, this defaults to 50% of the window_size parameter (default: None).
    window_size: int, optional
        size of interrogation window in amount of pixels. If not set, it is set equal to search_area_size (default: None).
    search_area_size: int, optional
        length of subsetted matrix to search for correlations (default: 30)
    correlation_method: str, optional
        method for correlation used, as openpiv setting (default: "circular")
    normalized_correlation: boolean, optional
        if True (default) return a normalized correlation number between zero and one, else not normalized

    Returns
    -------
    corr : np.ndarray (2D)
        maximum correlations found in search areas
    """
    # extract the correlation matrix
    window_size = search_area_size if window_size is None else window_size
    # get field shape

    n_rows, n_cols = openpiv.pyprocess.get_field_shape(
        frame_a.shape, search_area_size, overlap
    )

    # We implement the new vectorized code
    aa = openpiv.pyprocess.moving_window_array(frame_a, search_area_size, overlap)
    bb = openpiv.pyprocess.moving_window_array(frame_b, search_area_size, overlap)

    if search_area_size > window_size:
        # before masking with zeros we need to remove
        # edges

        aa = openpiv.pyprocess.normalize_intensity(aa)
        bb = openpiv.pyprocess.normalize_intensity(bb)

        mask = np.zeros((search_area_size, search_area_size)).astype(aa.dtype)
        pad = np.int((search_area_size - window_size) / 2)
        mask[slice(pad, search_area_size - pad), slice(pad, search_area_size - pad)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = openpiv.pyprocess.fft_correlate_images(
        aa,
        bb,
        correlation_method=correlation_method,
        normalized_correlation=normalized_correlation,
    )
    corr = corr.max(axis=-1).max(axis=-1).reshape((n_rows, n_cols))
    return corr


def extended_search_area_piv(
        frame_a,
        frame_b,
        window_size,
        overlap=0,
        dt=1.0,
        search_area_size=None,
        correlation_method="circular",
        subpixel_method="gaussian",
        sig2noise_method='peak2mean',
        width=2,
        normalized_correlation=True,
        use_vectorized=False,
):
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
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

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    sig2noise : 2d np.ndarray, ( optional: only if sig2noise_method != None )
        a two dimensional array the signal to noise ratio for each
        window pair.


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
    if search_area_size is None:
        search_area_size = window_size

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
        mask[slice(pady, search_area_size[0] - pady),
        slice(padx, search_area_size[1] - padx)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = openpiv.pyprocess.fft_correlate_images(aa, bb,
                                correlation_method=correlation_method,
                                normalized_correlation=normalized_correlation)
    if use_vectorized == True:
        u, v = openpiv.pyprocess.vectorized_correlation_to_displacements(corr, n_rows, n_cols,
                                                       subpixel_method=subpixel_method)
    else:
        u, v = openpiv.pyprocess.correlation_to_displacement(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)

    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        if use_vectorized == True:
            sig2noise = openpiv.pyprocess.vectorized_sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
        else:
            sig2noise = openpiv.pyprocess.sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
    else:
        sig2noise = np.zeros_like(u) * np.nan

    sig2noise = sig2noise.reshape(n_rows, n_cols)
    # extended code for exporting the maximum found value for corr
    corr = corr.max(axis=-1).max(axis=-1).reshape((n_rows, n_cols))

    return u / dt, v / dt, sig2noise, corr

