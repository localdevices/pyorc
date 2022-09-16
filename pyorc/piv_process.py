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
    v_x, v_y, s2n = openpiv.pyprocess.extended_search_area_piv(
        frame_a, frame_b, dt=dt, search_area_size=search_area_size, overlap=overlap, window_size=window_size, **kwargs
    )
    # cols, rows = get_piv_size(
    #     image_size=frame_a.shape, search_area_size=search_area_size, overlap=overlap
    # )

    if correlation:
        corr = piv_corr(
            frame_a,
            frame_b,
            search_area_size=search_area_size,
            overlap=overlap,
            window_size=window_size,
        )
    else:
        corr = np.zeros(s2n.shape)
        corr[:] = np.nan
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

