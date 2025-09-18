"""PIV processing wrappers for FF-PIV."""

import gc
import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import xarray as xr
from ffpiv import cross_corr, u_v_displacement, window
from tqdm import tqdm


def load_frame_chunk(da):
    """Load frame chunk into memory. If not successful try to reduce size of frame chunk."""
    da_loaded = da.copy(deep=True)
    try:
        da_loaded = da.load()
    except TypeError:
        da_loaded = da[:-1]
        da_loaded = load_frame_chunk(da_loaded)
    return da_loaded


def get_ffpiv(
    frames: xr.DataArray,
    y: np.ndarray,
    x: np.ndarray,
    dt: np.ndarray,
    window_size: Tuple[int, int],
    overlap: Tuple[int, int],
    search_area_size: Tuple[int, int],
    res_y: float,
    res_x: float,
    chunksize: Optional[int] = None,
    memory_factor: float = 2,
    engine: Literal["numba", "numpy", "openpiv"] = "numba",
    ensemble_corr: bool = False,
    corr_min: float = 0.2,
    s2n_min: float = 3,
    count_min: float = 0.2,
):
    """Compute time-resolved Particle Image Velocimetry (PIV) using Fast Fourier Transform (FFT) within FF-PIV.

    This function calculates the velocity field from a sequence of image frames using PIV techniques. The process
    involves dividing the images into interrogation windows, performing cross-correlation analysis to capture
    displacement, and then calculating the velocity components. The function efficiently handles large datasets by
    computing PIV per chunk of frames and manages memory usage.

    The method generally is much faster than OpenPIV because it utilizes parallelized functions implemented in numba.

    Parameters
    ----------
    frames : xr.DataArray
        image frames to be processed (pyorc.Frames compatible).
    y : np.ndarray
        Array of expected y-coordinates of velocimetry fields.
    x : np.ndarray
        Array of x-coordinates.
    dt : array_like
        Array of time intervals between each frame pair.
    window_size : [int, int]
        Size of the interrogation window.
    overlap : [int, int]
        Overlap between interrogation windows.
    search_area_size : [int, int]
        Size of the search area for cross-correlation.
    res_y : float
        Spatial resolution in the y-direction.
    res_x : float
        Spatial resolution in the x-direction.
    chunksize : int, optional
        if provided, the frames will be treated per chunk with `chunksize` as provided, if not, optimal chunk size is
        estimated based on available memory.
    memory_factor : float, optional
        available memory is divided by this factor to estimate the chunk size. Default is 4.
    engine : str, optional
        ff-piv engine to use, can be "numpy" or "numba". "numba" is generally much faster.
    ensemble_corr : bool, optional
        If True, performs PIV by first averaging cross-correlations across all frames and then deriving velocities.
        If False, computes velocities for each frame pair separately. Default is True.
    corr_min : float, optional
        Minimum correlation value to accept for vvelocity detection. Default is 0.2.
        In cases with background not removed, you may increase this value to reduce noise but preferred is to
        perform preprocessing in order to reduce background noise.
    s2n_min : float, optional
        Minimum signal-to-noise ratio to accept for velocity detection. Default is 3.
        A value of 1.0 means there is no signal at all, so velocities are entirely
        random. If you get very little velocities back, you may try to reduce this.
    count_min : float, optional
        Minimum amount of frame pairs that result in accepted correlation values after filtering on `corr_min` and
        `s2n_min`. Default 0.5. If less frame pairs are available, the velocity is filtered out.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the signal-to-noise ratio (`s2n`), correlation maxima (`corr`), and velocity
        components (`v_x`, `v_y`).

    """
    CHUNK_SIZE_ERROR = (
        "Chunk size with selected nr of chunks ({chunks}) is 2 or less. If you manually "
        "selected `chunks={chunks}` then consider increasing chunk size to at least 2, and preferrably more. If memory "
        "is limited, consider closing memory intensive applications. If pyorc crashes, then this is due to "
        " insufficient memory."
    )
    CHUNK_SIZE_WARNING = (
        "Memory availability is poor ({avail_mem} GB). Chunk size is automatically set to {chunksize} to avoid "
        "memory issues. If pyorc crashes, then this is due to insufficient memory. Consider to manually set a lower "
        "chunk size e.g using `get_piv(engine={engine}, chunk=2)` or `get_piv(engine={engine}, chunk=3)` or close "
        "memory intensive applications."
    )
    # compute memory availability and size of problem to pipe to ffpiv functions
    dim_size = frames[0].shape
    req_mem = window.required_memory(
        n_frames=len(frames),
        dim_size=dim_size,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
    )
    if chunksize is None:
        # estimate chunk size
        avail_mem = window.available_memory() / memory_factor
        chunks = int((req_mem // avail_mem) + 1)
        chunksize = int(np.ceil((len(frames)) / chunks))
        if chunksize <= 5:
            warnings.warn(
                CHUNK_SIZE_WARNING.format(avail_mem=avail_mem / 1e9, chunksize=chunksize, engine=engine), stacklevel=2
            )
            chunksize = 5  # hard override, try to manage with 5
            chunks = int(np.ceil((len(frames)) / chunksize))
    if chunksize < 2:
        raise OverflowError(CHUNK_SIZE_ERROR.format(chunks=chunks))
    frames_chunks = [frames[np.maximum(chunk * chunksize - 1, 0) : (chunk + 1) * chunksize] for chunk in range(chunks)]
    # check if there are chunks that are too small in size, needs to be at least 2 frames per chunk
    frames_chunks = [frames_chunk for frames_chunk in frames_chunks if len(frames_chunk) >= 2]
    n_rows, n_cols = len(y), len(x)
    if ensemble_corr:
        ds = _get_ffpiv_mean(
            frames_chunks,
            y,
            x,
            dt,
            res_y,
            res_x,
            n_cols,
            n_rows,
            window_size,
            overlap,
            search_area_size,
            engine,
            corr_min,
            s2n_min,
            count_min,
        )
    else:
        ds = _get_ffpiv_timestep(
            frames_chunks, y, x, dt, res_y, res_x, n_cols, n_rows, window_size, overlap, search_area_size, engine
        )
    return ds


def _get_ffpiv_mean(
    frames_chunks,
    y,
    x,
    dt,
    res_y,
    res_x,
    n_cols,
    n_rows,
    window_size,
    overlap,
    search_area_size,
    engine,
    corr_min,
    s2n_min,
    count_min,
):
    def process_frame_chunk(frame_chunk, corr_min, s2n_min):
        """Process a single frame chunk to compute correlation and signal-to-noise.

        Parameters
        ----------
        frame_chunk : xr.DataArray
            subset of all frames, manageable in memory size.
        corr_min : float, optional
            Minimum correlation value to accept for velocity detection.
        s2n_min : float, optional
            Minimum signal-to-noise ratio to accept for velocity detection.

        Returns
        -------
        corr : np.ndarray
            correlation windows, filtered for minimum correlation and signal-to-noise.
        corr_max : np.ndarray
            maximum correlation per interrogation window
        s2n : np.ndarray
            signal-to-noise ratio per interrogation window, computed as max(corr) / mean(corr) per window

        """
        x_, y_, corr = cross_corr(
            frame_chunk.values,
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
            normalize=False,
            engine=engine,
            verbose=False,
        )
        # Suppress RuntimeWarnings and calculate required metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr_max = np.max(corr, axis=(-1, -2))
            # corr_max2 = np.nanmax(corr, axis=(-1, -2))
            s2n = corr_max / np.mean(corr, axis=(-1, -2))
            # s2n_2 = corr_max2 / np.nanmean(corr, axis=(-1, -2))
        # Apply thresholds
        masks = (corr_max >= corr_min) & (s2n >= s2n_min) & (np.isfinite(corr_max))
        corr[~masks] = corr_max[~masks] = s2n[~masks] = 0.0
        # masks = (corr_max >= corr_min) & (s2n >= s2n_min)
        # corr[~masks] = corr_max[~masks] = s2n[~masks] = np.nan

        return corr, corr_max, s2n

    def aggregate_results(corr_chunks, s2n_chunks, corr_count, corr_sum, n_frames):
        """Aggregate correlation and signal-to-noise data from multiple chunks into average statistics.

        This function processes correlation and signal-to-noise data chunks to calculate
        mean correlation, mean maximum correlation, and mean signal-to-noise values after
        handling missing data and refining input arrays. Final results are reshaped
        into their respective dimensions for further analysis.

        Parameters
        ----------
        corr_chunks: list of numpy.ndarray
            List of correlation data chunks to be concatenated and processed.
        s2n_chunks: list of numpy.ndarray
            List of signal-to-noise ratio chunks to be concatenated and processed.
        corr_count: numpy.ndarray
            Array representing the count of non-missing correlation values across frames, this is used for averaging.
        corr_sum: numpy.ndarray
            Array representing the sum of correlation values across frames (ex NaN values), this is used for averaging.
        n_frames: int
            Total number of frames to normalize and process the data.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            - corr_mean: Mean correlation values across chunks and frames.
            - corr_max_mean: Mean maximum correlation values reshaped to fit data dimensions.
            - s2n_mean: Mean signal-to-noise ratio values reshaped to fit data dimensions.

        """
        s2n_concat = np.concatenate(s2n_chunks, axis=0)
        corr_max_concat = np.concatenate(corr_chunks, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr_sum[corr_count < count_min * n_frames] = np.nan
            corr_mean = corr_sum / corr_count
            corr_max_mean = np.nanmean(corr_max_concat, axis=0).reshape(-1, n_rows, n_cols)
            s2n_mean = np.nanmean(s2n_concat, axis=0).reshape(-1, n_rows, n_cols)

        return corr_mean, corr_max_mean, s2n_mean

    def finalize_ds(corr_mean, corr_max, s2n, res_x, res_y, dt_av, y, x):
        """Finalize the dataset by computing displacements, normalizing values, and assembling an `xr.Dataset`.

        Computes displacements from correlation data using pre-defined parameters, normalizes
        these displacements with the given spatial and temporal resolutions, and returns the results as an
        `xr.Dataset`. The resulting Dataset contains signal-to-noise ratio data, maximum correlation values,
        and displacement vectors.

        Parameters
        ----------
        corr_mean: numpy.ndarray
            Array containing mean correlation values.
        corr_max: numpy.ndarray
            Array containing maximum correlation values.
        s2n: numpy.ndarray
            Signal-to-noise ratio data array.
        res_x: float
            Spatial resolution in the x-direction used to convert pix/s into m/s.
        res_y: float
            Spatial resolution in the y-direction used to convert pix/s into m/s.
        dt_av: float
            Temporal resolution used to convert pix/s into m/s.
        y: numpy.ndarray
            Array of spatial coordinates in the y-dimension.
        x: numpy.ndarray
            Array of spatial coordinates in the x-dimension.

        Returns
        -------
        xarray.Dataset
            Contains signal-to-noise ratio data, correlation values, and displacement
            components in the x and y directions in m/s.

        """
        u, v = u_v_displacement(corr_mean, n_rows, n_cols, engine=engine)
        u = (u * res_x / dt_av).astype(np.float32)
        v = (v * res_y / dt_av).astype(np.float32)

        # Build xarray Dataset
        ds = xr.Dataset(
            {
                "s2n": (["time", "y", "x"], s2n),
                "corr": (["time", "y", "x"], corr_max),
                "v_x": (["time", "y", "x"], u),
                "v_y": (["time", "y", "x"], v),
            },
            coords={"time": time[0:1], "y": y, "x": x},
        )
        return ds

    # make progress bar
    pbar = tqdm(range(len(frames_chunks)), position=0, leave=True)
    pbar.set_description("Computing PIV per chunk")
    # predefine empty lists for processed chunks
    corr_chunks, s2n_chunks = [], []
    corr_sum, corr_count = 0.0, 0.0  # initialize the sum of the correlation windows by a zero

    # loop through frame chunks
    for n in pbar:
        da = frames_chunks[n]
        # get time slice
        da = load_frame_chunk(da)
        time = da.time[1:]
        # dt_chunk = dt.sel(time=time)
        # we need at least one image-pair to do PIV
        if len(da) < 2:
            continue

        # perform cross correlation analysis yielding masked correlations for each interrogation window
        corr, corr_max, s2n = process_frame_chunk(da, corr_min, s2n_min)
        # housekeeping
        corr_sum += np.sum(corr, axis=0, keepdims=True)
        corr_count += np.sum(~np.isclose(corr, 0.0), axis=0, keepdims=True)
        # corr_sum += np.nansum(corr, axis=0, keepdims=True)
        # corr_count += np.nansum(~np.isnan(corr), axis=0, keepdims=True)
        corr_chunks.append(corr_max)
        s2n_chunks.append(s2n)

        # remove chunk safely from memory ASAP
        frames_chunks[n] = None
        del da
        gc.collect()
    # concatenate results
    dt_av = dt.values.mean()
    n_frames = len(corr_chunks)
    corr_mean, corr_max_mean, s2n_mean = aggregate_results(corr_chunks, s2n_chunks, corr_count, corr_sum, n_frames)
    # create final dataset
    return finalize_ds(corr_mean, corr_max_mean, s2n_mean, res_x, res_y, dt_av, y, x)


def _get_ffpiv_timestep(
    frames_chunks, y, x, dt, res_y, res_x, n_cols, n_rows, window_size, overlap, search_area_size, engine
):
    # make progress bar
    pbar = tqdm(range(len(frames_chunks)), position=0, leave=True)
    pbar.set_description("Computing PIV per chunk")
    ds_piv_chunks = []
    for n in pbar:
        da = frames_chunks[n]
        # get time slice
        da = load_frame_chunk(da)
        time = da.time[1:]
        dt_chunk = dt.sel(time=time)
        # we need at least one image-pair to do PIV
        if len(da) >= 2:
            u, v, corr_max, s2n = _get_uv_timestep(
                frames_chunks[n], n_cols, n_rows, window_size, overlap, search_area_size, engine=engine
            )
            u = (u * res_x / np.expand_dims(dt_chunk, (1, 2))).astype(np.float32)
            v = (v * res_y / np.expand_dims(dt_chunk, (1, 2))).astype(np.float32)

            # put s2n, corr_max, u and v in one xarray dataset, with coordinates time, y and x
            ds = xr.Dataset(
                {
                    "s2n": (["time", "y", "x"], s2n),
                    "corr": (["time", "y", "x"], corr_max),
                    "v_x": (["time", "y", "x"], u),
                    "v_y": (["time", "y", "x"], v),
                },
                coords={
                    "time": time,
                    "y": y,
                    "x": x,
                },
            )
            # u and v to meter per second
            ds_piv_chunks.append(ds)
        # remove chunk safely from memory
        frames_chunks[n] = None
        del da
        gc.collect()
    # concatenate all parts in time
    ds = xr.concat(ds_piv_chunks, dim="time")
    return ds


def _get_uv_timestep(da, n_cols, n_rows, window_size, overlap, search_area_size, engine="numba"):
    # perform cross correlation analysis yielding correlations for each interrogation window
    x_, y_, corr = cross_corr(
        da.values,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        normalize=False,
        engine=engine,
        verbose=False,
    )

    # get the maximum correlation per interrogation window
    corr_max = np.nanmax(corr, axis=(-1, -2))

    # get signal-to-noise, whilst suppressing nanmean over empty slice warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s2n = corr_max / np.nanmean(corr, axis=(-1, -2))

    # reshape corr / s2n to the amount of expected rows and columns
    s2n = (s2n.reshape(-1, n_rows, n_cols)).astype(np.float32)
    corr_max = (corr_max.reshape(-1, n_rows, n_cols)).astype(np.float32)
    u, v = u_v_displacement(corr, n_rows, n_cols, engine=engine)

    # convert into meter per second and store as float32 to save memory / disk space
    return u, v, corr_max, s2n
