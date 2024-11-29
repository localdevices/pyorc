"""PIV processing wrappers for FF-PIV."""

import warnings

import numpy as np
import xarray as xr
from ffpiv import cross_corr, u_v_displacement, window
from tqdm import tqdm


def get_ffpiv(frames, y, x, dt, window_size, overlap, search_area_size, res_y, res_x, chunks=None, engine="numba"):
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
    chunks : int, optional
        if provided, the frames will be treated in chunks as provided, if not, optimal chunk size is estimated based
        on available memory.
    engine : str, optional
        ff-piv engine to use, can be "numpy" or "numba". "numba" is generally much faster.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the signal-to-noise ratio (`s2n`), correlation maxima (`corr`), and velocity
        components (`v_x`, `v_y`).

    """
    # compute memory availability and size of problem to pipe to ffpiv functions
    dim_size = frames[0].shape
    req_mem = window.required_memory(
        n_frames=len(frames),
        dim_size=dim_size,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
    )
    if chunks is None:
        # estimate chunk size
        avail_mem = window.available_memory()
        print(f"Mem available: {avail_mem/1e9} GB")
        print(f"Mem required: {req_mem/1e9} GB")

        chunks = int((req_mem // avail_mem) + 1)

    chunksize = int(len(frames) // chunks + 1)
    frames_chunks = [frames[np.maximum(chunk * chunksize - 1, 0) : (chunk + 1) * chunksize] for chunk in range(chunks)]
    n_rows, n_cols = len(y), len(x)

    # make progress bar
    pbar = tqdm(frames_chunks, position=0, leave=True)
    pbar.set_description("Computing PIV per chunk")

    # Loop over list
    ds_piv_chunks = []  # datasets with piv results per chunk
    for da in pbar:
        # get time slice
        time = da.time[1:]
        dt_chunk = dt.sel(time=time)
        da.load()
        # perform cross correlation analysis yielding correlations for each interrogation window
        x_, y_, corr = cross_corr(
            da.values,
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
            normalize=False,
            engine=engine,
        )

        # get the maximum correlation per interrogation window
        corr_max = np.nanmax(corr, axis=(-1, -2))

        # get signal-to-noise, whilst suppressing nanmean over empty slice warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s2n = corr_max / np.nanmean(corr, axis=(-1, -2))

        # reshape corr / s2n to the amount of expected rows and columns
        s2n = s2n.reshape(-1, n_rows, n_cols)
        corr_max = corr_max.reshape(-1, n_rows, n_cols)
        u, v = u_v_displacement(corr, n_rows, n_cols, engine=engine)
        # convert into meter per second
        u *= res_x / np.expand_dims(dt_chunk, (1, 2))
        v *= res_y / np.expand_dims(dt_chunk, (1, 2))
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
    # concatenate all parts in time
    ds = xr.concat(ds_piv_chunks, dim="time")
    return ds
