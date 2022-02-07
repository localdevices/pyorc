import cv2
import numpy as np
import rasterio
from pyproj import CRS
import geojson
from pyorc.cv import _corr_lens, _corr_color
import xarray as xr
from rasterio import warp


def frames(
    fn,
    frame_int=1,
    start_frame=0,
    end_frame=125,
    grayscale=False,
    lens_pars=None,
):
    """

    :param fn:  filename (str) or BytesIO object containing a movie dataset
    :param frame_int: int - frame interval, difference between frames
    :param start_time=0: start time of first frame to extract (ms)
    :param end_time=None: end time of last frame to extract (ms). If None, it is assumed the entire movie must be extracted
    :param grayscale=False: turn into grayscale image if set to True
    :param lens_pars=None: set of parameters passed to lens_corr if needed (e.g. {"k1": -10.0e-6, "c": 2, "f": 8.0}
    :return: list of time since start (ms), list of files generated
    """
    if isinstance(fn, str):
        cap = cv2.VideoCapture(fn)
    else:
        cap = fn
    # elif isinstance(fn, )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = frame_count
    if (start_frame > frame_count and frame_count > 0) :
        raise ValueError("Start frame is larger than total amount of frames")
    if end_frame < start_frame:
        raise ValueError(
            f"Start frame {start_frame} is larger than end frame {end_frame}"
        )
    # go to the right frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    _n = start_frame
    _t = 0.0
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    if (np.isinf(fps)) or (fps <= 0):
        t_index_start = 0 # assume it is zero
        # count framerate until the desired frame (add one to ensure you are at the start of the next frame)
        for n, no_frame in enumerate(range(end_frame - start_frame + 2)):
            dummy = cap.read()
            if n == 0:
                t_index_start = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_index_end = cap.get(cv2.CAP_PROP_POS_MSEC)
        fps = 1./((t_index_end - t_index_start) / (1000 * no_frame))
        print(f"Computed FPS is: {fps}")
        # cap.release()
        # reopen file
        # cap = cv2.VideoCapture(fn)
        # go to the right frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    #     # assume fps is 27
    #     fps = 27

    while (cap.isOpened()) and (_n <= end_frame):
        try:
            ret, img = cap.read()
        except:
            raise IOError(f"Cannot read next frame no. {_n}")
        if ret:
            # logger.debug(f"Saving frame {_n} with ret {ret} to {fn_out}")
            if lens_pars is not None:
                # apply lens distortion correction
                img = _corr_lens(img, **lens_pars)
            # apply gray scaling, contrast- and gamma correction
            if grayscale:
                # img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
                img = img.mean(axis=2)
            # update frame number
            _n += 1
            # # update frame time
            # _newt = cap.get(cv2.CAP_PROP_POS_MSEC)
            # print(f"Time diff is {_newt}")
            # _t = _newt
            _t += 1.0 / fps
            yield _t, img
        else:
            break
    return

def get_frame(cap, n=None, grayscale=False, lens_pars=None):
    if n is not None:
        # first move to the right position
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    try:
        ret, img = cap.read()
    except:
        raise IOError(f"Cannot read")
    if ret:
        if lens_pars is not None:
            # apply lens distortion correction
            img = _corr_lens(img, **lens_pars)
        if grayscale:
            # apply gray scaling, contrast- and gamma correction
            # img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
            img = img.mean(axis=2)
    return img

def to_geotiff(fn, z, transform, crs=None, compress=None):
    """
    Writes geotiff from an array (assumed 3d)
    :param fn: filename
    :param z: array (3-D)
    :param transform: Affine transform
    :param crs=None: coordinate ref system
    :param compress: compression if needed
    :return:
    """
    if crs is not None:
        try:
            crs = CRS.from_user_input(crs)
        except:
            raise ValueError(f'CRS "{crs}" is not valid')
        if crs.is_geographic:
            raise TypeError(
                "CRS is of geographic type, a projected type (unit: meters) is required"
            )
    with rasterio.open(
        fn,
        "w",
        driver="GTiff",
        height=z.shape[1],
        width=z.shape[2],
        count=z.shape[0],
        dtype=z.dtype,
        crs=crs.to_proj4() if crs is not None else None,
        transform=transform,
        compress=compress,
    ) as ds:
        for n, _z in enumerate(z):
            ds.write(_z, n + 1)


def to_geojson(geom, crs=None):
    """
    Converts a single geometry in a geographically aware geojson
    :param geom: shapely feature (single!!)
    :param crs=None: pyproj readible crs
    :return: str, geojson format
    """
    # prepare a crs
    if crs is not None:
        try:
            crs = CRS.from_user_input(crs)
        except:
            raise ValueError(f'CRS "{crs}" is not valid')
        if crs.is_geographic:
            raise TypeError(
                "CRS is of geographic type, a projected type (unit: meters) is required"
            )
        try:
            epsg = crs.to_epsg()
        except:
            raise ValueError(f"CRS cannot be converted to EPSG code")
    # prepare json compatible crs dict
    crs_json = (
        {"type": "EPSG", "properties": {"code": epsg}} if crs is not None else None
    )
    # prepare geojson feature
    f = geojson.Feature(geometry=geom, properties={"ID": 0})
    # collate all into a geojson feature collection
    return geojson.FeatureCollection([f], crs=crs_json)


def to_dataarray(data, name, x, y, time=None, attrs={}):
    """
    Converts list of data slices (per time) to a xarray DataArray with axes, name and attributes
    :param data: list - containing all separate data slices per time step in 2D numpy arrays
    :param name: string - name to provide to DataArray
    :param x: 1D numpy array - x-coordinates
    :param y: 1D numpy array - y-coordinates
    :param time: list - containing datetime objects as retrieved from bmi model
    :param attrs: dict - attrs to provide to DataArray
    :return: DataArray of data
    """
    if time is None:
        return xr.DataArray(
            data, name=name, dims=("y", "x"), coords={"y": y, "x": x}, attrs=attrs
        )
    else:
        return xr.DataArray(
            data,
            name=name,
            dims=("time", "y", "x"),
            coords={"time": time, "y": y, "x": x},
            attrs=attrs,
        )


def to_dataset(
    arrays, names, x, y, time, lat=None, lon=None, xs=None, ys=None, attrs=[]
):
    """
    Converts lists of arrays per time step to xarray Dataset
    :param names: list - containing strings with names of datas
    :param datas: list of lists with collected datasets (in 2D numpy slices per time step)
    :param x: 1D numpy array - x-coordinates
    :param y: 1D numpy array - y-coordinates
    :param time: list - containing datetime objects as retrieved from bmi model
    :param attributes: list - containing attributes belonging to datas
    :return: Dataset of all data in datas
    """
    # define lon and lat attributes
    lon_attrs = {
        "long_name": "longitude",
        "units": "degrees_east",
    }
    lat_attrs = {
        "long_name": "latitude",
        "units": "degrees_north",
    }
    x_attrs = {
        "axis": "X",
        "long_name": "x-coordinate in Cartesian system",
        "units": "m",
    }
    y_attrs = {
        "axis": "Y",
        "long_name": "y-coordinate in Cartesian system",
        "units": "m",
    }
    time_attrs = {
        "standard_name": "time",
        "long_name": "time",
    }

    # ensure attributes are available for all datasets
    if len(names) != len(arrays):
        raise ValueError(
            "the amount of data arrays is different from the amount of names provided"
        )
    if len(attrs) < len(arrays):
        # add ampty attributes
        for n in range(0, len(arrays) - len(attrs)):
            attrs.append({})
    # convert list of lists to list of arrays if needed
    arrays = [np.array(d) for d in arrays]
    # merge arrays together into one large dataset, using names and coordinates
    ds = xr.merge(
        [
            to_dataarray(a, name, x, y, time, attrs)
            for a, name, attrs in zip(arrays, names, attrs)
        ]
    )
    ds["y"] = y
    ds["x"] = x
    ds["time"] = time
    ds["x"].attrs = x_attrs
    ds["y"].attrs = y_attrs
    ds["time"].attrs = time_attrs
    if (lon is not None) and (lat is not None):
        lon_da = to_dataarray(lon, "lon", x, y, attrs=lon_attrs)
        lat_da = to_dataarray(lat, "lat", x, y, attrs=lat_attrs)
        ds = xr.merge([ds, lon_da, lat_da])
    if (xs is not None) and (ys is not None):
        xs_da = to_dataarray(xs, "x_grid", x, y, attrs=x_attrs)
        ys_da = to_dataarray(ys, "y_grid", x, y, attrs=y_attrs)
        ds = xr.merge([ds, xs_da, ys_da])

    return ds


def convert_cols_rows(fn, cols, rows, dst_crs=rasterio.crs.CRS.from_epsg(4326)):
    with rasterio.open(fn) as ds:
        coord_data = get_xs_ys(cols, rows, ds.transform, dst_crs)
        # xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
        # xs, ys = np.array(xs), np.array(ys)
        # lons, lats = warp.transform(ds.crs, dst_crs, xs.flatten(), ys.flatten())
        # lons, lats = (
        #     np.array(lons).reshape(xs.shape),
        #     np.array(lats).reshape(ys.shape),
        # )
        return coord_data

def get_xs_ys(cols, rows, transform, src_crs, dst_crs=rasterio.crs.CRS.from_epsg(4326)):
    """
    Computes rasters of x and y coordinates, and longitude and latitude coordinates of a certain raster
    based on row and column counts and a defined transform, source crs of that raster and target crs.

    :param cols: list of ints, defining the column counts
    :param rows: list of ints, defining the row counts
    :param transform: np.ndarray, 1D, with 6 rasterio compatible transform parameters
    :param src_crs: coordinate reference system of the source grid
    :param dst_crs: coordinate reference system of a transformed set of coordinates, defaults ot EPSG:4326 but can be altered to any other CRS if needed
    :return: 4 np.ndarray (MxN): xs: x-coordinates, ys: y-coordinates, lons: longitude coordinates, lats: latitude coordinates
    """
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    lons, lats = warp.transform(src_crs, dst_crs, xs.flatten(), ys.flatten())
    lons, lats = (
        np.array(lons).reshape(xs.shape),
        np.array(lats).reshape(ys.shape),
    )
    return xs, ys, lons, lats

def get_axes(cols, rows, resolution):
    """
    Retrieve a locally spaced axes for PIV results on the basis of resolution and row and col distances from the
    original frames
    :param cols: list with ints, columns, sampled from the original projected frames
    :param rows: list with ints, rows, sampled from the original projected frames
    :param resolution: resolution of original frames
    :return: np.ndarray (N), containing x-axis with origin at the left
             np.ndarray (N), containing y-axis with origin on the top

    """
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
    return x, y

def interp_coords(ds, xs, ys, zs=None, x_grid="x_grid", y_grid="y_grid"):
    """
    Interpolate all variables to supplied x and y coordinates. This function assumes that the grid
    can be rotated and that xs and ys are supplied following the projected coordinates supplied in
    "x_grid" and "y_grid" variables in ds. x-coordinates and y-coordinates that fall outside the
    domain of ds, are still stored in the result. Original coordinate values supplied are stored in
    coordinates "xcoords", "ycoords" and (if supplied) "zcoords"
    :param ds: xarray dataset
    :param xs: tuple or list-like, x-coordinates on which interpolation should be done
    :param ys: tuple or list-like, y-coordinates on which interpolation should be done
    :param zs: tuple or list-like, z-coordinates on which interpolation should be done, defaults to None
    :param x_grid: str, name of variable that stores the x coordinates in the projection in which "xs" is supplied
    :param y_grid: str, name of variable that stores the y coordinates in the projection in which "ys" is supplied
    :return: ds_points: xarray dataset, containing interpolated data at the supplied x and y coordinates
    """

    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    transform = affine_from_grid(ds[x_grid].values, ds[y_grid].values)

    # make a cols and rows temporary variable
    coli, rowi = np.meshgrid(np.arange(len(ds["x"])), np.arange(len(ds["y"])))
    ds["cols"], ds["rows"] = (["y", "x"], coli), (["y", "x"], rowi)

    # compute rows and cols locations of coordinates (x, y)
    rows, cols = rasterio.transform.rowcol(transform, list(xs), list(ys))
    rows, cols = np.array(rows), np.array(cols)

    # select x and y coordinates from axes
    idx = np.all(
        np.array([cols >= 0, cols < len(ds["x"]), rows >= 0, rows < len(ds["y"])]),
        axis=0,
    )
    x = np.empty(len(cols))
    x[:] = np.nan
    y = np.empty(len(rows))
    y[:] = np.nan
    x[idx] = ds["x"].isel(x=cols[idx])
    y[idx] = ds["y"].isel(y=rows[idx])
    # interpolate values from grid to list of x-y coordinates to grid in xarray format
    x = xr.DataArray(list(x), dims="points")
    y = xr.DataArray(list(y), dims="points")
    if np.isnan(x).all():
        raise ValueError("All bathymetry points are outside valid domain")
    else:
        ds_points = ds.interp(x=x, y=y)
    # add the xcoords and ycoords (and zcoords if available) originally assigned so that even points outside the grid covered by ds can be
    # found back from this dataset
    ds_points = ds_points.assign_coords(xcoords=("points", list(xs)))
    ds_points = ds_points.assign_coords(ycoords=("points", list(ys)))
    if zs is not None:
        ds_points = ds_points.assign_coords(zcoords=("points", list(zs)))

    return ds_points
