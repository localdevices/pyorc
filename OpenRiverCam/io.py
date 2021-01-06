import cv2
import numpy as np
import rasterio
from pyproj import CRS
import geojson
from OpenRiverCam.cv import _corr_lens, _corr_color
import xarray as xr
from rasterio import warp

def frames(
    fn,
    frame_int=1,
    start_frame=0,
    end_frame=None,
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
    # elif isinstance(fn, )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = frame_count
    if start_frame > frame_count:
        raise ValueError("Start frame is larger than total amount of frames")
    if end_frame <= start_frame:
        raise ValueError(
            f"Start frame {start_frame} is larger than end frame {end_frame}"
        )
    # go to the right frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    _n = start_frame
    _t = 0.0
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
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
                img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
            # update frame number
            _n += 1
            # update frame time
            _t += 1.0 / fps
            yield _t, img
        else:
            break
    return

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
    crs_json = {"type": "EPSG", "properties": {"code": epsg}} if crs is not None else None
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
        return xr.DataArray(data,
                            name=name,
                            dims=('y', 'x'),
                            coords={
                                    'y': y,
                                    'x': x
                                    },
                            attrs=attrs
                            )
    else:
        return xr.DataArray(data,
                            name=name,
                            dims=('time', 'y', 'x'),
                            coords={
                                'time': time,
                                'y': y,
                                'x': x
                            },
                            attrs=attrs
                           )

def to_dataset(arrays, names, x, y, time, lat=None, lon=None, attrs=[]):
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
    time_attrs ={
        "standard_name": "time",
        "long_name": "time",
    }

    # ensure attributes are available for all datasets
    if len(names) != len(arrays):
        raise ValueError("the amount of data arrays is different from the amount of names provided")
    if len(attrs) < len(arrays):
        # add ampty attributes
        for n in range(0, len(arrays) - len(attrs)):
            attrs.append({})
    # convert list of lists to list of arrays if needed
    arrays = [np.array(d) for d in arrays]
    # merge arrays together into one large dataset, using names and coordinates
    ds = xr.merge([to_dataarray(a, name, x, y, time, attrs) for a, name, attrs in zip(arrays, names, attrs)])
    ds["y"] = y
    ds["x"] = x
    ds["time"] = time
    ds["x"].attrs = x_attrs
    ds["y"].attrs = y_attrs
    ds["time"].attrs = time_attrs
    if (lon is not None) and (lat is not None):
        lon_da = to_dataarray(lon, 'lon', x, y, attrs=lon_attrs)
        lat_da = to_dataarray(lat, 'lat', x, y, attrs=lat_attrs)
    ds = xr.merge([ds, lon_da, lat_da])
    return ds

def convert_cols_rows(fn, cols, rows, dst_crs=rasterio.crs.CRS.from_epsg(4326)):
    with rasterio.open(fn) as ds:
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
        xs, ys = np.array(xs), np.array(ys)
        xcoords, ycoords = warp.transform(ds.crs, dst_crs, xs.flatten(), ys.flatten())
        xcoords, ycoords = np.array(xcoords).reshape(xs.shape), np.array(ycoords).reshape(ys.shape)
        return xcoords, ycoords

