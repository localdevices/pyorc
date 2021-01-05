import openpiv.tools
import openpiv.pyprocess
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio import warp
from datetime import datetime, timedelta
import xarray as xr
from rasterio.plot import show

def list_to_dataarray(data, name, x, y, time=None, attrs={}):
    """
    Converts list of data slices (per time) to a xarray DataArray with axes, name and attributes
    :param data: list - containing all separate data slices per time step in 2D numpy arrays
    :param time: list - containing datetime objects as retrieved from bmi model
    :param x: 1D numpy array - x-coordinates
    :param y: 1D numpy array - y-coordinates
    :param name: string - name to provide to DataArray
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

def merge_outputs(datas, time, x, y, names, attributes):
    """
    Converts datasets collected per time step from bmi run to xarray Dataset
    :param datas: list of lists with collected datasets (in 2D numpy slices per time step)
    :param time: list - containing datetime objects as retrieved from bmi model
    :param x: 1D numpy array - x-coordinates
    :param y: 1D numpy array - y-coordinates
    :param names: list - containing strings with names of datas
    :param attributes: list - containing attributes belonging to datas
    :return: Dataset of all data in datas
    """
    ds = xr.merge([list_to_dataarray(data, name, x, y, time, attrs) for data, name, attrs in zip(datas, names, attributes)])
    ds["y"] = y
    ds["x"] = x
    ds["time"] = time
    ds["x"].attrs = {
        "axis": "X",
        "long_name": "x-coordinate in Cartesian system",
        "units": "m",
    }
    ds["y"].attrs = {
        "axis": "Y",
        "long_name": "y-coordinate in Cartesian system",
        "units": "m",
    }
    ds["time"].attrs = {
        "standard_name": "time",
        "long_name": "time",
    }
    return ds


folder = r"/home/hcwinsemius/OpenRiverCam"
start_time = "2020-12-17 07:45:00"
t = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

dst_crs = rasterio.crs.CRS.from_epsg(4326)  # lat lon projection
src = os.path.join(folder, "ortho_proj")
dst = os.path.join(folder, "piv")

if not(os.path.isdir(dst)):
    os.makedirs(dst)
dst_fn = os.path.join(dst, "v.nc")

fns = glob.glob(os.path.join(src, "*.tif"))
fns.sort()
print(fns)


u = []
v = []
sig2noise = []
time = []
for n in range(len(fns)-16):
    print(f"Treating frame {n}")
    frame_a = openpiv.tools.imread(fns[n])
    frame_b = openpiv.tools.imread(fns[n+1])
    _u, _v, _sig2noise = openpiv.pyprocess.extended_search_area_piv( frame_a, frame_b, window_size=60, overlap=30, search_area_size=60, dt=1./25)
    # time
    ms = timedelta(milliseconds=int(fns[n][-10:-4]))
    time.append(t + ms)

    u.append(_u)
    v.append(_v)
    sig2noise.append(_sig2noise)

cols, rows = openpiv.pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=60, overlap=30)
var_names = ['u', 'v', 's2n']
var_attrs = [
    {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flow element center velocity vector, x-component",
        "units": "m s-1",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
    {
        "standard_name": "sea_water_y_velocity",
        "long_name": "Flow element center velocity vector, y-component",
        "units": "m s-1",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
    {
        "standard_name": "ratio",
        "long_name": "signal to noise ratio",
        "units": "",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
]
lon_attrs = {
    "long_name": "longitude",
    "units": "degrees_east",
}

lat_attrs = {
    "long_name": "latitude",
    "units": "degrees_north",
}

encoding = {var: {"zlib": True} for var in var_names}

arrays = [
    np.array(u),
    np.array(v),
    np.array(sig2noise),
]

with rasterio.open(fns[0]) as ds:
    print(ds.transform)
    band = ds.read(1)
    shape = band.shape
    coli, rowi = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xi, yi = rasterio.transform.xy(ds.transform, rowi, coli)
    xi, yi = np.array(xi), np.array(yi)
    trans_ = rasterio.Affine(ds.transform[0]*1e6, ds.transform[1]*1e6, ds.transform[2], ds.transform[3]*1e6, ds.transform[4]*1e6, ds.transform[5])
    xq, yq = rasterio.transform.xy(trans_, rows, cols)

    # xq, yq = rasterio.transform.xy(ds.transform, rows, cols)

    xq, yq = np.array(xq), np.array(yq)
    loni, lati = warp.transform(ds.crs, dst_crs, xq.flatten(), yq.flatten())
    loni, lati = np.array(loni).reshape(xq.shape), np.array(lati).reshape(yq.shape)



    # show(ds.read(1), transform=ds.transform)
#
#
#

dataset = merge_outputs(arrays, time, cols[0], rows[:, 0], var_names, var_attrs)

# add lat and lon
lon_da = list_to_dataarray(loni, 'lon', cols[0], rows[:, 0], attrs=lon_attrs)
lat_da = list_to_dataarray(lati, 'lat', cols[0], rows[:, 0], attrs=lat_attrs)
dataset = xr.merge([dataset, lon_da, lat_da])

dataset.to_netcdf(dst_fn, encoding=encoding)

frame_cv = cv2.imread(fns[0])

plt.pcolormesh(xi, yi, band)
plt.imshow(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
plt.quiver(xq, yq, u[0], v[0], color='r')
plt.show()
