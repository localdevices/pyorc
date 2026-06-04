"""Functions for writing results to geographical formats."""

import time
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from pyproj import CRS
from pyproj.enums import WktVersion

from pyorc import __version__ as pyorc_version
from pyorc.helpers import pixel_to_map

UGRID_GLOBAL_ATTRS = {
    "source": "pyOpenRiverCam v" + pyorc_version,
    "date_created": time.ctime(),
    "Conventions": "CF-1.13 UGRID-1.0",
    "title": "Surface velocimetry results from pyOpenRiverCam",
    "history": f"Created by pyOpenRiverCam version {pyorc_version} on {time.ctime()}",
}


# fixed mesh2d structure for UGRID
UGRID_MESH2D = xr.DataArray(
    np.int32(0),
    attrs={
        "cf_role": "mesh_topology",
        "long_name": "Topology data of 2D mesh",
        "topology_dimension": np.int32(2),
        "node_coordinates": "mesh2d_node_x mesh2d_node_y",
        "max_face_nodes_dimension": "mesh2d_nMax_face_nodes",
        "face_node_connectivity": "mesh2d_face_nodes",
        "face_dimension": "mesh2d_nFaces",
        "face_coordinates": "mesh2d_face_x mesh2d_face_y",
    },
)

UGRID_FACE_NODES_ATTRS = {
    "cf_role": "face_node_connectivity",
    "mesh": "mesh2d",
    "location": "face",
    "long_name": "Mapping from every face to its corner nodes (counterclockwise)",
    "start_index": np.int32(0),
    "coordinates": "mesh2d_face_x mesh2d_face_y",
}

UGRID_VAR_ATTRS = var_attrs = {
    "mesh2d_ucx": {
        "mesh": "mesh2d",
        "location": "face",
        "standard_name": "sea_water_x_velocity",
        # 'long_name': 'Flow element center velocity vector, x-component',
        "long_name": "velocity, x-component",
        "units": "m s-1",
        # '_FillValue': -999.,
        "grid_mapping": "projected_coordinate_system",
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    },
    "mesh2d_ucy": {
        "mesh": "mesh2d",
        "location": "face",
        "standard_name": "sea_water_y_velocity",
        "long_name": "velocity, y-component",
        # 'long_name': 'Flow element center velocity vector, y-component',
        "units": "m s-1",
        "grid_mapping": "projected_coordinate_system",
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    },
    "v_s": {
        "mesh": "mesh2d",
        "location": "face",
        "standard_name": "sea_water_speed",
        "long_name": "velocity magnitude",
        "units": "m s-1",
        # '_FillValue': -999.,
        "grid_mapping": "projected_coordinate_system",
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    },
    "s2n": {
        "mesh": "mesh2d",
        "location": "face",
        "standard_name": "noise",
        "long_name": "Signal to noise ratio",
        "units": "-",
        # '_FillValue': -999.,
        "grid_mapping": "projected_coordinate_system",
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    },
    "corr": {
        "mesh": "mesh2d",
        "location": "face",
        "standard_name": "correlation",
        "long_name": "Correlation value",
        "units": "-",
        # '_FillValue': -999.,
        "grid_mapping": "projected_coordinate_system",
        "coordinates": "mesh2d_face_x mesh2d_face_y",
    },
}


def _get_mesh_face_nodes(aff, x, y):
    """Get the face nodes and expected indexes of noides around each face."""
    # get a set of columns and rows for the nodes, so we need 1 more row and column than the amount of faces
    nr_nodes = (len(x) + 1) * (len(y) + 1)
    node_idx = np.arange(nr_nodes).reshape(len(y) + 1, len(x) + 1)
    mesh_face_nodes = np.array(
        [
            node_idx[0:-1, 0:-1].flatten(),
            node_idx[0:-1, 1:].flatten(),
            node_idx[1:, 1:].flatten(),
            node_idx[1:, 0:-1].flatten(),
        ]
    ).swapaxes(0, 1)
    mesh_face_nodes = [list(n) for n in mesh_face_nodes]
    return mesh_face_nodes


def _get_mesh_faces(aff, x, y):
    """Get the faces for a mesh based on the affine transform and row and column lists counts."""
    # get a set of columns and rows for the nodes, so we need 1 more row and column than the amount of faces
    coli, rowi = np.meshgrid(np.arange(len(x)), np.arange(len(y)))
    # use super fast pixel to map function
    face_x, face_y = pixel_to_map(coli, rowi, aff)
    # get the face nodes by taking the average of the corner nodes
    return face_x.flatten(), face_y.flatten()


def _get_mesh_nodes(aff, x, y):
    """Get nodes of a rectangular mesh based on the affine transform and row and column lists counts."""
    # get a set of columns and rows for the nodes, so we need 1 more row and column than the amount of faces
    coli, rowi = np.meshgrid(np.arange(len(x) + 1), np.arange(len(y) + 1))
    # use super fast pixel to map function
    node_x, node_y = pixel_to_map(coli, rowi, aff)
    return node_x.flatten(), node_y.flatten()


def to_geotiff(
    data: np.ndarray, fn: str, transform: Affine, crs: Optional[Union[CRS, str]] = None, compress: Optional[str] = None
):
    """Write a geotiff file with rasterio."""
    # make at least 3D
    data = np.atleast_3d(data)
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    with rasterio.open(
        fn,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=data.shape[2],
        dtype=data.dtype,
        crs=crs.to_proj4() if crs is not None else None,
        transform=transform,
        compress=compress,
    ) as dst:
        # rasterio expects (bands, rows, cols), but we have (rows, cols, bands)
        data = np.transpose(data, (2, 0, 1))
        dst.write(data)


def to_ugrid(
    data_vars: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    aff: Affine,
    crs: Optional[Union[CRS, str]] = None,
    time0: Optional[datetime] = None,
    title: Optional[str] = None,
    fill_na: Optional[float] = None,
):
    """Write a ugrid file with xarray."""
    # make at least 3D
    for d in data_vars:
        if d not in UGRID_VAR_ATTRS:
            raise ValueError(f"Variable {d} is not in known variable keys {list(UGRID_VAR_ATTRS.keys())}")
        # make at least 3d
        data_vars[d] = np.atleast_3d(data_vars[d])
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    elif isinstance(crs, CRS):
        pass
    else:
        crs = None
    # get the mesh face nodes from the data, assuming that the first two dimensions are rows and columns
    mesh_face_nodes = _get_mesh_face_nodes(aff, x, y)
    face_x, face_y = _get_mesh_faces(aff, x, y)
    node_x, node_y = _get_mesh_nodes(aff, x, y)

    # create a start for a variables dict, with mesh2d and projected coordinate system
    variables = {
        "mesh2d": UGRID_MESH2D,
        "mesh2d_face_nodes": (
            ["mesh2d_nFaces", "mesh2d_nMax_face_nodes"],
            np.int32(np.array(mesh_face_nodes)),
            UGRID_FACE_NODES_ATTRS,
        ),
        # "projected_coordinate_system": ds2.projected_coordinate_system
    }
    grid_map_attrs = {}
    if crs:
        # take from rioxarray to ensure compatibility with QGIS and GDAL, which expect a certain format of WKT
        crs_wkt = crs.to_wkt()  # for QGIS / GDAL compatibility
        crs_wkt_gdal = crs.to_wkt(pretty=True, version=WktVersion.WKT1_GDAL)
        grid_map_attrs["spatial_ref"] = crs_wkt
        grid_map_attrs["crs_wkt"] = crs_wkt
        if aff is not None:
            grid_map_attrs["GeoTransform"] = " ".join([str(item) for item in aff.to_gdal()])
        variables["projected_coordinate_system"] = (
            (),
            np.int32(0),
            {"wkt": crs_wkt_gdal},
        )
    # make a mask (first dim just one) with zeros and ones, getting the edge to be zero and internal part one
    shape = data_vars[list(data_vars.keys())[0]].shape[1:3]
    mask = np.zeros(shape)
    mask[1:-1, 1:-1] = 1
    # add a dim for time
    mask = np.expand_dims(mask, axis=0)
    # for var in ds.data_vars:
    for var in data_vars:
        data_var = data_vars[var]

        # apply mask
        data_var *= mask
        # flatten over x and y
        new_shape = (data_var.shape[0], -1)  # keep time dimension, flatten x and y
        data_var = np.reshape(data_var, new_shape).astype(
            np.float32
        )  # np.expand_dims(data_var.flatten().astype(np.float64), axis=0)  # TODO: also try with np,float32.
        # data_var = data_var.flatten().astype(np.float64).reshape(1, len(face_x.flatten()))
        # make internal value zero
        if fill_na is not None:
            data_var[np.isnan(data_var)] = fill_na
        variables[var] = (["time", "mesh2d_nFaces"], data_var, var_attrs[var])

    ds_ugrid = xr.Dataset(
        variables,
        coords={
            "mesh2d_node_x": (
                ["mesh2d_nNodes"],
                np.array(node_x).flatten(),
                {
                    "mesh": "mesh2d",
                    "location": "node",
                    "_FillValue": -999.0,
                    "long_name": "x-coordinate of mesh nodes",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "mesh2d_node_y": (
                ["mesh2d_nNodes"],
                np.array(node_y).flatten(),
                {
                    "mesh": "mesh2d",
                    "location": "node",
                    "_FillValue": -999.0,
                    "long_name": "y-coordinate of mesh nodes",
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
            "mesh2d_face_x": (
                ["mesh2d_nFaces"],
                np.array(face_x).flatten(),
                {
                    "mesh": "mesh2d",
                    "location": "face",
                    "_FillValue": -999.0,
                    "long_name": "x-coordinate of mesh faces",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "mesh2d_face_y": (
                ["mesh2d_nFaces"],
                np.array(face_y).flatten(),
                {
                    "mesh": "mesh2d",
                    "location": "face",
                    "_FillValue": -999.0,
                    "long_name": "y-coordinate of mesh faces",
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
            "time": (
                ["time"],
                time,
                {"long_name": "time", "standard_name": "time", "units": "seconds since 1970-01-01T00:00:00Z"},
            ),
        },
        attrs=UGRID_GLOBAL_ATTRS,
    )
    ENCODING_PARAMS = {
        "zlib": True,
        "_FillValue": -9999.0,
    }

    # set encoding pars
    for k in data_vars:
        ds_ugrid[k].encoding = ENCODING_PARAMS
        # ds_ugrid[k] = ds_ugrid[k].rio.write_crs(27700)  # # TODO replace by normal xarray instead of rioxarray

    ds_ugrid["mesh2d_face_nodes"].encoding = {"_FillValue": -999}
    return ds_ugrid
