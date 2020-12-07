import cv2
import numpy as np
import copy
import geojson

from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate

from pyproj import CRS


def _get_gcps_a(gcps, cam_loc, h_a):
    """
    Get the actual x, y locations of ground control points at the actual water level

    :param gcps:  Ground control points containing x, y, z_0 (zero water level in crs [m]) and h_ref (water level during measuring campaign)
    :param cam_loc: dict with "x", "y" and "z", location of cam in local crs [m]
    :param h_a: float - actual water level in local level measuring system [m]
    :return: gcps, where dst is replaced by new dst with actual water level

    """
    # get modified gcps based on camera location and elevation values
    dest_x, dest_y = zip(*gcps["dst"])
    z_ref = gcps["h_ref"] + gcps["z_0"]
    z_a = gcps["z_0"] + h_a
    # compute the water table to camera height difference during field referencing
    cam_height_ref = cam_loc["z"] - z_ref
    # compute the actual water table to camera height difference
    cam_height_a = cam_loc["z"] - z_a
    rel_diff = cam_height_a / cam_height_ref
    # apply the diff on all coordinate, both x, and y directions
    _dest_x = list(cam_loc["x"] + (np.array(dest_x) - cam_loc["x"]) * rel_diff)
    _dest_y = list(cam_loc["y"] + (np.array(dest_y) - cam_loc["y"]) * rel_diff)
    gcps_out = copy.deepcopy(gcps)
    gcps_out["dst"] = list(zip(_dest_x, _dest_y))
    return gcps_out


def _get_M(gcps):
    """
    Image orthorectification parameters based on 4 GCPs.
    GCPs need to be at water level.

    Input:
    ------
    img - original image
    gcps - Dict containing in "src" a list of (col, row) pairs and in "dst" a list of projected (x, y) coordinates
        of the GCPs in the imagery

    Output:
    -------
    Transformation matrix based on image corners
    """

    # # set points to float32
    # pts1 = np.float32(df_from.values)
    # pts2 = np.float32(df_to.values * PPM)
    _src = np.float32(gcps["src"])
    _dst = np.float32(gcps["dst"])
    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(_src, _dst)
    return M
    # find locations of transformed image corners


def orthorectification(img, aoi, dst_resolution=0.01):
    """
    This function takes the original gcps and water level, and uses the actual water level, defined resolution
    and AOI to determine a resulting outcoming image. Image orthorectification parameters based on 4 GCPs.
    GCPs need to be at water level.

    :return:
    BytesIO object or written GeoTiff file
    """
    raise NotImplementedError("Implement me")


def get_aoi(gcps, src_corners, crs=None):
    """
    Get rectangular AOI from 4 user defined points within frames.


    Input:
    ------
    gcps - Dict containing in "src" a list of (col, row) pairs and in "dst" a list of projected (x, y) coordinates
        of the GCPs in the imagery
    src_corners - dict with 4 (x,y) tuples names "up_left", "down_left", "up_right", "down_right"
    crs=None - str, project coordinate reference system as "EPSG:XXXX", if available, results are stored into this crs in GeoTiff

    Output:
    -------
    GeoJSON - AOI

    Transformation matrix based on image corners
    """

    # retrieve the M transformation matrix for the conditions during GCP. These are used to define the AOI so that
    # dst AOI remains the same for any movie
    M_gcp = _get_M(gcps)
    # prepare a simple temporary np.array of the src_corners
    try:
        _src_corners = np.array(
            [
                src_corners["up_left"],
                src_corners["down_left"],
                src_corners["down_right"],
                src_corners["up_right"],
            ]
        )
    except:
        raise ValueError("src_corner coordinates not having expected format")
    # reproject corner points to the actual space in coordinates
    _dst_corners = cv2.perspectiveTransform(np.float32([_src_corners]), M_gcp)[0]
    polygon = Polygon(_dst_corners)
    coords = np.array(polygon.exterior.coords)
    # estimate the angle of the bounding box
    # retrieve average line across AOI
    point1 = (coords[0] + coords[3]) / 2
    point2 = (coords[1] + coords[2]) / 2
    diff = point2 - point1
    angle = np.arctan2(diff[1], diff[0])
    # rotate the polygon over this angle to get a proper bounding box
    polygon_rotate = rotate(
        polygon, -angle, origin=tuple(_dst_corners[0]), use_radians=True
    )
    xmin, ymin, xmax, ymax = polygon_rotate.bounds
    bbox_coords = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin), (xmin, ymax)]
    bbox = Polygon(bbox_coords)
    # now rotate back
    aoi = rotate(bbox, angle, origin=tuple(_dst_corners[0]), use_radians=True)

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
    crs_json = {"type": "EPSG", "properties": {"code": epsg}}
    f = geojson.Feature(geometry=aoi, properties={"ID": 0})
    return geojson.FeatureCollection([f], crs=crs_json)


def surf_velocity():
    # FIXME
    raise NotImplementedError("")


def river_flow():
    # FIXME
    raise NotImplementedError("")
