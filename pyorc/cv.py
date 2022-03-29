import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate


def _corr_color(img, alpha=None, beta=None, gamma=0.5):
    """
    Grey scaling, contrast- and gamma correction. Both alpha and beta need to be
    defined in order to apply contrast correction.

    :param img: np.ndarray, 3D cv2 img object
    :param alpha=None: float, gain parameter for contrast correction)
    :param beta=None: float, bias parameter for contrast correction
    :param gamma=0.5 float, brightness parameter for gamma correction (default: 0.5)
    :return img: np.ndarray, 2D gray scale
    """

    # turn image into grey scale
    corr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if alpha and beta:
        # apply contrast correction
        corr_img = cv2.convertScaleAbs(corr_img, alpha=alpha, beta=beta)

    # apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    corr_img = cv2.LUT(corr_img, table)

    return corr_img


def _get_dist_coefs(k1):
    """
    Establish distance coefficient matrix for use in cv2.undistort

    :param k1: barrel lens distortion parameter
    :return: distance coefficient matrix (4 parameter)
    """
    # define distortion coefficient vector
    dist = np.zeros((4, 1), np.float64)
    dist[0, 0] = k1
    return dist


def _get_cam_mtx(height, width, c=2.0, f=1.0):
    """
    Get 3x3 camera matrix from lens parameters

    :param height: height of image from camera
    :param width: width of image from camera
    :param c: float, optical center (default: 2.)
    :param f: float, focal length (default: 1.)
    :return: camera matrix, to be used by cv2.undistort
    """
    # define camera matrix
    mtx = np.eye(3, dtype=np.float32)
    mtx[0, 2] = width / c  # define center x
    mtx[1, 2] = height / c  # define center y
    mtx[0, 0] = f  # define focal length x
    mtx[1, 1] = f  # define focal length y
    return mtx


def get_shape(bbox, resolution=0.01, round=1):
    """
    defines the number of rows and columns needed in a target raster, to fit a given bounding box.

    :param bbox: shapely Polygon, bounding box
    :param resolution: resolution of target raster
    :param round: number of pixels to round intended shape to
    :return: numbers of rows and columns for target raster
    """
    coords = bbox.exterior.coords
    box_length = LineString(coords[0:2]).length
    box_width = LineString(coords[1:3]).length
    cols = int(np.ceil((box_length / resolution) / round)) * round
    rows = int(np.ceil((box_width / resolution) / round)) * round
    return cols, rows


def get_transform(bbox, resolution=0.01):
    """
    return a rotated Affine transformation that fits with the bounding box and resolution.

    :param bbox: shapely Polygon, polygon of bounding box. The coordinate order is very important and has to be:
        (upstream-left, downstream-left, downstream-right, upstream-right, upstream-left)
    :param resolution: float, resolution of target grid in meters (default: 0.01)
    :return: rasterio compatible Affine transformation matrix
    """

    corners = np.array(bbox.exterior.coords)
    # estimate the angle of the bounding box
    top_left_x, top_left_y = corners[0]
    # retrieve average line across AOI
    point1 = corners[0]
    point2 = corners[1]
    diff = point2 - point1
    # compute the angle of the projected bbox area of interest
    angle = np.arctan2(diff[1], diff[0])
    # compute per col the x and y diff
    dx_col, dy_col = np.cos(angle) * resolution, np.sin(angle) * resolution
    # compute per row the x and y offsets
    dx_row, dy_row = (
        np.cos(angle + 1.5 * np.pi) * resolution,
        np.sin(angle + 1.5 * np.pi) * resolution,
    )
    return rasterio.transform.Affine(
        dx_col, dy_col, top_left_x, dx_row, dy_row, top_left_y
    )


def get_gcps_a(lensPosition, h_a, coords, z_0=0.0, h_ref=0.0):
    """
    Get the actual x, y locations of ground control points at the actual water level

    :param lensPosition: list, with [x, y, z], location of cam in local crs [m]
    :param h_a: float, actual water level in local level measuring system [m]
    :param coords: list, containing lists [x, y] with gcp coordinates in original water level
    :param z_0: float, reference zero plain level, i.e. the crs amount of meters of the zero level of staff gauge
    :param h_ref: float, reference water level during taking of gcp coords with ref to staff gauge zero level
    :return: coords, in rows/cols for use in getPerspectivetransform

    """
    # get modified gcps based on camera location and elevation values
    cam_x, cam_y, cam_z = lensPosition
    x, y = zip(*coords)
    # compute the z during gcp coordinates
    z_ref = h_ref + z_0
    # compute z during current frame
    z_a = z_0 + h_a
    # compute the water table to camera height difference during field referencing
    cam_height_ref = cam_z - z_ref
    # compute the actual water table to camera height difference
    cam_height_a = cam_z - z_a
    rel_diff = cam_height_a / cam_height_ref
    # apply the diff on all coordinate, both x, and y directions
    _dest_x = list(cam_x + (np.array(x) - cam_x) * rel_diff)
    _dest_y = list(cam_y + (np.array(y) - cam_y) * rel_diff)
    dest_out = list(zip(_dest_x, _dest_y))
    return dest_out


def get_M(src, dst):
    """
    Retrieve transformation matrix for between (4) src and (4) dst points

    :param src: list of lists [x, y] with source coordinates, typically cols and rows in image
    :param dst: list of lists [x, y] with target coordinates after reprojection, can e.g. be in crs [m]

    :return: transformation matrix, used in cv2.warpPerspective
    """

    # set points to float32
    _src = np.float32(src)
    _dst = np.float32(dst)
    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(_src, _dst)
    return M


def transform_to_bbox(coords, bbox, res):
    """
    transforms a set of coordinates defined in crs of bbox, into a set of coordinates in cv2 compatible pixels
    
    :param coords: list, containing lists [x, y] with coordinates
    :param bbox: shapely Polygon, polygon of bounding box. The coordinate order is very important and has to be:
        (upstream-left, downstream-left, downstream-right, upstream-right, upstream-left)
    :param res: float, resolution of target pixels within bbox
    :return: list, containing tuples of columns and rows
    """
    # first assemble x and y coordinates
    xs, ys = zip(*coords)
    transform = get_transform(bbox, res)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    return list(zip(cols, rows))


def get_ortho(img, M, shape, flags=cv2.INTER_AREA):
    """
    Reproject an image to a given shape using perspective transformation matrix M

    :param img: nd-array, image to transform
    :param M: image perspective transformation matrix
    :param shape: tuple with ints (cols, rows)
    :param flags: cv2.flags to pass with cv2.warpPerspective
    :return: np.ndarray with reprojected data with shape=shape
    """
    if not(isinstance(img, np.ndarray)):
        # load values here
        img = img.values
    return cv2.warpPerspective(img, M, shape, flags=flags)


def get_aoi(src, dst, src_corners):

    """
    Get rectangular AOI from 4 user defined points within frames.

    :param src: list, (col, row) pairs of ground control points
    :param dst: list, projected (x, y) coordinates of ground control points
    :param src_corners: - dict with 4 (x,y) tuples names "up_left", "down_left", "up_right", "down_right"
    :return: shapely Polygon, representing bounding box of aoi (rotated)

    """

    # retrieve the M transformation matrix for the conditions during GCP. These are used to define the AOI so that
    # dst AOI remains the same for any movie
    M_gcp = get_M(src=src, dst=dst)
    # prepare a simple temporary np.array of the src_corners
    try:
        _src_corners = np.array(
            src_corners
        )
    except:
        raise ValueError("src_corner coordinates not having expected format")
    assert(_src_corners.shape==(4, 2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. Current shape is {src_corners.shape}"
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
    bbox = rotate(bbox, angle, origin=tuple(_dst_corners[0]), use_radians=True)
    return bbox


def undistort_img(img, k1=0.0, c=2.0, f=1.0):
    """
    Lens distortion correction of image based on lens characteristics.
    Function by Gerben Gerritsen / Sten Schurer, 2019.

    :param img: np.ndarray, 3D array with image
    :param k1: float, barrel lens distortion parameter (default: 0.)
    :param c: float, optical center (default: 2.)
    :param f: float, focal length (default: 1.)
    :return undistorted img
    """

    # define imagery characteristics
    height, width, __ = img.shape
    dist = _get_dist_coefs(k1)

    # get camera matrix
    mtx = _get_cam_mtx(height, width, c=c, f=f)

    # correct image for lens distortion
    corr_img = cv2.undistort(img, mtx, dist)
    return corr_img


def undistort_points(points, height, width, k1=0.0, c=2.0, f=1.0):
    """
    Undistorts x, y point locations with provided lens parameters, so that points
    can be undistorted together with images from that lens.

    :param points: list, containing lists of points [x, y], provided as float
    :param height: int, height of camera images [nr. of pixels]
    :param width: int, width of camera images [nr. of pixels]
    :param k1: float, barrel lens distortion parameter (default: 0.)
    :param c: float, optical center (default: 2.)
    :param f: float, focal length (default: 1.)
    :return: list, containg lists of undistorted point coordinates [x, y] as floats
    """
    mtx = _get_cam_mtx(height, width, c=c, f=f)
    dist = _get_dist_coefs(k1)
    points_undistort = cv2.undistortPoints(
        np.expand_dims(np.float32(points), axis=1),
        mtx,
        dist,
        P=mtx
    )
    return points_undistort[:, 0].tolist()

