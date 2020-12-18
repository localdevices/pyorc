import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate

def _corr_color(img, alpha=None, beta=None, gamma=0.5):
    """
    Grey scaling, contrast- and gamma correction. Both alpha and beta need to be
    defined in order to apply contrast correction.

    Input:
    ------
    :param img: 3D cv2 img object
    :param alpha=None: float - gain parameter for contrast correction)
    :param beta=None: bias parameter for contrast correction
    :param gamma=0.5 brightness parameter for gamma correction (default: 0.5)
    :return img 2D gray scale
    Output:
    -------
    return: img gray scaled, contrast- and gamma corrected image
    """
    # turn image into grey scale
    corr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if alpha and beta:
        # apply contrast correction
        corr_img = cv2.convertScaleAbs(corr_img, alpha=alpha, beta=beta)

    # apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    corr_img = cv2.LUT(corr_img, table)

    return corr_img

def _corr_lens(img, k1=0.0, c=2.0, f=1.0):
    """
    Lens distortion correction based on lens characteristics.
    Function by Gerben Gerritsen / Sten Schurer, 2019

    :param img:  3D cv2 img matrix
    :param k1=0.: float - barrel lens distortion parameter
    :param c=2.: float - optical center
    :param f=1.: float - focal length
    :return undistorted img
    """

    # define imagery characteristics
    height, width, __ = img.shape

    # define distortion coefficient vector
    dist = np.zeros((4, 1), np.float64)
    dist[0, 0] = k1

    # define camera matrix
    mtx = np.eye(3, dtype=np.float32)

    mtx[0, 2] = width / c  # define center x
    mtx[1, 2] = height / c  # define center y
    mtx[0, 0] = f  # define focal length x
    mtx[1, 1] = f  # define focal length y

    # correct image for lens distortion
    corr_img = cv2.undistort(img, mtx, dist)
    return corr_img

def _get_shape(bbox, resolution=0.01, round=1):
    coords = bbox.exterior.coords
    box_length = LineString(coords[0:2]).length
    box_width = LineString(coords[1:3]).length
    cols = int(np.ceil((box_length / resolution) / round)) * round
    rows = int(np.ceil((box_width / resolution) / round)) * round
    return cols, rows

def _get_transform(bbox, resolution=0.01):
    """
    return a rotated Affine transformation that fits with the bounding box and resolution
    :param bbox: shapely Polygon, polygon of bounding box. The coordinate order is very important and has to be:
        (upstream-left, downstream-left, downstream-right, upstream-right, upstream-left)
    :param res=0.01: float, resolution of target grid in meters
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


def _get_gcps_a(lensPosition, h_a, coords, z_0=0., h_ref=0.):
    """
    Get the actual x, y locations of ground control points at the actual water level

    :param lensPosition: list with [x, y, z], location of cam in local crs [m]
    :param h_a: float - actual water level in local level measuring system [m]
    :param coords: list of lists [x, y] with gcp coordinates in original water level
    :param z_0: reference zero plain level, i.e. the crs amount of meters of the zero level of staff gauge
    :param h_ref: reference water level during taking of gcp coords with ref to staff gauge zero level
    :return: coords, where dst is replaced by new dst with actual water level

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


def _get_M(src, dst):
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
    _src = np.float32(src)
    _dst = np.float32(dst)
    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(_src, _dst)
    return M
    # find locations of transformed image corners


def _transform_to_bbox(coords, bbox, res):
    """
    transforms a set of coordinates defined in crs of bbox, into a set of coordinates in cv2 compatible pixels
    """
    # first assemble x and y coordinates
    xs, ys = zip(*coords)
    transform = _get_transform(bbox, res)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    return list(zip(cols, rows))


def orthorectification(img, lensPosition, h_a, src, dst, z_0, h_ref, bbox, resolution=0.01, round=1):
    """
    This function takes the original gcps and water level, and uses the actual water level, defined resolution
    and AOI to determine a resulting outcoming image. Image orthorectification parameters based on 4 GCPs.
    GCPs need to be at water level.

    :return:
    raterio ds
    """
    # compute the geographical location of the gcps with the actual water level (h_a)
    dst_a = _get_gcps_a(
        lensPosition, h_a, dst, z_0, h_ref,
    )

    dst_colrow_a = _transform_to_bbox(dst_a, bbox, resolution)

    # retrieve M for destination row and col
    M = _get_M(src=src, dst=dst_colrow_a)
    # estimate size of required grid
    transform = _get_transform(bbox, resolution=resolution)
    # TODO: alter method to determine window_size based on how PIV is done. If only squares are possible, then this can be one single nr.
    cols, rows = _get_shape(bbox, resolution=resolution, round=10)  # for now hard -coded on 10, alter dependent on how PIV is done
    corr_img = cv2.warpPerspective(img, M, (cols, rows))
    return corr_img, transform


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
    M_gcp = _get_M(src=src, dst=dst)
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
    bbox = rotate(bbox, angle, origin=tuple(_dst_corners[0]), use_radians=True)
    return bbox

def surf_velocity():
    # FIXME
    raise NotImplementedError("")


def river_flow():
    # FIXME
    raise NotImplementedError("")


