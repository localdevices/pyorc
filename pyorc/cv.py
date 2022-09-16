import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate
from tqdm import tqdm
from scipy.cluster.vq import vq, kmeans
import operator

def _classify_displacements(positions, method="kmeans", q_threshold=0.8, abs_threshold=None, op=operator.le):
    """
    Classifies set of displacements of points through time in two groups based on a difference measure. Can e.g. be used
    to mask values that are not moving enough (in case one wishes to detect water) or values that move too much
    (in case one wishes to detect rigid points for image stabilization).

    Parameters
    ----------
    positions : list of arrays
        time sequences of x,y locations per points
    method : str, optional
        method to filter or split points into population. Currently implemented are "kmeans" (split using a simple
        kmeans classification with two assumed groups), "std" (split using a standard deviation criterium), or "dist"
        (using an absolute distance in pixels).
    q_threshold: float (0-1), optional
        tolerance percentile used for either the "dist" or "std" method (i.e. points that move more or less than the
        distance, measured as provided quantile, based on all points in the population are in either one or the other
        group). Default: 0.8 meaning that points that have a standard deviation smaller than (larger or equal than, if
        other operator is selected) the 0.8 quantile of all standard deviations are returned.
    abs_threshold: float (0-1), optional
        tolerance absolute value used for either the "dist" or "std" method (i.e. points that move more or less than the
        distance, measured as provided absolute threshold, based on all points in the population are in either one or
        the other group). Overrules q_threshold when set.
    op: operator, optional
        type of operation to test point values (default: operator.ge)

    Returns
    -------
    filter: np.ndarray 1D [length positions]
        Boolean per position, defining if a point filters out as True or False given the set criterion.
        time sequences of x,y locations per points, after filtering
    """
    assert(method in ["kmeans", "std", "dist"]), f'Method must be "kmeans", "std" or "dist", but instead is {method}.'
    if q_threshold is not None:
        assert (0.99 > q_threshold > 0.01), \
            f'q_threshold represents a quantile and must be between 0.01 and 0.99, {q_threshold} given '
    if method in ["kmeans", "std"]:
        test_variable = positions.std(axis=0).mean(axis=-1)
    elif method == "dist":
        distance_xy = positions[-1] - positions[0]
        test_variable = (distance_xy[:, 0]**2 + distance_xy[:, 1]**2)**0.5
    if method == "kmeans":
        centroids, mean_value = kmeans(test_variable, 2)
        clusters, distances = vq(test_variable, np.sort(centroids))
        return clusters == 0
    # if not kmeans, then follow the same route for "dist" or "std"
    # derive tolerance quantile
    if abs_threshold is None:
        # tolerance from quantile in distribution
        tolerance = np.quantile(test_variable, q_threshold)  # PARAMETER
    else:
        # tolerance as absolute value
        tolerance = abs_threshold
    return op(test_variable, tolerance)


def _convert_edge(img, stride_1, stride_2):
    """
    internal function to do emphasize gradients with a band filter method, see main method
    """
    blur1 = cv2.GaussianBlur(img.astype("float32"), (stride_1, stride_1), 0)
    blur2 = cv2.GaussianBlur(img.astype("float32"), (stride_2, stride_2), 0)
    edges = blur2 - blur1
    return edges


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

def _get_displacements(cap, start_frame=0, end_frame=None, n_pts=None):
    """
    compute displacements from trackable features found in start frame

    Parameters
    ----------
    cap : cv2.Capture object
        video object, opened with cv2
    start_frame : int, optional
        first frame to perform point displacement analysis (default : 0)
    end_frame : int, optional
        last frame to process (must be larger than start_frame). Default: None, meaning the last frame in the video will
        be used).
    n_pts : int, optional
        Number of features to track. If not set, the square root of the amount of pixels of the frames will be used

    Returns
    -------
    positions : np.ndarray [M x N x 2]
        positions of the points from frame to frame with M the amount of frames, N the amount of points, and 2 the x, y
        coordinates
    status : np.ndarray [M x N]
        status of tracking of points, normally 1 is expected, 0 means that tracking for point in given frame did not
         yield results (see also
         https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)

    """
    # set end_frame to last if not defined
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get start frame and points
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # prepare outputs
    n_frames = end_frame - start_frame
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    if n_pts is None:
        # use the square root of nr of pixels in a frame to decide on n_pts
        n_pts = int(np.sqrt(len(prev_gray.flatten())))

    # prepare storage for points
    positions = np.zeros((0, n_pts, 2), np.float32)
    stats = np.ones((1, n_pts))
    errs = np.ones((1, n_pts))

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=n_pts,
        qualityLevel=0.1,
        minDistance=10,
        blockSize=3
    )
    # add start point to matrix
    positions = np.append(positions, np.swapaxes(prev_pts, 0, 1), axis=0)
    # loop through start to end frame
    pbar = tqdm(range(n_frames - 1))
    for i in pbar:
        # Read next frame
        pbar.set_description(f"Stabilizing frames")
        success, curr = cap.read()
        if not success:
            raise IOError(f"Could not read frame {start_frame + i} from video")

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # store curr_pts
        positions = np.append(positions, np.swapaxes(curr_pts, 0, 1), axis=0)
        stats = np.append(stats, np.swapaxes(status, 0, 1), axis=0)
        errs = np.append(errs, np.swapaxes(err, 0, 1), axis=0)
        # prepare next frame
        prev_gray = curr_gray
        prev_pts = curr_pts
    return positions, stats, errs


def _get_shape(bbox, resolution=0.01, round=1):
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


def _get_transform(bbox, resolution=0.01):
    """Return a rotated Affine transformation that fits with the bounding box and resolution.

    Parameters
    ----------
    bbox : shapely.geometry.Polygon
        polygon of bounding box. The coordinate order is very important and has to be:
        (upstream-left, downstream-left, downstream-right, upstream-right, upstream-left)
    resolution : float, optional
        resolution of target grid in meters (default: 0.01)

    Returns
    -------
    affine : rasterio.transform.Affine
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


def _get_gcps_a(lensPosition, h_a, coords, z_0=0.0, h_ref=0.0):
    """Get the actual x, y locations of ground control points at the actual water level

    Parameters
    ----------
    lensPosition : list of floats
        x, y, z location of cam in local crs [m]
    h_a : float
        actual water level in local level measuring system [m]
    coords : list of lists
        gcp coordinates  [x, y] in original water level
    z_0 : float, optional
        reference zero plain level, i.e. the crs amount of meters of the zero level of staff gauge (default: 0.0)
    h_ref : float, optional
        reference water level during taking of gcp coords with ref to staff gauge zero level (default: 0.0)

    Returns
    -------
    coords : list
        rows/cols for use in getPerspectivetransform

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


def m_from_displacement(p1, p2, status):
    """
    Calculate transform from pair of point locations, derived from Lukas Kanade optical flow.
    The accompanying status array (see
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323 is used to only
    select points that were found in the optical flow algorithm.

    Parameters
    ----------
    p1 : np.ndarray [n x 2]
        point locations
    p2 : np.ndarray [n x 2]
        point locations (same as p1, but possibly displaced or rotated)

    Returns
    -------
    m : affine matrix derived for 2D affine transform. Can be used with cv2.warpAffine

    """
    # remove any points that have a status zero according to optical flow
    p1 = p1[status == 1]
    p2 = p2[status == 1]
    # add dim in the middle to match cv2 required array shape
    prev_pts = np.float64(np.expand_dims(p1, 1))
    curr_pts = np.float64(np.expand_dims(p2, 1))
    return cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]


def _ms_from_displacements(pts, stats, key_point=0, smooth=True):
    """
    Computes all transform matrices from list of point locations, found in frames, that are possibly moving.
    The function returns transformation matrices that transform the position of all frames to one single frame (default
    is the first frame)

    Parameters
    ----------
    pts : np.ndarray [t x n x 2]
        Location of traceable rigid body corner points (e.g. detected with good features to track and traced with
        Lukas Kanade optical flow) through time
        with t time steps, for n number of points, row column coordinates
    stats : np.ndarray [t x n]
        status of resolving optical flow (zero: not resolved, one: resolved, see
        https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
    center_frame : int, optional

    Returns
    -------
    ms : list
        Contains affine transform matrix for each set of points. This list can be used to affine transform each
        frame to match as closely as possible the frame chosen as central frame.

    """
    assert key_point >= 0 and key_point < len(pts), f"Key point {int(key_point)} must be within range of point locations (0 - {len(pts) - 1}."
    # TODO: approach to remove points before going to transformation is not working properly yet.
    # # remove points that at some point disappear, by finding which points have status that sometimes is zero
    # idx = stats.min(axis=0) == 0
    # pts = pts[:, idx, :]
    # stats = stats[:, idx]

    ms = [m_from_displacement(p2, pts[int(key_point)], status) for p2, status in zip(pts, stats)]
    if smooth:
        ms = _ms_smooth(ms)
    return ms

def _ms_smooth(ms, q=98):
    def fill_1d(yp, xp):
        idx = np.isfinite(yp)
        yp_sel = yp[idx]
        xp_sel = xp[idx]
        return np.interp(xp, xp_sel, yp_sel)

    ms = np.array(ms).reshape(len(ms), 6)

    dms = np.diff(ms, axis=0)
    tol = np.percentile(np.abs(dms), q, axis=0)

    # remove values beyond the quyantile
    dms[np.abs(dms) > tol] = np.nan

    # also remove when one of the signals is nan
    dms[np.isnan(dms.sum(axis=-1))] = np.nan

    # fill the removed values with linear interpolated values
    dms_fill = np.apply_along_axis(fill_1d, 0, dms, np.arange(len(dms)))
    dms_fill = np.append(np.expand_dims(ms[0], 0), dms_fill, axis=0)
    ms_out = list(np.cumsum(dms_fill, axis=0).reshape(len(ms), 2, 3))
    return ms_out


def _transform(img, m):
    """
    Affine transforms an image using a specified affine transform matrix. Typically the transformation is derived
    for image stabilization purposes.

    Parameters
    ----------
    img : np.ndarray 2D [MxN] or 3D [MxNx3] if RGB image
        Image used as input
    m : np.ndarray [2x3]
        Affine transformation

    Returns
    -------
    img_transform : np.ndarray 2D [MxN] or 3D [MxNx3] if RGB image
        Image after affine transform applied

    """

    h = img.shape[0]
    w = img.shape[1]
    # Apply affine wrapping to the given frame
    img_transform = cv2.warpAffine(img, m, (w, h))
    return img_transform


def get_M(src, dst):
    """Retrieve transformation matrix for between (4) src and (4) dst points

    Parameters
    ----------
    src : list of lists
        [x, y] with source coordinates, typically cols and rows in image
    dst : list of lists
        [x, y] with target coordinates after reprojection, can e.g. be in crs [m]

    Returns
    -------
    M : np.ndarray
        transformation matrix, used in cv2.warpPerspective
    """
    # set points to float32
    _src = np.float32(src)
    _dst = np.float32(dst)
    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(_src, _dst)
    return M


def transform_to_bbox(coords, bbox, res):
    """transforms a set of coordinates defined in crs of bbox, into a set of coordinates in cv2 compatible pixels

    Parameters
    ----------
    coords : list of lists
        [x, y] with coordinates
    bbox : shapely Polygon
        Bounding box. The coordinate order is very important and has to be upstream-left, downstream-left,
        downstream-right, upstream-right, upstream-left
    res : float
        resolution of target pixels within bbox

    Returns
    -------
    colrows : list
        tuples of columns and rows

    """
    # first assemble x and y coordinates
    xs, ys = zip(*coords)
    transform = _get_transform(bbox, res)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    return list(zip(cols, rows))


def get_ortho(img, M, shape, flags=cv2.INTER_AREA):
    """Reproject an image to a given shape using perspective transformation matrix M

    Parameters
    ----------

    img: np.ndarray
        image to transform
    M: np.ndarray
        image perspective transformation matrix
    shape: tuple of ints
        (cols, rows)
    flags: cv2.flags
        passed with cv2.warpPerspective

    Returns
    -------
        img : np.ndarray
            reprojected data with shape=shape
    """
    return cv2.warpPerspective(img, M, shape, flags=flags)


def get_aoi(src, dst, src_corners):
    """Get rectangular AOI from 4 user defined points within frames.

    Parameters
    ----------
    src : list of tuples
        (col, row) pairs of ground control points
    dst : list of tuples
        projected (x, y) coordinates of ground control points
    src_corners : dict with 4 (x,y) tuples
        names "up_left", "down_left", "up_right", "down_right", source corners

    Returns
    -------
    aoi : shapely.geometry.Polygon
        bounding box of aoi (rotated)
    """
    # retrieve the M transformation matrix for the conditions during GCP. These are used to define the AOI so that
    # dst AOI remains the same for any movie
    M_gcp = get_M(src=src, dst=dst)
    # prepare a simple temporary np.array of the src_corners
    _src_corners = np.array(src_corners)
    assert(_src_corners.shape==(4, 2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. " \
                                        f"Current shape is {src_corners.shape} "
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
    """Lens distortion correction of image based on lens characteristics.
    Function by Gerben Gerritsen / Sten Schurer, 2019.

    Parameters
    ----------
    img : np.ndarray
        3D array with image
    k1: float, optional
        barrel lens distortion parameter (default: 0.)
    c: float, optional
        optical center (default: 2.)
    f: float, optional
        focal length (default: 1.)

    Returns
    -------
    img: np.ndarray
        undistorted img
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
    """Undistorts x, y point locations with provided lens parameters, so that points
    can be undistorted together with images from that lens.

    Parameters
    ----------
    points : list of lists
        points [x, y], provided as float
    height: int
        height of camera images [nr. of pixels]
    width: int
        width of camera images [nr. of pixels]
    k1: float, optional
        barrel lens distortion parameter (default: 0.)
    c: float, optional
        optical center (default: 2.)
    f: float, optional
        focal length (default: 1.)

    Returns
    -------
    points : list of lists
        undistorted point coordinates [x, y] as floats
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

