"""OpenCV functions for pyorc."""

import copy
import os
import warnings

import cv2
import numpy as np
import rasterio
from scipy import optimize
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, Polygon
from tqdm import tqdm

from . import helpers

# default distortion coefficients for no distortion
DIST_COEFFS = [[0.0], [0.0], [0.0], [0.0], [0.0]]

# criteria for finding subpix corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _check_valid_frames(cap, frame_number):
    """Determine last valid index in the frame sequence of a video capture object.

    Determines where at the end of the sequence a frame cannot be retrieved due to some error.
    The function iterates backward through the frame numbers and sets the frame position to each one, attempting to
    read the frame. If a frame cannot be read, it updates the `last_valid_idx`
    to that position. If all frames are invalid, it defaults the index to zero.

    Parameters
    ----------
    cap : cv2.VideoCapture
        capture object.
    frame_number : list[int]
        A list representing the sequence of frame numbers to check for validity.

    Returns
    -------
    int or None
        The last valid index in the frame sequence or `None` if no invalid frames
        are found at the end of the sequence.

    """
    # start with None
    last_valid_idx = None
    ret = False
    n = -1
    while ret is False:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number[n])
        ret, img = cap.read()
        if not (ret):
            last_valid_idx = n
        n -= 1
        if n == -len(frame_number) - 1:
            last_valid_idx = 0
            break

    return last_valid_idx


def _combine_m(m1, m2):
    """Combine two 2x3 serial transformation matrices into one.

    Done by extending them to 3x3 matrices for matrix multiplication, compute the product, and return the resulting
    2x3 matrix.

    Parameters
    ----------
    m1 : array_like
        First matrix to be combined, should be of shape (2, 3).

    m2 : array_like
        Second matrix to be combined, should be of shape (2, 3).

    Returns
    -------
    np.ndarray
        The resulting matrix of shape (2, 3) after matrix
        multiplication.

    """
    # extend to a 3x3 for matrix multiplication
    _m1 = np.append(m1, np.array([[0.0, 0.0, 1.0]]), axis=0)
    _m2 = np.append(m2, np.array([[0.0, 0.0, 1.0]]), axis=0)
    m_combi = _m1.dot(_m2)[0:2]
    return m_combi


def _get_aoi_corners(dst_corners, resolution=None):
    polygon = Polygon(dst_corners)
    coords = np.array(polygon.exterior.coords)
    # estimate the angle of the bounding box
    # retrieve average line across AOI
    point1 = (coords[0] + coords[3]) / 2
    point2 = (coords[1] + coords[2]) / 2
    diff = point2 - point1
    angle = np.arctan2(diff[1], diff[0])
    # rotate the polygon over this angle to get a proper bounding box
    polygon_rotate = rotate(polygon, -angle, origin=tuple(dst_corners[0]), use_radians=True)

    xmin, ymin, xmax, ymax = polygon_rotate.bounds
    if resolution is not None:
        xmin = helpers.round_to_multiple(xmin, resolution)
        xmax = helpers.round_to_multiple(xmax, resolution)
        ymin = helpers.round_to_multiple(ymin, resolution)
        ymax = helpers.round_to_multiple(ymax, resolution)

    bbox_coords = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin), (xmin, ymax)]
    bbox = Polygon(bbox_coords)
    # now rotate back
    bbox = rotate(bbox, angle, origin=tuple(dst_corners[0]), use_radians=True)
    return bbox


def _get_aoi_width_length(dst_corners):
    points = [Point(x, y) for x, y, _ in dst_corners]
    linecross = LineString([points[0], points[1]])
    # linecross = LineString(dst_corners[0:2])
    length = np.abs(_get_perpendicular_distance(points[-1], linecross))
    point1 = np.array(dst_corners[0][0:2])
    point2 = np.array(dst_corners[1][0:2])
    diff = np.array(point2 - point1)
    angle = np.arctan2(diff[1], diff[0])

    # compute xy distance from line to other line making up the bounding box
    xy_diff = np.array([np.sin(-angle) * length, np.cos(angle) * length])
    points_pol = np.array([point1 - xy_diff, point1 + xy_diff, point2 + xy_diff, point2 - xy_diff])
    # always make sure the order of the points of upstream-left, downstream-left, downstream-right, upstream-right
    # if length <= 0:
    #     # negative length means the selected length is selected upstream of left-right cross section
    #     points_pol = np.array([point1 + xy_diff, point1, point2, point2 + xy_diff])
    # else:
    #     # postive means it is selected downstream of left-right cross section
    #     points_pol = np.array([point1, point1 + xy_diff, point2 + xy_diff, point2])

    return Polygon(points_pol)


def _smooth(img, stride):
    """Blur image through gaussian smoothing.

    Parameters
    ----------
    img: np.ndarray
        Input image.
    stride: int
        Size of the kernel used for the Gaussian blur.

    Returns
    -------
    np.ndarray
        blurred image.

    """
    blur = cv2.GaussianBlur(img.astype("float32"), (stride, stride), 0)
    return blur


def _convert_edge(img, stride_1, stride_2):
    """Enhance image gradients by a band filter.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    stride_1 : int
        Size of the kernel used for the first Gaussian blur.
    stride_2 : int
        Size of the kernel used for the second Gaussian blur.

    Returns
    -------
    np.ndarray
        Edge enhanced image.

    """
    blur1 = cv2.GaussianBlur(img.astype("float32"), (stride_1, stride_1), 0)
    blur2 = cv2.GaussianBlur(img.astype("float32"), (stride_2, stride_2), 0)
    edges = blur2 - blur1
    return edges


def _get_dist_coefs(k1):
    """Convert barrel distortion parameter into distortion coefficient matrix.

    Parameters
    ----------
    k1 : float
        barrel lens distortion parameter

    Returns
    -------
    np.ndarray
        distortion coefficient matrix (4 parameter)

    """
    # define distortion coefficient vector
    dist = np.zeros((4, 1), np.float64)
    dist[0, 0] = k1
    return dist


def _get_perpendicular_distance(point, line):
    """Calculate perpendicular distance from point to line.

    Line is extended if perpendicular distance is larger than the distance to the endpoints.

    Parameters
    ----------
    point : shapely.geometry.Point
        x, y coordinates of point
    line : shapely.geometry.LineString
        line to calculate distance to

    Returns
    -------
    float
        perpendicular distance from point to line

    """
    # Get coordinates of line endpoints
    p1 = np.array(line.coords[0])
    p2 = np.array(line.coords[1])
    # Convert point to numpy array
    p3 = np.array(point.coords[0])

    # Calculate line vector
    line_vector = p2 - p1
    # Calculate vector from point to line start
    point_vector = p3 - p1

    # Calculate unit vector of line
    unit_line = line_vector / np.linalg.norm(line_vector)

    # Calculate projection length
    projection_length = np.dot(point_vector, unit_line)

    # Calculate perpendicular vector
    perpendicular_vector = point_vector - projection_length * unit_line
    perpendicular_distance = np.linalg.norm(perpendicular_vector)

    # Use cross product to calculate side
    cross_product = np.cross(line_vector, point_vector)

    # Determine the sign of the perpendicular distance
    return perpendicular_distance if cross_product > 0 else -perpendicular_distance


def get_cam_mtx(height, width, c=2.0, focal_length=None):
    """Compute camera matrix based on the given parameters for height, width, scaling factor, and focal length.

    Parameters
    ----------
    height : int
        The height of the image or frame from which the camera matrix is derived.
    width : int
        The width of the image or frame from which the camera matrix is derived.
    c : float, optional
        The scaling factor used to determine the principal point in the matrix.
        Default value is 2.0.
    focal_length : float, optional
        The focal length to set in the camera matrix. If not provided, defaults
        to the width of the image for both the x and y directions.

    Returns
    -------
    numpy.ndarray
        A 3x3 camera matrix defining the intrinsic parameters of the camera,
        including the principal point and focal length.

    """
    # define camera matrix
    mtx = np.eye(3, dtype=np.float32)
    mtx[0, 2] = width / c  # define center x
    mtx[1, 2] = height / c  # define center y
    if focal_length is None:
        mtx[0, 0] = width  # define focal length x
        mtx[1, 1] = width  # define focal length y
    else:
        mtx[0, 0] = focal_length  # define focal length x
        mtx[1, 1] = focal_length  # define focal length y
    return mtx


def get_ms_gftt(cap, start_frame=0, end_frame=None, n_pts=None, split=2, mask=None, wdw=4, progress=True):
    """Calculate motion smoothing of video frames using Good Features to Track and Lucas-Kanade Optical Flow methods.

    This function processes each frame between `start_frame` and `end_frame` to estimate and smooth affine
    transformations, which indicate motions. The function supports frame splitting for detecting features,
    applying an optional mask, and defines a smoothing window for the transformations.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Video capture object to read frames from.
    start_frame : int, optional
        Index of the starting frame for processing. Defaults to 0.
    end_frame : int, optional
        Index of the ending frame for processing. If not specified, it is set to the last frame.
    n_pts : int, optional
        Number of good features to track within each frame. If None, the square root of
        the number of pixels in a frame is used.
    split : int, optional
        Number of segments to split each frame into for feature detection. Defaults to 2. Applied over both x and y
        direction.
    mask : np.ndarray, optional
        Optional mask to specify regions of interest within the frame for feature detection.
    wdw : int, optional
        Window size for smoothing the affine transformations over time. Defaults to 4.
    progress : bool, optional
        Show progress bar or not. Defaults to True.

    Returns
    -------
    list of ndarray
        A list of affine transformation matrices for each processed frame after smoothing.

    """
    # set end_frame to last if not defined
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_frame is None else end_frame
    # make a start transform which does not change the first frame
    m = np.eye(3)[0:2]
    ms = []
    m_key = copy.deepcopy(m)
    # get start frame and points
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # determine the mount of frames that must be processed
    n_frames = int(end_frame + 1) - int(start_frame)

    # Read first frame
    _, img_key = cap.read()
    # Convert frame to grayscale
    img1 = cv2.cvtColor(img_key, cv2.COLOR_BGR2GRAY)
    img_key = img1

    if n_pts is None:
        # use the square root of nr of pixels in a frame to decide on n_pts
        n_pts = int(np.sqrt(len(img_key.flatten())))

    # get features from first key frame
    prev_pts = _gftt_split(img_key, split, n_pts, mask=mask)

    pbar = tqdm(range(n_frames - 1), position=0, leave=True, disable=not (progress))
    pbar.set_description("Deriving stabilization parameters from second frame onwards")
    for i in pbar:
        ms.append(m)
        _, img2 = cap.read()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(img_key, img2, prev_pts, None)
        m_part = cv2.estimateAffine2D(curr_pts, prev_pts)[0]
        m = _combine_m(m_key, m_part)
        if i % 30 == 0:
            img_key = img1
            prev_pts = _gftt_split(img_key, split, n_pts, mask=mask)
            m_key = copy.deepcopy(m)
        img1 = img2
    # add the very last transformation
    ms.append(m)
    # smooth the affines over time
    ma = np.array(ms)
    for m in range(ma.shape[1]):
        for n in range(ma.shape[2]):
            ma[wdw:-wdw, m, n] = np.convolve(ma[:, m, n], np.ones(wdw * 2 + 1) / (wdw * 2 + 1), mode="valid")
    ms_smooth = list(ma)
    return ms_smooth


def _get_gcps_2_4(src, dst, img_width, img_height):
    """Convert 2 points GCPs to 4 points GCPs in the corners of the image frame.

    Function is used in case a nadir video is provided, where it may be assumed that transformation can be done
    through scaling and translation.

    Parameters
    ----------
    src : list or array-like
        source control points (list of lists)
    dst : list or array-like
        destination control points (list of lists)
    img_width : int
        width of original image frame
    img_height : int
        height of original image frame

    Returns
    -------
    src : list or array-like
        source control points (list of lists) converted to corner points
    dst : list or array-like
        destination control points (list of lists) converted to corner points

    """
    # first reverse the coordinate order of the y-axis
    _src = [[x, img_height - y] for x, y in src]
    # estimate affine transform (only rotation and translation)
    M = cv2.estimateAffinePartial2D(np.array(_src), np.array(dst))
    # complete M with a line indicating no perspective
    M = np.array(M[0].tolist() + [[0, 0, 1]])
    # establish corner coordinates
    corners = [[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]]
    dst = cv2.perspectiveTransform(np.float32([corners]), M)[0].tolist()
    # now get the corners back transformed to the real image coordinates
    src = [[x, img_height - y] for x, y in corners]
    return src, dst


def _get_shape(bbox, resolution=0.01, round=1):
    """Calculate the number of columns and rows based on the dimensions of bounding box and provided resolution.

    Rounding factor can be used to adjust the number of columns and rows to a specified granularity.

    Parameters
    ----------
    bbox : shapely.geometry.Polygon
        Rectangular bounding box of AOI.
    resolution : float, optional
        x and y resolution in meters used to define grid
    round : int, optional
        A rounding factor that allows adjusting the number of columns and rows to be multiples of this factor.

    Returns
    -------
    tuple of int
        A tuple containing the number of columns and rows (cols, rows)
        calculated based on the bounding box dimensions, resolution, and
        rounding factor.

    """
    coords = bbox.exterior.coords
    box_length = LineString(coords[0:2]).length
    box_width = LineString(coords[1:3]).length
    cols = int(np.round((box_length / resolution) / round)) * round
    rows = int(np.round((box_width / resolution) / round)) * round
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
    return rasterio.transform.Affine(dx_col, dy_col, top_left_x, dx_row, dy_row, top_left_y)


def _gftt_split(img, split, n_pts, mask=None):
    # split image in smaller chunks if user wants
    v = 0
    h = 0
    ver_split, hor_split = np.int16(np.ceil(np.array(img.shape) / split))
    pts = np.zeros((0, 1, 2), np.float32)
    while v < img.shape[0]:
        while h < img.shape[1]:
            sub_img = img[v : v + ver_split, h : h + hor_split]
            # get points over several quadrants
            subimg_pts = cv2.goodFeaturesToTrack(
                sub_img,
                mask=mask[v : v + ver_split, h : h + hor_split] if mask is not None else None,
                maxCorners=int(n_pts / split**2),
                qualityLevel=0.3,
                minDistance=10,
                blockSize=1,
            )
            # add offsets for quadrants
            if subimg_pts is not None:
                subimg_pts[:, :, 0] += h
                subimg_pts[:, :, 1] += v
                pts = np.append(pts, subimg_pts, axis=0)
            h += hor_split
        h = 0
        v += ver_split
    return pts


def solvepnp(dst, src, camera_matrix, dist_coeffs):
    """Solve p-n-p problem.

    Wrapper for cv2.SolvePnP with pre-processing of the input data and selection of the correct flags.

    Parameters
    ----------
    src : list of lists
        [x, y] with source coordinates, typically cols and rows in image
    dst : list of lists
        [x, y] (in case of 4 planar points) or [x, y, z] (in case of 6+ 3D points) with target coordinates after
        reprojection, can e.g. be in crs [m]
    camera_matrix : np.ndarray (3x3)
        Camera intrinsic matrix
    dist_coeffs : p.ndarray, optional
        1xN array with distortion coefficients (N = 4, 5 or 8)

    Returns
    -------
    succes : int
        0, 1 for succes or no succes
    rvec : np.array
        rotation vector
    tvec : np.array
        translation vector

    """
    # set points to float32
    _src = np.float64(src)
    _dst = np.float64(dst)

    if len(_dst) == 4:
        flags = cv2.SOLVEPNP_P3P
        # if 4 x, y points are provided, add a column with zeros
        # _dst = np.c_[_dst, np.zeros(4)]
    else:
        flags = cv2.SOLVEPNP_ITERATIVE

    camera_matrix = np.float32(camera_matrix)
    dist_coeffs = np.float32(dist_coeffs)
    # define transformation matrix based on GCPs
    return cv2.solvePnP(_dst, _src, camera_matrix, dist_coeffs, flags=flags)


def transform(img, m):
    """Affine-transform image using specified affine transform matrix.

    Typically the transformation is derived for image stabilization purposes.

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


def calibrate_camera(
    fn,
    chessboard_size=(9, 6),
    max_imgs=30,
    plot=True,
    progress_bar=True,
    criteria=criteria,
    to_file=False,
    frame_limit=None,
    tolerance=None,
):
    """Calculate intrinsic matrix and distortion coefficients.

    This follows recipe from:
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    dir = os.path.split(os.path.abspath(fn))[0]
    cap = cv2.VideoCapture(fn)
    # make a list of logical frames in order to read
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_list = helpers.staggered_index(start=0, end=frames_count - 1)

    # set the expected object points from the chessboard size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

    obj_pts = []
    img_pts = []
    imgs = []

    ret_img, img = cap.read()
    frame_size = img.shape[1], img.shape[0]
    if frame_limit is not None:
        frames_list = frames_list[0:frame_limit]
    if progress_bar:
        frames_list = tqdm(frames_list, position=0, leave=True)
        # pbar.update(1)
    for f in frames_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret_img, img = cap.read()
        if ret_img:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags=cv2.CALIB_CB_FAST_CHECK)
            if ret:
                # append the expected point coordinates
                obj_pts.append(objp)
                imgs.append(copy.deepcopy(img))
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_pts.append(corners)
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                # add frame number
                cv2.putText(img, f"Frame {f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 8, 2)
                cv2.putText(img, f"Frame {f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 2)
                height = 720
                width = int(img.shape[1] * height / img.shape[0])
                imS = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                if plot:
                    cv2.imshow("img", imS)
                    cv2.waitKey(500)
                if to_file:
                    # write a small version to jpg
                    jpg = os.path.join(dir, "frame_{:06d}.png".format(int(f)))
                    cv2.imwrite(jpg, imS)

                if len(imgs) == max_imgs:
                    print(f"Maximum required images {max_imgs} found")
                    break
    if progress_bar:
        frames_list.close()

    cap.release()
    # close the plot window if relevant
    cv2.destroyAllWindows()
    # do calibration
    assert len(obj_pts) >= 5, (
        f"A minimum of 5 frames with chessboard patterns must be available, only {len(obj_pts)} found. Please check "
        f"if the video contains chessboard patterns of size {chessboard_size} "
    )
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame_size, None, None)
    # remove badly performing images and recalibrate
    errs = []
    for i in range(len(obj_pts)):
        img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        errs.append(cv2.norm(img_pts[i], img_pts2, cv2.NORM_L2) / len(img_pts2))

    if tolerance is not None:
        # remove high error
        idx = np.array(errs) < tolerance
        obj_pts = list(np.array(obj_pts)[idx])
        img_pts = list(np.array(img_pts)[idx])
        # do calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame_size, None, None)
        errs = []
        for i in range(len(obj_pts)):
            img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            errs.append(cv2.norm(img_pts[i], img_pts2, cv2.NORM_L2) / len(img_pts2))
    print(f"Average error on point reconstruction is {np.array(errs).mean()}")
    return camera_matrix, dist_coeffs


def _Rt_to_M(rvec, tvec, camera_matrix, z=0.0, reverse=False):
    R = cv2.Rodrigues(rvec)[0]
    # assume height of projection plane
    R[:, 2] = R[:, 2] * z
    # add translation vector
    R[:, 2] = R[:, 2] + tvec.flatten()
    # compute homography
    if reverse:
        # From perspective to objective
        M = np.dot(camera_matrix, R)
    else:
        # from objective to perspective
        M = np.linalg.inv(np.dot(camera_matrix, R))
    # normalize homography before returning
    return M / M[-1, -1]
    return M / M[-1, -1]


def pose_world_to_camera(rvec, tvec):
    """Convert a world coordinate pose to a camera coordinate pose or vice versa.

    Parameters
    ----------
    rvec : numpy.ndarray
        A 3x1 or 1x3 rotation vector in world coordinates.
    tvec : numpy.ndarray
        A 3x1 translation vector in world coordinates.

    Returns
    -------
    tuple
        A tuple containing:
        - rvec (numpy.ndarray): A 1-dimensional array representing the rotation vector
          in camera coordinates.
        - tvec (numpy.ndarray): A 3x1 array representing the translation vector
          in camera coordinates.

    """
    # Get Rodriguez rotation matrix
    R_input, _ = cv2.Rodrigues(rvec.flatten())

    # 2. Create the OpenCV-compatible transformation
    R = R_input.T  # Transpose of the rotation matrix
    tvec = -R @ tvec.flatten()  # Transform the translation vector

    # 3. Convert the adjusted rotation matrix back to a rotation vector
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()
    return rvec, tvec


def distort_points(points, camera_matrix, dist_coeffs):
    """Distort x, y point locations with provided lens parameters.

    Points can be back projected on original (distorted) frame positions.

    Adapted from https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/

    Parameters
    ----------
    points : list of lists
        undistorted points [x, y], provided as float
    camera_matrix : array-like [3x3]
        camera matrix
    dist_coeffs : array-like [4]
        distortion coefficients

    Returns
    -------
    points : list of lists
        distorted point coordinates [x, y] as floats

    """
    points = np.array(points, dtype=np.float64)
    # ptsTemp = np.array([], dtype='float32')
    # make empty rotation and translation vectors (we are only undistorting)
    rtemp = ttemp = np.array([0, 0, 0], dtype="float32")
    # normalize the points to be independent of the camera matrix using undistortPoints with no distortion matrix
    ptsOut = cv2.undistortPoints(points, camera_matrix, None)
    # convert points to 3d points
    ptsTemp = cv2.convertPointsToHomogeneous(ptsOut)
    # project them back to image space using the distortion matrix
    return np.int32(
        np.round(cv2.projectPoints(ptsTemp, rtemp, ttemp, camera_matrix, dist_coeffs, ptsOut)[0][:, 0])
    ).tolist()


def get_M_2D(src, dst, reverse=False):
    """Retrieve homography matrix for between (4) src and (4) dst points with only x, y coordinates (no z).

    Parameters
    ----------
    src : list of lists
        [x, y] with source coordinates, typically cols and rows in image
    dst : list of lists
        [x, y] with target coordinates after reprojection, can e.g. be in crs [m]
    reverse : bool, optional
        If set, the reverse homography to back-project to camera objective will be retrieved

    Returns
    -------
    M : np.ndarray
        homography matrix (3x3), used in cv2.warpPerspective

    """
    # set points to float32
    _src = np.float32(src)
    _dst = np.float32(dst)
    # define transformation matrix based on GCPs
    if reverse:
        M = cv2.getPerspectiveTransform(_dst, _src)
    else:
        M = cv2.getPerspectiveTransform(_src, _dst)
    return M


def get_M_3D(src, dst, camera_matrix, dist_coeffs=None, z=0.0, reverse=False):
    """Retrieve homography matrix for between (6+) 2D src and (6+) 3D dst (x, y, z) points.

    Parameters
    ----------
    src : list of lists
        [x, y] with source coordinates, typically cols and rows in image
    dst : list of lists
        [x, y, z] with target coordinates after reprojection, can e.g. be in crs [m]
    camera_matrix : np.ndarray (3x3)
        Camera intrinsic matrix
    dist_coeffs : p.ndarray, optional
        1xN array with distortion coefficients (N = 4, 5 or 8)
    z : float, optional
        Elevation plane (real-world coordinate crs) of projected image
    reverse : bool, optional
        If set, the reverse homography to back-project to camera objective will be retrieved

    Returns
    -------
    M : np.ndarray
        homography matrix (3x3), used in cv2.warpPerspective

    Notes
    -----
    See rectification workflow OpenCV
    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Code based on:
    https://www.openearth.nl/flamingo/_modules/flamingo/rectification/rectification.html

    """
    dist_coeffs = np.zeros((1, 4)) if dist_coeffs is None else dist_coeffs
    success, rvec, tvec = solvepnp(dst, src, camera_matrix, dist_coeffs)
    return _Rt_to_M(rvec, tvec, camera_matrix, z=z, reverse=reverse)


def color_scale(img, method):
    """Transform color space of an image according to the specified method.

    The function offers support for grayscale conversion, RGB transformation
    for plotting, and HSV conversion. Images initially are expected to be in
    BGR format and will remain unchanged if 'bgr' is specified as the method.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR color space [uint8 type).
    method : str
        The color space conversion to apply to the image. Supported methods are
        'grayscale', 'rgb', 'hsv', and 'bgr'.

    Returns
    -------
    np.ndarray
        Image transformed into the specified color space.

    """
    if method == "grayscale":
        # apply gray scaling, contrast- and gamma correction
        # img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # mean(axis=2)
    elif method == "rgb":
        # turn bgr to rgb for plotting purposes
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif method == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif method == "bgr":
        pass
    elif method == "hue":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    elif method == "sat":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    elif method == "val":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    return img


def get_frame(cap, rotation=None, ms=None, method="grayscale"):
    """Get single frame from video capture.

    The function captures an image frame from the given video capture device
    and applies optional rotation, stabilization, and color scaling.

    Parameters
    ----------
    cap : VideoCapture
        The video capture object from which the frame is to be read.
    rotation : int, optional
        Specifies the angle for the rotation of the image. If None, no
        rotation is applied.
    ms : np.ndarray, optional
        2x3 affina transformation parameters. If None, the image will not undergo stabilization.
    method : str, optional
        Describes the color scaling method to be applied on the image.
        Default is "grayscale". Can also be "rgb", "hsv", "bgr", "hue", "sat", or "val".

    Returns
    -------
    ret : bool
        A flag indicating whether a frame was successfully read.
    img : ndarray
        The image after applying the optional processing steps.

    Raises
    ------
    IOError
        If the function is unable to read from the capture device, an
        IOError is raised indicating a possible issue with the stream.

    """
    try:
        ret, img = cap.read()
        if rotation is not None:
            img = cv2.rotate(img, rotation)
    except IOError:
        raise IOError("Cannot read")
    if ret:
        if ms is not None:
            # apply stabilization on image
            img = transform(img, ms)
        img = color_scale(img, method)
    return ret, img


def get_time_frames(cap, start_frame, end_frame, lazy=True, fps=None, progress=True, **kwargs):
    """Obtain valid time stamps and frame numbers from video capture object.

    Valid frames may start and end at start_frame and end_frame, respectively. However, certain required frames may
    turn out not readable. These will be captured, and resulting set of valid frames returned only.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Opened VideoCapture object
    start_frame : int
        first frame to consider for reading
    end_frame : int
        last frame to consider for reading
    lazy : bool, optional
        read frames lazily (default) or not. Set to False for direct reading (faster, but more memory)
    fps : float, optional
        hard enforced frames per second number (used when metadata of video is incorrect)
    progress : bool, optional
        display progress bar. Default is True.
    **kwargs : dict, optional
        additional keyword arguments passed to get_frame() function

    Returns
    -------
    time : list
        list with valid time stamps in milliseconds. time stamps belongs to the start of the frame in frame number.
    frame_number : list
        list with valid frame numbers

    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, np.float64(start_frame))
    pbar = tqdm(
        total=end_frame - start_frame + 1, position=0, desc="Scanning video", disable=not (progress), leave=True
    )
    ret, img = get_frame(cap, **kwargs)
    # pbar.update(1)
    n = start_frame
    time = []
    frame_number = []
    frames = None if lazy else []
    while ret:
        if n > end_frame:
            break
        if not lazy:
            frames.append(img)
        t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
        time.append(n * 1000.0 / fps) if fps is not None else time.append(t1)
        frame_number.append(n)
        n += 1
        ret, img = get_frame(cap, **kwargs)  # read frame 1 + ...
        pbar.update(1)
        if ret == False:
            break
        t2 = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t2 <= 0.0:
            # invalid time difference, stop reading.
            break
    # do a final check if the last frame(s) are readable by direct seek and read. Sometimes this results in not being
    # able to r
    last_valid_idx = _check_valid_frames(cap, frame_number)
    if last_valid_idx is not None:
        time = time[:last_valid_idx]
        frame_number = frame_number[:last_valid_idx]
        if not lazy:
            frames = frames[:last_valid_idx]
    return time, frame_number, frames


def get_ortho(img, M, shape, flags=cv2.INTER_AREA):
    """Reproject an image to a given shape using perspective transformation matrix M.

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


def get_aoi(dst_corners, resolution=None, method="corners"):
    """Get rectangular AOI from 3 or 4 user defined points within frames.

    Parameters
    ----------
    dst_corners : np.ndarray
        corners of aoi, with `method="width_length"` in order: left-bank, right-bank, up/downstream point,
        with `method="corners"` in order: upstream-left, downstream-left, downstream-right, upstream-right.
    resolution : float
        resolution of intended reprojection, used to round the bbox to a whole number of intended pixels
    method : str
        can be "corners" or "width_length". With "corners", the AOI is defined by the four corners of the rectangle.
        With "width" length, the AOI is defined by the width (2 points) and length (1 point) of the rectangle.

    Returns
    -------
    bbox : shapely.geometry.Polygon
        bounding box of aoi (with rotated affine)

    """
    if method == "corners":
        bbox = _get_aoi_corners(dst_corners, resolution)
    elif method == "width_length":
        bbox = _get_aoi_width_length(dst_corners)

    else:
        raise ValueError("method must be 'corners' or 'width_length'")

    return bbox


def get_polygon_pixels(img, pol, reverse_y=False):
    """Get pixel intensities within a polygon."""
    if pol.is_empty:
        return np.array([np.nan])
    polygon_coords = list(pol.exterior.coords)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([polygon_coords], np.int32), color=255)
    if reverse_y:
        return np.flipud(img)[mask == 255]
    return img[mask == 255]


def optimize_intrinsic(src, dst, height, width, c=2.0, lens_position=None, camera_matrix=None, dist_coeffs=None):
    """Optimize the intrinsic parameters of a camera model.

    The function finds optimal intrinsic camera parameters, including focal length and distortion coefficients, by
    minimizing the reprojection error from 3D source points to 2D destination points. It uses differential evolution
    for optimization. Optionally lens position can be provided to include additional geometric constraints.

    Parameters
    ----------
    src : array_like
        Source points in the original 3D space to be projected.
    dst : array_like
        Destination points in the 2D image space, serving as the target for
        projecting the source points.
    height : int
        The height of the image in pixels.
    width : int
        The width of the image in pixels.
    c : float, optional
        Center parameter of the camera matrix.
    lens_position : array_like, optional
        The assumed position of the lens in the 3D space.
    camera_matrix : Optional[List[List]]
        Predefined camera matrix. If not provided focal length will be fitted and camera matrix returned
    dist_coeffs : Optional[List[List]]
        Distortion coefficients to be used for the camera. If not provided, the first two (k1, k2)
        distortion coefficients are fitted on data.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, float]
        A tuple containing the optimized camera matrix, distortion coefficients,
        and the minimized reprojection error.

    """

    def error_intrinsic(x, src, dst, height, width, c=2.0, lens_position=None, camera_matrix=None, dist_coeffs=None):
        """Compute the reprojection error for the intrinsic parameters of a camera model.

        This function optimizes for the focal length and first two distortion coefficients based on the source and
        destination point correspondences provided. Lens position may be provided as additional geometric constraint.

        Parameters
        ----------
        x : array_like
            The array containing the optimization parameters, where `x[0]` is used
            to compute the focal length and `x[1]` and `x[2]` are used to adjust
            the distortion coefficients.
        src : array_like
            Source points in the original 3D space that need to be projected.
        dst : array_like
            Destination points in the 2D image space, which are the target for
            the projection of the source points.
        height : int
            The height of the image in pixels.
        width : int
            The width of the image in pixels.
        c : float, optional
            center parameter of camera matrix.
        lens_position : array_like, optional
            The assumed position of the lens in the 3D space.
        camera_matrix : array_like, optional
            camera matrix [3x3]
        dist_coeffs : array_like, optional
            Distortion coefficients.

        Returns
        -------
        float
            The computed mean reprojection error, with optional contributions from
            the camera position error if the lens position is provided.

        """
        param_nr = 0
        # set the parameters
        if camera_matrix is None:
            f = x[param_nr] * width
            camera_matrix_sample = get_cam_mtx(height, width, c=c, focal_length=f)
            param_nr += 1
        else:
            # take the existing camera matrix
            camera_matrix_sample = camera_matrix.copy()
        if dist_coeffs is None:
            dist_coeffs_sample = DIST_COEFFS.copy()
            k1 = x[param_nr]
            k2 = x[param_nr + 1]
            dist_coeffs_sample[0][0] = k1
            dist_coeffs_sample[1][0] = k2
        else:
            # take the existing distortion coefficients
            dist_coeffs_sample = dist_coeffs.copy()
            k1 = dist_coeffs_sample[0][0]
            k2 = dist_coeffs_sample[1][0]

        # initialize error
        err = 100
        cam_err = None

        # reduce problem space to centered around gcp average
        coord_mean = np.array(dst).mean(axis=0)
        _dst = np.float64(np.array(dst) - coord_mean)
        zs = np.zeros(4) if len(_dst[0]) == 2 else np.array(_dst)[:, -1]
        if lens_position is not None:
            _lens_pos = np.array(lens_position) - coord_mean

        # camera_matrix = _get_cam_mtx(height, width, c=c, focal_length=f)
        success, rvec, tvec = solvepnp(_dst, src, camera_matrix_sample, dist_coeffs_sample)
        if success:
            # estimate destination locations from pose
            dst_est = unproject_points(src, zs, rvec, tvec, camera_matrix_sample, dist_coeffs_sample)
            # src_est = np.array([list(point[0]) for point in src_est])
            dist_xy = np.array(_dst)[:, 0:2] - np.array(dst_est)[:, 0:2]
            dist = (dist_xy**2).sum(axis=1) ** 0.5
            gcp_err = dist.mean()
            if lens_position is not None:
                rmat = cv2.Rodrigues(rvec)[0]
                lens_pos2 = np.array(-rmat).T @ tvec
                cam_err = ((_lens_pos - lens_pos2.flatten()) ** 2).sum() ** 0.5
                # TODO: for now camera errors are weighted with 10% needs further investigation
            err = float(0.1 * cam_err + gcp_err) if cam_err is not None else gcp_err
        return err  # assuming gcp pixel distance is about 5 cm

    # determine optimization bounds
    bounds = []
    if camera_matrix is not None and dist_coeffs is not None:
        # both are already known, so nothing to do
        return camera_matrix, dist_coeffs, None
    if camera_matrix is None:
        bounds.append([float(0.25), float(2)])
    if len(dst) > 4 and dist_coeffs is None:
        bounds.append([-0.9, 0.9])  # k1
        bounds.append([-0.5, 0.5])  # k2
    else:
        # set a warning if dist_coeffs is provided without sufficient ground control
        if dist_coeffs:
            warnings.warn(
                "You are trying to optimize distortion coefficients with only 4 GCPs. "
                "This would lead to overfitting, setting distortion coefficients to zero.",
                stacklevel=2,
            )
        dist_coeffs = DIST_COEFFS.copy()
    # if len(dst) == 4:
    #     bnds_k1 = (-0.0, 0.0)
    #     bnds_k2 = (-0.0, 0.0)
    # else:
    #     # bnds_k1 = (-0.2501, -0.25)
    #     bnds_k1 = (-0.9, 0.9)
    #     bnds_k2 = (-0.5, 0.5)
    # bnds_k1 = (-0.0, 0.0)
    # bnds_k2 = (-0.0, 0.0)
    opt = optimize.differential_evolution(
        error_intrinsic,
        # bounds=[(float(0.25), float(2)), bnds_k1],#, (-0.5, 0.5)],
        bounds=bounds,
        # bounds=[(1710./width, 1714./width), bnds_k1, bnds_k2],
        args=(src, dst, height, width, c, lens_position, camera_matrix, dist_coeffs),
        atol=0.001,  # one mm
    )
    param_nr = 0
    if camera_matrix is None:
        camera_matrix = get_cam_mtx(height, width, focal_length=opt.x[param_nr] * width)
        # move to next parameter
        param_nr += 1
    if dist_coeffs is None:
        dist_coeffs = DIST_COEFFS
        dist_coeffs[0][0] = opt.x[param_nr]
        dist_coeffs[1][0] = opt.x[param_nr + 1]
    # dist_coeffs[4][0] = opt.x[3]
    # dist_coeffs[3][0] = opt.x[4]
    # print(f"CAMERA MATRIX: {camera_matrix}")
    # print(f"DIST COEFFS: {dist_coeffs}")
    return camera_matrix, dist_coeffs, opt.fun


def transform_to_bbox(coords, bbox, resolution):
    """Transform coordinates defined in crs of bbox, into cv2 compatible pixel coordinates.

    Parameters
    ----------
    coords : list of lists
        [x, y, z] with coordinates
    bbox : shapely Polygon
        Bounding box. The coordinate order is very important and has to be upstream-left, downstream-left,
        downstream-right, upstream-right, upstream-left
    resolution : float
        resolution of target pixels within bbox

    Returns
    -------
    colrows : list
        tuples of columns and rows

    """
    # first assemble x and y coordinates
    transform = _get_transform(bbox, resolution)
    if len(coords[0]) == 3:
        xs, ys, zs = zip(*coords)
    else:
        xs, ys = zip(*coords)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys, op=float)
    return list(zip(cols, rows)) if len(coords[0]) == 2 else list(zip(cols, rows, zs))


def undistort_img(img, camera_matrix, dist_coeffs):
    """Correct lens distortion of image based on lens characteristics.

    Function by Gerben Gerritsen / Sten Schurer, 2019.

    Parameters
    ----------
    img : np.ndarray
        3D array with image
    camera_matrix: np.ndarray
        Camera matrix
    dist_coeffs: np.ndarray
        distortion coefficients

    Returns
    -------
    img: np.ndarray
        undistorted img

    """
    # correct image for lens distortion
    return cv2.undistort(img, np.array(camera_matrix), np.array(dist_coeffs))


def unproject_points(src, z, rvec, tvec, camera_matrix, dist_coeffs):
    """Reverse-project points from the image to the 3D world.

    As points on the objective are a ray line in the real world, a x, y, z coordinate can only be reconstructed if
    the points have one known coordinate (z).

    Parameters
    ----------
    src : list (of lists)
        pixel coordinates
    z : float or list of floats
        z-level belonging to src points (if list, then the length should be equal to the number of src points)
    rvec : array-like
        rotation vector
    tvec : array-like
        translation vector
    camera_matrix : array-like
        camera matrix
    dist_coeffs : array_like
        distortion coefficients

    Returns
    -------
    np.ndarray
        unprojected points (x, y, z)

    """
    src = np.float64(np.atleast_1d(src))
    # first undistort points
    src = np.float64(np.array(undistort_points(src, camera_matrix, dist_coeffs)))
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    if isinstance(z, (list, np.ndarray)):
        #         zs = np.atleast_1d(zs)
        z = np.float64(z)
        dst = []
        assert len(z) == len(
            src
        ), f"Amount of src points {len(src)} is not equal to amount of vertical levels z {len(z)}"
        for pt, _z in zip(src, z):
            M = _Rt_to_M(rvec, tvec, camera_matrix, z=_z, reverse=False)
            x, y = list(cv2.perspectiveTransform(pt[None, None, ...], M)[0][0])
            dst.append([x, y, _z])
    else:
        # z is only one value, so there is no change in M. We can do this faster in one go
        M = _Rt_to_M(rvec, tvec, camera_matrix, z=z, reverse=False)
        dst = cv2.perspectiveTransform(src[None, ...], M)[0]
        dst = np.hstack(
            (
                dst,
                np.ones((len(dst), 1)) * z,
            )
        )
    return dst


def undistort_points(points, camera_matrix, dist_coeffs, reverse=False):
    """Undistort x, y point locations with provided lens parameters.

    points can be undistorted together with images from that lens.

    Parameters
    ----------
    points : list of lists
        points [x, y], provided as float
    camera_matrix : array-like [3x3]
        camera matrix
    dist_coeffs : array-like [4]
        distortion coefficients
    reverse : bool, optional
        if set, the distortion will be undone, so that points can be back projected on original (distorted) frame
        positions.

    Returns
    -------
    points : list of lists
        undistorted point coordinates [x, y] as floats

    """
    camera_matrix = np.array(camera_matrix)
    dist_coeffs = np.array(dist_coeffs)
    if reverse:
        return distort_points(points, camera_matrix, dist_coeffs)

    points_undistort = cv2.undistortPoints(
        np.expand_dims(np.float64(points), axis=1), camera_matrix, dist_coeffs, P=camera_matrix
    )
    return points_undistort[:, 0].tolist()


def world_to_camera(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
    """Transform points from the world coordinate system to the camera coordinate system.

    Parameters
    ----------
    points : np.ndarray
        An array of 3D points in the world coordinate system.
    rvec : np.ndarray
        OpenCV compatible rotation vector.
    tvec : np.ndarray
        OpenCV comptatible translation vector.

    Returns
    -------
    np.ndarray
        array of 3D points in the camera coordinate system.

    """
    # get Rodriguez rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # rotate and translate points
    return (points @ R.T) + tvec.T
