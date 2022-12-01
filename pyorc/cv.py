import copy
import cv2
import numpy as np
import os
import rasterio
from pyorc import helpers
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate
from tqdm import tqdm
from scipy.cluster.vq import vq, kmeans
import operator


# criteria for finding subpix corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
    Establish distortion coefficient matrix for use in cv2.undistort

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
    # mtx[0, 2] = width / c  # define center x
    # mtx[1, 2] = height / c  # define center y
    # mtx[0, 0] = f  # define focal length x
    # mtx[1, 1] = f  # define focal length y

    mtx[0, 2] = width / c  # define center x
    mtx[1, 2] = height / c  # define center y
    mtx[0, 0] = width  # define focal length x
    mtx[1, 1] = width  # define focal length y
    return mtx

def _get_displacements(cap, start_frame=0, end_frame=None, n_pts=None, split=2, mask=None):
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
    split : int, optional
        Number of regions to split the entire field of view in to find points equally spread, defaults to 2
    mask : np.ndarray (2D), optional
        if set, the areas that are one, are assumed to be region of interest and therefore masked out for finding points

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
    if mask is not None:
        prev_gray[mask > 0] = 0.

    # prepare outputs
    n_frames = int(end_frame) - int(start_frame)
    transforms = np.zeros((n_frames - 1, 3), np.float32)


    if n_pts is None:
        # use the square root of nr of pixels in a frame to decide on n_pts
        n_pts = int(np.sqrt(len(prev_gray.flatten())))

    # split image in smaller chunks if user wants
    v = 0
    h = 0
    ver_split, hor_split = np.int16(np.ceil(np.array(prev_gray.shape) / split))
    prev_pts = np.zeros((0, 1, 2), np.float32)
    while v < prev_gray.shape[0]:
        while h < prev_gray.shape[1]:
            sub_img = prev_gray[v:v + ver_split, h:h + hor_split]
            # get points over several quadrants
            subimg_pts = cv2.goodFeaturesToTrack(
                sub_img,
                maxCorners=int(n_pts/split**2),
                qualityLevel=0.3,
                minDistance=10,
                blockSize=1
            )
            # add offsets for quadrants
            if subimg_pts is not None:
                subimg_pts[:, :, 0] += h
                subimg_pts[:, :, 1] += v
                prev_pts = np.append(prev_pts, subimg_pts, axis=0)
            h += hor_split
        h = 0
        v += ver_split
    # # get points over several quadrants
    # prev_pts = cv2.goodFeaturesToTrack(
    #     prev_gray,
    #     maxCorners=n_pts,
    #     qualityLevel=0.1,
    #     minDistance=10,
    #     blockSize=3
    # )
    # add start point to matrix

    # get another frame a little bit ahead in time
    # get start frame and points
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 30)
    #
    # success, curr = cap.read()
    # Convert to grayscale
    # curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # # Calculate optical flow
    # curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    # # Calculate optical flow
    # curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # # store curr_pts
    # positions = np.append(positions, np.swapaxes(curr_pts, 0, 1), axis=0)
    # stats = np.append(stats, np.swapaxes(status, 0, 1), axis=0)
    # errs = np.append(errs, np.swapaxes(err, 0, 1), axis=0)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use("Qt5Agg")
    # plt.imshow(prev_gray, cmap="Greys_r")
    # x = prev_pts[:, :, 0].flatten()
    # y = prev_pts[:, :, 1].flatten()
    # intensity = prev_gray[np.int16(y), np.int16(x)]
    # # x_curr = prev_pts[intensity<100, :, 0].flatten()
    # # y_curr = prev_pts[intensity<100, :, 1].flatten()
    # # x_curr = curr_pts[status.flatten()==1, :, 0].flatten()
    # # y_curr = curr_pts[status.flatten()==1, :, 1].flatten()
    # # prev_pts = prev_pts[intensity < 100]
    #
    # plt.plot(x, y, ".")
    # # # plt.plot(x[status.flatten()==0], y[status.flatten()==0], "ro")
    # # plt.plot(x_curr, y_curr, "r.")
    # plt.show()
    positions = np.swapaxes(prev_pts, 0, 1)
    # update n_pts to the amount truly found
    n_pts = positions.shape[1]
    # prepare storage for points
    # positions = np.zeros((0, n_pts, 2), np.float32)
    stats = np.ones((1, n_pts))
    errs = np.ones((1, n_pts))
    # loop through start to end frame
    pbar = tqdm(range(n_frames - 1), position=0, leave=True)
    for i in pbar:
        # Read next frame
        pbar.set_description(f"Stabilizing frames")
        success, curr = cap.read()
        if not success:
            raise IOError(f"Could not read frame {start_frame + i} from video")

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            curr_gray[mask > 0] = 0.
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
    key_point : int, optional

    Returns
    -------
    ms : list
        Contains affine transform matrix for each set of points. This list can be used to affine transform each
        frame to match as closely as possible the frame chosen as central frame.

    """
    assert key_point >= 0 and key_point < len(pts), f"Key point {int(key_point)} must be within range of point locations (0 - {len(pts) - 1}."
    # TODO: approach to remove points before going to transformation is not working properly yet.
    # remove points that at some point disappear, by finding which points have status that sometimes is zero
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
    """
    Intrinsic matrix calculation and distortion coefficients calculation following
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    dir = os.path.split(os.path.abspath(fn))[0]
    cap = cv2.VideoCapture(fn)
    # make a list of logical frames in order to read
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_list = helpers.staggered_index(start=0, end=frames_count - 1)

    # set the expected object points from the chessboard size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

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

                #         print(corners)
                # skip 25 frames
                # cap.set(cv2.CAP_PROP_POS_FRAMES, cur_f + df)
                if len(imgs) == max_imgs:
                    print(f"Maximum required images {max_imgs} found")
                    break
    if progress_bar:
        frames_list.close()

    cap.release()
    # close the plot window if relevant
    cv2.destroyAllWindows()
    # do calibration
    assert(len(obj_pts) >= 5),\
        f"A minimum of 5 frames with chessboard patterns must be available, only {len(obj_pts)} found. Please check " \
        f"if the video contains chessboard patterns of size {chessboard_size} "
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame_size, None, None)
    # remove badly performing images and recalibrate
    errs = []
    for n, i in enumerate(range(len(obj_pts))):
        img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        errs.append(cv2.norm(img_pts[i], img_pts2, cv2.NORM_L2) / len(img_pts2))

    if tolerance is not None:
        # remove high error
        idx = np.array(errs) < tolerance
        obj_pts = list(np.array(obj_pts)[idx])
        img_pts = list(np.array(img_pts)[idx])
        print(len(img_pts))
        # do calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, frame_size, None, None)
        errs = []
        for n, i in enumerate(range(len(obj_pts))):
            img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            errs.append(cv2.norm(img_pts[i], img_pts2, cv2.NORM_L2) / len(img_pts2))
    print(f"Average error on point reconstruction is {np.array(errs).mean()}")
    return camera_matrix, dist_coeffs


def get_M_2D(src, dst, reverse=False):
    """
    Retrieve homography matrix for between (4) src and (4) dst points with only x, y coordinates (no z)

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

def get_M_3D(src, dst, camera_matrix, dist_coeffs=np.zeros((1, 4)), z=0., reverse=False):
    """
    Retrieve homography matrix for between (6+) 2D src and (6+) 3D dst (x, y, z) points

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
    # set points to float32
    _src = np.float32(src)
    _dst = np.float32(dst)
    # import pdb;pdb.set_trace()
    camera_matrix = np.float32(camera_matrix)
    dist_coeffs = np.float32(dist_coeffs)
    # define transformation matrix based on GCPs
    success, rvec, tvec = cv2.solvePnP(_dst, _src, camera_matrix, dist_coeffs)
    # convert rotation vector to rotation matrix
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


def transform_to_bbox(coords, bbox, resolution):
    """transforms a set of coordinates defined in crs of bbox, into a set of coordinates in cv2 compatible pixels

    Parameters
    ----------
    coords : list of lists
        [x, y] with coordinates
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
    xs, ys = zip(*coords)
    transform = _get_transform(bbox, resolution)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys, op=float)
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


def get_aoi(M, src_corners, resolution=None):
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
    # prepare a simple temporary np.array of the src_corners
    _src_corners = np.array(src_corners)
    assert(_src_corners.shape==(4, 2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. " \
                                        f"Current shape is {src_corners.shape} "
    # reproject corner points to the actual space in coordinates
    _dst_corners = cv2.perspectiveTransform(np.float32([_src_corners]), M)[0]
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
    if resolution is not None:
        xmin = helpers.round_to_multiple(xmin, resolution)
        xmax = helpers.round_to_multiple(xmax, resolution)
        ymin = helpers.round_to_multiple(ymin, resolution)
        ymax = helpers.round_to_multiple(ymax, resolution)

    bbox_coords = [(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin), (xmin, ymax)]
    bbox = Polygon(bbox_coords)
    # now rotate back
    bbox = rotate(bbox, angle, origin=tuple(_dst_corners[0]), use_radians=True)
    return bbox


def undistort_img(img, camera_matrix, dist_coeffs):
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
    # height, width, __ = img.shape
    # dist = _get_dist_coefs(k1)
    #
    # # get camera matrix
    # mtx = _get_cam_mtx(height, width, c=c, f=f)

    # correct image for lens distortion
    return cv2.undistort(img, np.array(camera_matrix), np.array(dist_coeffs))


def distort_points(points, camera_matrix, dist_coeffs):
    """
    Distorts x, y point locations with provided lens parameters, so that points
    can be back projected on original (distorted) frame positions.

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

    points = np.array(points, dtype='float32')
    # ptsTemp = np.array([], dtype='float32')
    # make empty rotation and translation vectors (we are only undistorting)
    rtemp = ttemp = np.array([0, 0, 0], dtype='float32')
    # normalize the points to be independent of the camera matrix using undistortPoints with no distortion matrix
    ptsOut = cv2.undistortPoints(points, camera_matrix, None)
    #convert points to 3d points
    ptsTemp = cv2.convertPointsToHomogeneous(ptsOut)
    # project them back to image space using the distortion matrix
    return np.int32(
        np.round(
            cv2.projectPoints(
                ptsTemp,
                rtemp,
                ttemp,
                camera_matrix,
                dist_coeffs,
                ptsOut
            )[0][:,0]
        )
    ).tolist();



def undistort_points(points, camera_matrix, dist_coeffs, reverse=False):
    """Undistorts x, y point locations with provided lens parameters, so that points
    can be undistorted together with images from that lens.

    Parameters
    ----------
    points : list of lists
        points [x, y], provided as float
    camera_matrix : array-like [3x3]
        camera matrix
    dist_coeffs : array-like [4]
        distortion coefficients

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
        np.expand_dims(np.float32(points), axis=1),
        camera_matrix,
        dist_coeffs,
        P=camera_matrix
    )
    return points_undistort[:, 0].tolist()

