import copy
import cv2
import numpy as np
import os
import rasterio
from . import helpers
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate
from tqdm import tqdm
from scipy.cluster.vq import vq, kmeans
from scipy import optimize
import operator

# default distortion coefficients for no distortion
DIST_COEFFS = [[0.], [0.], [0.], [0.], [0.]]

# criteria for finding subpix corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)



def _combine_m(m1, m2):
    # extend to a 3x3 for matrix multiplication
    _m1 = np.append(m1, np.array([[0., 0., 1.]]), axis=0)
    _m2 = np.append(m2, np.array([[0., 0., 1.]]), axis=0)
    m_combi = _m1.dot(_m2)[0:2]
    return m_combi


def _smooth(img, stride):
    """
    Internal function to filter on too large differences from spatial mean

    Parameters
    ----------
    img: image
    stride: window edge size

    Returns
    -------
    img
    """
    blur = cv2.GaussianBlur(img.astype("float32"), (stride, stride), 0)
    return blur


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

def _get_cam_mtx(height, width, c=2.0, focal_length=None):
    """
    Get 3x3 camera matrix from lens parameters

    :param height: height of image from camera
    :param width: width of image from camera
    :param c: float, optical center (default: 2.)
    :param f: float, focal length (optional)
    :return: camera matrix, to be used by cv2.undistort
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


def get_ms_gftt(cap, start_frame=0, end_frame=None, n_pts=None, split=2, mask=None, wdw=4):
    # set end_frame to last if not defined
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    m = np.eye(3)[0:2]
    # m2 = np.eye(3)[0:2]
    ms = []
    # ms2 = []
    m_key = copy.deepcopy(m)
    # get start frame and points
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # determine the mount of frames that must be processed
    n_frames = int(end_frame + 1) - int(start_frame)

    # Read first frame
    _, img_key = cap.read()
    _, img_key = cap.read()
    # Convert frame to grayscale
    img1 = cv2.cvtColor(img_key, cv2.COLOR_BGR2GRAY)
    img_key = img1

    if n_pts is None:
        # use the square root of nr of pixels in a frame to decide on n_pts
        n_pts = int(np.sqrt(len(img_key.flatten())))

    # get features from first key frame
    prev_pts = _gftt_split(img_key, split, n_pts, mask=mask)

    pbar = tqdm(range(n_frames), position=0, leave=True)
    for i in pbar:
        ms.append(m)
        #     ms2.append(m2)
        _, img2 = cap.read()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(img_key, img2, prev_pts, None)
        #     curr_pts = curr_pts[status == 1]
        #     prev_pts = prev_pts[status == 1]
        m_part = cv2.estimateAffine2D(curr_pts, prev_pts)[0]
        m = _combine_m(m_key, m_part)
        if i % 30 == 0:
            img_key = img1
            prev_pts = _gftt_split(img_key, split, n_pts, mask=mask)
            m_key = copy.deepcopy(m)
        img1 = img2

    # smooth the affines over time
    ma = np.array(ms)
    for m in range(ma.shape[1]):
        for n in range(ma.shape[2]):
            ma[wdw:-wdw, m, n] = np.convolve(ma[:, m, n], np.ones(wdw * 2 + 1) / (wdw * 2 + 1), mode="valid")
    ms_smooth = list(ma)
    return ms_smooth


def _get_gcps_2_4(src, dst, img_width, img_height):
    """

    Parameters
    ----------
    src : list or array-like
        source control points (list of lists)
    dst : list or array-like
        destination control points (list of lists)
    img_width : width of original image frame
    img_height : height of original image frame

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
    corners = [
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ]
    dst = cv2.perspectiveTransform(np.float32([corners]), M)[0].tolist()
    # now get the corners back transformed to the real image coordinates
    src = [[x, img_height - y] for x, y in corners]
    return src, dst


def _get_shape(bbox, resolution=0.01, round=1):
    """
    defines the number of rows and columns needed in a target raster, to fit a given bounding box.

    :param bbox: shapely Polygon, bounding box
    :param resolution: resolution of target raster
    :param round: number of pixels to round intended sh ape to
    :return: numbers of rows and columns for target raster
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
    return rasterio.transform.Affine(
        dx_col, dy_col, top_left_x, dx_row, dy_row, top_left_y
    )



def _gftt_split(img, split, n_pts, mask=None):
    # split image in smaller chunks if user wants
    v = 0
    h = 0
    ver_split, hor_split = np.int16(np.ceil(np.array(img.shape) / split))
    pts = np.zeros((0, 1, 2), np.float32)
    while v < img.shape[0]:
        while h < img.shape[1]:
            sub_img = img[v:v + ver_split, h:h + hor_split]
            # get points over several quadrants
            subimg_pts = cv2.goodFeaturesToTrack(
                sub_img,
                mask=mask[v:v + ver_split, h:h + hor_split] if mask is not None else None,
                maxCorners=int(n_pts/split**2),
                qualityLevel=0.3,
                minDistance=10,
                blockSize=1
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
    """
    Short version with preprocessing for cv2.SolvePnP

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
    _src = np.float32(src)
    _dst = np.float32(dst)

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


def _Rt_to_M(rvec, tvec, camera_matrix, z=0., reverse=False):
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

def unproject_points(src, z, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Reverse projects points from the image to the 3D world. As points on the objective are a ray line in
    the real world, a x, y, z coordinate can only be reconstructed if the points have one known coordinate (z).

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

    """
    src = np.float32(np.atleast_1d(src))
    # first undistort points
    src = np.float32(
        np.array(
            undistort_points(
                src,
                camera_matrix,
                dist_coeffs
            )
        )
    )
    if isinstance(z, (list, np.ndarray)):
        #         zs = np.atleast_1d(zs)
        z = np.float64(z)
        dst = []
        assert (len(z) == len(
            src)), f"Amount of src points {len(src)} is not equal to amount of vertical levels z {len(z)}"
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
    success, rvec, tvec = solvepnp(dst, src, camera_matrix, dist_coeffs)
    return _Rt_to_M(rvec, tvec, camera_matrix, z=z, reverse=reverse)


def optimize_intrinsic(src, dst, height, width, c=2., lens_position=None):
    def error_intrinsic(x, src, dst, height, width, c=2., lens_position=None, dist_coeffs=DIST_COEFFS):
        """
        estimate errors in known points using provided control on camera matrix.
        returns error in gcps and camera position (if provided)
        """
        f = x[0]*width  # only one parameter to optimize for now, can easily be extended!
        dist_coeffs[0][0] = float(x[1])
        dist_coeffs[1][0] = float(x[2])
        # dist_coeffs[4][0] = float(x[3])
        # dist_coeffs[3][0] = float(x[4])
        coord_mean = np.array(dst).mean(axis=0)
        # _src = np.float32(src)
        _dst = np.float32(np.array(dst) - coord_mean)
        zs = np.zeros(4) if len(_dst[0]) == 2 else np.array(_dst)[:, -1]
        if lens_position is not None:
            _lens_pos = np.array(lens_position) - coord_mean

        camera_matrix = _get_cam_mtx(height, width, c=c, focal_length=f)
        success, rvec, tvec = solvepnp(_dst, src, camera_matrix, dist_coeffs)
        if success:
            # estimate destination locations from pose
            dst_est = unproject_points(src, zs, rvec, tvec, camera_matrix, dist_coeffs)
            # src_est = np.array([list(point[0]) for point in src_est])
            dist_xy = np.array(_dst)[:, 0:2] - np.array(dst_est)[:, 0:2]
            dist = (dist_xy ** 2).sum(axis=1) ** 0.5
            gcp_err = dist.mean()
            if lens_position is not None:
                rmat = cv2.Rodrigues(rvec)[0]
                lens_pos2 = np.array(-rmat).T @ tvec
                cam_err = ((_lens_pos - lens_pos2.flatten()) ** 2).sum() ** 0.5
            else:
                cam_err = None
                # TODO: for now camera errors are weighted with 10% needs further investigation
            err = float(0.1*cam_err + gcp_err) if cam_err is not None else gcp_err
        else:
            err = 100
        return err  # assuming gcp pixel distance is about 5 cm

    if len(dst) == 4:
        bnds_k1 = (-0.0, 0.0)
        bnds_k2 = (-0.0, 0.0)
    else:
        # bnds_k1 = (-0.2501, -0.25)
        bnds_k1 = (-0.9, 0.9)
        bnds_k2 = (-0.5, 0.5)
    opt = optimize.differential_evolution(
        error_intrinsic,
        # bounds=[(float(0.25), float(2)), bnds_k1],#, (-0.5, 0.5)],
        bounds=[(float(0.25), float(2)), bnds_k1, bnds_k2],
        # bounds=[(1710./width, 1714./width), bnds_k1, bnds_k2],
        args=(src, dst, height, width, c, lens_position, DIST_COEFFS),
        atol=0.001 # one mm
    )
    camera_matrix = _get_cam_mtx(height, width, focal_length=opt.x[0]*width)
    dist_coeffs = DIST_COEFFS
    dist_coeffs[0][0] = opt.x[1]
    dist_coeffs[1][0] = opt.x[2]
    # dist_coeffs[4][0] = opt.x[3]
    # dist_coeffs[3][0] = opt.x[4]
    # print(f"CAMERA MATRIX: {camera_matrix}")
    # print(f"DIST COEFFS: {dist_coeffs}")
    return camera_matrix, dist_coeffs, opt.fun


def color_scale(img, method):
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
    return img


def get_frame(
        cap,
        rotation=None,
        ms=None,
        method="grayscale"
):
    try:
        ret, img = cap.read()
        if rotation is not None:
           img = cv2.rotate(img, rotation)
    except:
        raise IOError(f"Cannot read")
    if ret:
        if ms is not None:
            # apply stabilization on image
            img = transform(img, ms)
        img = color_scale(img, method)
    return ret, img


def get_time_frames(cap, start_frame, end_frame, lazy=True, fps=None, **kwargs):
    """
    Gets a list of valid time stamps and frame numbers for the provided video capture object, starting from start_frame
    ending at end_frame

    Parameters
        ----------
        cap : cv2.VideoCapture
            Opened VideoCapture object
        start_frame : int
            first frame to consider for reading
        end_frame : int
            last frame to consider for reading
        lazy : bool
            read frames lazily (default) or not. Set to False for direct reading (faster, but more memory)
        fps : float
            hard enforced frames per second number (used when metadata of video is incorrect)

        Returns
        -------
        time : list
            list with valid time stamps in milliseconds. each time stamp belongs to the start of the frame in frame number
        frame_number : list
            list with valid frame numbers

    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, img = get_frame(cap, **kwargs)
    n = start_frame
    time = []
    frame_number = []
    if lazy:
        frames = None
    else:
        # already collect the frames
        frames = []
    while ret:
        if n > end_frame:
            break
        if not lazy:
            frames.append(img)
        t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
        if fps is not None:
            time.append(n*1000./fps)
        else:
            time.append(t1)
        # ret, img = cap.read()  # read frame 1 + ...
        ret, img = get_frame(cap, **kwargs)    # read frame 1 + ...
        frame_number.append(n)
        if ret == False:
            break
        # cv2.imwrite("test_{:04d}.jpg".format(n), img)
        t2 = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t2 <= 0.:
            # we can no longer estimate time difference in the last frame read, so stop reading and set end_frame to one frame back
            break

        n += 1
    # time[0] = 0
    return time, frame_number, frames


def transform_to_bbox(coords, bbox, resolution):
    """transforms a set of coordinates defined in crs of bbox, into a set of coordinates in cv2 compatible pixels

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


def get_aoi(dst_corners, resolution=None):
    """Get rectangular AOI from 4 user defined points within frames.

    Parameters
    ----------
    M : np.ndarray
        Homography matrix
    src_corners : dict with 4 (x,y) tuples
        names "up_left", "down_left", "up_right", "down_right", source corners
    resolution : float
        resolution of intended reprojection, used to round the bbox to a whole number of intended pixels

    Returns
    -------
    bbox : shapely.geometry.Polygon
        bounding box of aoi (with rotated affine)
    """
    # prepare a simple temporary np.array of the src_corners
    # assert(_src_corners.shape==(4, 2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. " \
    #                                     f"Current shape is {src_corners.shape} "
    polygon = Polygon(dst_corners)
    coords = np.array(polygon.exterior.coords)
    # estimate the angle of the bounding box
    # retrieve average line across AOI
    point1 = (coords[0] + coords[3]) / 2
    point2 = (coords[1] + coords[2]) / 2
    diff = point2 - point1
    angle = np.arctan2(diff[1], diff[0])
    # rotate the polygon over this angle to get a proper bounding box
    polygon_rotate = rotate(
        polygon, -angle, origin=tuple(dst_corners[0]), use_radians=True
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
    bbox = rotate(bbox, angle, origin=tuple(dst_corners[0]), use_radians=True)
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
