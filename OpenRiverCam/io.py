import os
import io
import cv2
import numpy as np
import rasterio
from pyproj import CRS

def frames(
    fn,
    frame_int=1,
    start_frame=0,
    end_frame=None,
    lens_pars=None,
):
    """

    :param fn:  filename (str) or BytesIO object containing a movie dataset
    :param dst_path: str - destination for resulting frames
    :param dst_prefix: str - prefix used for naming of frames
    :param frame_int: int - frame interval, difference between frames
    :param start_time=0: start time of first frame to extract (ms)
    :param end_time=None: end time of last frame to extract (ms). If None, it is assumed the entire movie must be extracted
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
                img = lens_corr(img, **lens_pars)
            # apply gray scaling, contrast- and gamma correction
            img = color_corr(img, alpha=None, beta=None, gamma=0.4)
            ret, im_en = cv2.imencode(".jpg", img)
            buf = io.BytesIO(im_en)
            # update frame number
            _n += 1
            # update frame time
            _t += 1.0 / fps
            yield _t, buf
        else:
            break
    return


def lens_corr(img, k1=0.0, c=2.0, f=1.0):
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


def color_corr(img, alpha=None, beta=None, gamma=0.5):
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

def to_geotiff(fn, z, transform, crs=None):
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
    ) as ds:
        for n, _z in enumerate(z):
            ds.write(_z, n + 1)
