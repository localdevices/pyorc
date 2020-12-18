import io
import cv2
import rasterio
from rasterio.io import MemoryFile
from pyproj import CRS
import geojson
from OpenRiverCam.cv import _corr_lens, _corr_color

def frames(
    fn,
    frame_int=1,
    start_frame=0,
    end_frame=None,
    grayscale=False,
    lens_pars=None,
):
    """

    :param fn:  filename (str) or BytesIO object containing a movie dataset
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
                img = _corr_lens(img, **lens_pars)
            # apply gray scaling, contrast- and gamma correction
            if grayscale:
                img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
            # update frame number
            _n += 1
            # update frame time
            _t += 1.0 / fps
            yield _t, img
        else:
            break
    return

def to_geotiff(fn, z, transform, crs=None, compress=None):
    if crs is not None:
        # try:
        crs = CRS.from_user_input(crs)
        # except:
            # raise ValueError(f'CRS "{crs}" is not valid')
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
        compress=compress,
    ) as ds:
        for n, _z in enumerate(z):
            ds.write(_z, n + 1)

def to_geojson(geom, crs=None):
    """
    Converts a single geometry in a geographically aware geojson
    :param geom: shapely feature (single!!)
    :param crs=None: pyproj readible crs
    :return: str, geojson format
    """
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
    # prepare json compatible crs dict
    crs_json = {"type": "EPSG", "properties": {"code": epsg}} if crs is not None else None
    # prepare geojson feature
    f = geojson.Feature(geometry=geom, properties={"ID": 0})
    # collate all into a geojson feature collection
    return geojson.FeatureCollection([f], crs=crs_json)

