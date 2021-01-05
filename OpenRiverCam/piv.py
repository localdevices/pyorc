import openpiv.tools
import openpiv.pyprocess

def piv(frame_a, frame_b, res_x=1., res_y=1., search_area_size=60, overlap=30, **kwargs):
    """
    Typical kwargs are for instance
    window_size=60, overlap=30, search_area_size=60, dt=1./25
    :param fn1:
    :param fn2:
    :param res_x: float, resolution of x-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param res_y: float, resolution of y-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param kwargs:
    :return:
    """
    v_x, v_y, s2n = openpiv.pyprocess.extended_search_area_piv(frame_a, frame_b, search_area_size=search_area_size, overlap=overlap, **kwargs)
    cols, rows = openpiv.pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    return cols, rows, v_x*res_x, v_y*res_y, s2n

def imread(fn):
    return openpiv.tools.imread(fn)