import cv2
import numpy as np

def get_gcps_a(gcps, cam_loc, h_a):
    """
    :param gcps:  Ground control points containing x, y, z_0 (zero water level in crs [m]) and h_ref (water level during measuring campaign)
    :param cam_loc: dict with "x", "y" and "z", location of cam in local crs [m]
    :param h_a: float - actual water level in local level measuring system [m]
    :return: gcps, where dst is replaced by new dst with actual water level

    """
    #get modified gcps based on camera location and elevation values
    dest_x, dest_y = zip(*gcps["dst"])
    z_ref = gcps["h_ref"] + gcps["z_0"]
    z_a = gcps["z_0"] + h_a
    # compute the water table to camera height difference during field referencing
    cam_height_ref = cam_loc["z"] - z_ref
    # compute the actual water table to camera height difference
    cam_height_a = cam_loc["z"] - z_a
    rel_diff = cam_height_a / cam_height_ref
    # apply the diff on all coordinate, both x, and y directions
    _dest_x = list(cam_loc["x"] + (np.array(dest_x)-cam_loc["x"])*rel_diff)
    _dest_y = list(cam_loc["y"] + (np.array(dest_y)-cam_loc["y"])*rel_diff)
    gcps["dst"] = list(zip(_dest_x, _dest_y))
    return gcps


def get_M(gcps):
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
    _src = np.float32(gcps['src'])
    _dst = np.float32(gcps['dst'])
    # define transformation matrix based on GCPs
    M = cv2.getPerspectiveTransform(_src, _dst)
    return M
    # find locations of transformed image corners

def orthorectification():
    # FIXME
    raise NotImplementedError("")

def surf_velocity():
    # FIXME
    raise NotImplementedError("")

def river_flow():
    # FIXME
    raise NotImplementedError("")

