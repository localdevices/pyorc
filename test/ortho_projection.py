import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape, MultiPoint
from shapely.affinity import rotate

from descartes.patch import PolygonPatch

import OpenRiverCam as ORC
import cv2

# this test will do orthographic projection with developed functions on one example

# it'll also include options to assess changes in orthography given changes in water levels. For this the position
# of the camera is needed, in the same x, y, z reference system as the measurements of the GCPs.
# folder = r"c:\OpenRiverCam"
folder = r"/home/hcwinsemius/OpenRiverCam"

src = os.path.join(folder, "with_lens")
dst = os.path.join(folder, "ortho_proj")

fns = glob.glob(os.path.join(src, "*.jpg"))
gcps = {
    "src": [(992, 366), (1545, 403), (1773, 773), (943, 724)],
    # "dst": [(0.25, 0.3), (4.25, 0.2), (4.5, 3.5), (0.0, 3.5)],
    "dst": [(0.25, 6.7), (4.25, 6.8), (4.5, 3.5), (0.0, 3.5)],
    "z_0": 100.0,  # reference height of zero water level compared to the crs used
    "h_ref": 2.0,  # actual water level during taking of the gcp points
}
img = cv2.imread(fns[0])
h_a = 2.0  # now the actual water level is 5 m above reference
cam_loc = {"x": -3.0, "y": 8.0, "z": 110.0}

# corner points provided by user in pixel coordinates, starting at upstream-left, downstream-left, downstream-right, upstream-right
src_corners = {
    "up_left": (200, 100),
    "down_left": (1850, 150),
    "down_right": (1890, 900),
    "up_right": (50.0, 970),
}

# make a polygon from corner points
src_polygon = Polygon([src_corners[s] for s in src_corners])
bbox = ORC.ortho.get_aoi(gcps, src_corners)

print(src_polygon)
print(bbox)

# next step is to reproject
# retrieve polygon from geojson
# bbox = shape(dst_polygon["features"][0]["geometry"])
coords = np.array(bbox.exterior.coords)
# estimate the angle of the bounding box
# retrieve average line across AOI
point1 = coords[0]
point2 = coords[1]
diff = point2 - point1
angle = np.arctan2(diff[1], diff[0])
bbox_rotate = rotate(
    bbox, -angle, origin=tuple(coords[0]), use_radians=True
)

# retrieve the angle of one of the lines and rotate all points and bbox around this angle to get to a surrogate projection

gcps_a = ORC.ortho._get_gcps_a(copy.deepcopy(gcps), cam_loc, h_a)
gcps_dst_rot = rotate(MultiPoint(gcps_a["dst"]), -angle, origin=tuple(coords[0]), use_radians=True)

# convert into a list of tuples

# also rotate the "actual" gcps (i.e. after correction for water surface elevation) to the same system as the AOI bbox
gcps_a["dst"] = [(g.xy[0][0], g.xy[1][0]) for g in gcps_dst_rot]

# retrieve M for original situation
M = ORC.ortho._get_M(gcps)

# retrieve M for modified water levels
M2 = ORC.ortho._get_M(gcps_a)

# now reproject with the retrieved M
# get the corner points in pixel coordinates
corners_pix = cv2.perspectiveTransform(np.float32([np.array(bbox.exterior.coords)[0:4]]), np.linalg.pinv(M))[0]
corners_pix2 = cv2.perspectiveTransform(np.float32([np.array(bbox_rotate.exterior.coords)[0:4]]), np.linalg.pinv(M2))[0]

# TODO figure out what happens if the area is outside of the pixel domain


plt.figure()
ax = plt.subplot(121)
_src = np.array(gcps_a["src"])
_dst = np.array(gcps["dst"])
_dst2 = np.array(gcps_a["dst"])
src_patch = PolygonPatch(
    src_polygon, facecolor="b", edgecolor="b", alpha=0.25, zorder=2
)
dst_patch = PolygonPatch(
    bbox, facecolor="b", edgecolor="b", alpha=0.25, zorder=2
)
dst_patch_rotate = PolygonPatch(
    bbox_rotate, facecolor="b", edgecolor="b", alpha=0.25, zorder=2
)

plt.imshow(img)
plt.plot(_src[:, 0], _src[:, 1], ".", label="gcps")
# plt.plot(_src_corners[:, 0], _src_corners[:, 1], "r.", label="corners")
ax.add_patch(src_patch)
# plt.gca().invert_yaxis()
plt.legend()
ax2 = plt.subplot(122)
plt.plot(_dst[:, 0], _dst[:, 1], ".", label="gcps1")
# plt.plot(_dst_corners[:, 0], _dst_corners[:, 1], "r.", label="corners1")
plt.plot(_dst2[:, 0], _dst2[:, 1], "m.", label="gcps2")
ax2.add_patch(dst_patch)
ax2.add_patch(dst_patch_rotate)

# plt.plot(_dst_corners2[:, 0], _dst_corners[:, 1], "k.", label="corners2")
# plt.gca().invert_yaxis()
plt.legend()
plt.show()

print(gcps)
