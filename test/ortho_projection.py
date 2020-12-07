import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import OpenRiverCam as ORC
import cv2

# this test will do orthographic projection with developed functions on one example

# it'll also include options to assess changes in orthography given changes in water levels. For this the position
# of the camera is needed, in the same x, y, z reference system as the measurements of the GCPs.
folder = r"c:\OpenRiverCam"
# path = r"/home/hcwinsemius/OpenRiverCam

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
h_a = 5.0  # now the actual water level is 5 m above reference
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


dst_polygon = ORC.ortho.get_aoi(gcps, cam_loc, src_corners, crs="EPSG:32737")

print(src_polygon)
print(dst_polygon)

gcps_a = ORC.ortho._get_gcps_a(copy.deepcopy(gcps), cam_loc, h_a)

# retrieve M for original situation
M = ORC.ortho._get_M(gcps)

# retrieve M for modified water levels
M2 = ORC.ortho._get_M(gcps_a)


plt.figure()
ax = plt.subplot(121)
_src = np.array(gcps_a["src"])
_dst = np.array(gcps["dst"])
_dst2 = np.array(gcps_a["dst"])
src_patch = PolygonPatch(src_polygon, facecolor="b", edgecolor="b", alpha=0.25, zorder=2)
dst_patch = PolygonPatch(dst_polygon, facecolor="b", edgecolor="b", alpha=0.25, zorder=2)

plt.imshow(img)
plt.plot(_src[:, 0], _src[:, 1], ".", label="gcps")
# plt.plot(_src_corners[:, 0], _src_corners[:, 1], "r.", label="corners")
ax.add_patch(src_patch)
# plt.gca().invert_yaxis()
plt.legend()
ax2 = plt.subplot(122)
plt.plot(_dst[:, 0], _dst[:, 1], ".", label="gcps1")
# plt.plot(_dst_corners[:, 0], _dst_corners[:, 1], "r.", label="corners1")
plt.plot(_dst2[:, 0], _dst[:, 1], "m.", label="gcps2")
ax2.add_patch(dst_patch)

# plt.plot(_dst_corners2[:, 0], _dst_corners[:, 1], "k.", label="corners2")
# plt.gca().invert_yaxis()
plt.legend()
plt.show()

print(gcps)
