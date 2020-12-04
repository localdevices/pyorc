import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import OpenRiverCam
import cv2

# this test will do orthographic projection with developed functions on one example

# it'll also include options to assess changes in orthography given changes in water levels. For this the position
# of the camera is needed, in the same x, y, z reference system as the measurements of the GCPs.

src = r"/home/hcwinsemius/OpenRiverCam/with_lens"
dst = r"/home/hcwinsemius/OpenRiverCam/ortho_proj"

fns = glob.glob(os.path.join(src, "*.jpg"))
gcps = {
    "src": [(992, 366), (1545, 403), (1773, 773), (943, 724)],
    "dst": [(0.25, 0.3), (4.25, 0.2), (4.5, 3.5), (0., 3.5)],
    "z_0": 100., # reference height of zero water level compared to the crs used
    "h_ref": 2., # actual water level during taking of the gcp points
}
img = cv2.imread(fns[0])
h_a = 5.  # now the actual water level is 5 m above reference
cam_loc = {
    "x": -3.,
    "y": 8.,
    "z": 110.
}

# corner points provided by user in pixel coordinates, starting at upstream-left, downstream-left, downstream-right, upstream-right
ext = [(200, 100), (1850, 150), (1890, 900), (50., 970)]

polygon = Polygon(ext)


gcps_a = OpenRiverCam.ortho.get_gcps_a(copy.deepcopy(gcps), cam_loc, h_a)

# retrieve M for original situation
M = OpenRiverCam.ortho.get_M(gcps)

# retrieve M for modified water levels
M2 = OpenRiverCam.ortho.get_M(gcps_a)

# for now use entire image
# FIXME: use a polygon with 4 corners of AOI instead of entire image
height, width, __ = img.shape
_src_corners = np.array([[0., height], [width, height], [width, 0.], [0., 0.]])
_dst_corners = cv2.perspectiveTransform(np.float32([_src_corners]), M)[0]
_dst_corners2 = cv2.perspectiveTransform(np.float32([_src_corners]), M2)[0]

plt.figure()
ax = plt.subplot(121)
_src = np.array(gcps_a["src"])
_dst = np.array(gcps["dst"])
_dst2 = np.array(gcps_a["dst"])
patch = PolygonPatch(polygon, facecolor='b', edgecolor='b', alpha=0.25,
                     zorder=2)

plt.imshow(img)
plt.plot(_src[:, 0], _src[:, 1], '.', label='gcps')
plt.plot(_src_corners[:, 0], _src_corners[:, 1], 'r.', label='corners')
ax.add_patch(patch)
# plt.gca().invert_yaxis()
plt.legend()
plt.subplot(122)
plt.plot(_dst[:, 0], _dst[:, 1], '.', label='gcps1')
plt.plot(_dst_corners[:, 0], _dst_corners[:, 1], 'r.', label='corners1')
plt.plot(_dst2[:, 0], _dst[:, 1], 'm.', label='gcps2')
plt.plot(_dst_corners2[:, 0], _dst_corners[:, 1], 'k.', label='corners2')
# plt.gca().invert_yaxis()
plt.legend()
plt.show()

print(gcps)

