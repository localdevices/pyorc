import matplotlib.pyplot as plt
import shapely.affinity
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

# from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid

fig = plt.figure(figsize=(20, 8))

# 3: invalid polygon, ring touch along a line
ax = fig.add_subplot(111)


ext = [(0, 0), (10, -1.5), (14, 3.0), (1.0, 0.5), (0, 0)]

x, y = zip(*ext)

# int = [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1), (0.5, 0)]
polygon = Polygon(ext)

polygon_user = shapely.affinity.rotate(polygon, 45, origin=ext[2])
polygon_back = shapely.affinity.rotate(polygon_user, -45, origin=ext[2])
x, y = polygon_user.exterior.coords.xy
ax.plot(x, y, "k")

box = shapely.geometry.box(
    *shapely.affinity.rotate(polygon_user, -45, origin=ext[2]).bounds
)
box_user = shapely.affinity.rotate(box, 45, origin=ext[2])

# plot_coords(ax, polygon.interiors[0])
# plot_coords(ax, polygon.exterior)

patch = PolygonPatch(polygon_user, facecolor="b", edgecolor="b", alpha=0.5, zorder=2)
patch2 = PolygonPatch(polygon, facecolor="k", edgecolor="k", alpha=0.5, zorder=2)
patch3 = PolygonPatch(box_user, facecolor="b", edgecolor="b", alpha=0.5, zorder=2)
patch4 = PolygonPatch(box, facecolor="k", edgecolor="k", alpha=0.5, zorder=2)
ax.add_patch(patch2)
ax.add_patch(patch)
ax.add_patch(patch3)
ax.add_patch(patch4)
ax.set_aspect("equal")
ax.set_title("original")


plt.show()
print("Done")
