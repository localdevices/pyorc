import openpiv.tools
import openpiv.pyprocess
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

folder = r"/home/hcwinsemius/OpenRiverCam"

src = os.path.join(folder, "ortho_proj")
dst = os.path.join(folder, "piv")

fns = glob.glob(os.path.join(src, "*.tif"))
fns.sort()
print(fns)



for n in range(len(fns)-16):
    print(f"Treating frame {n}")
    frame_a = openpiv.tools.imread(fns[n])
    frame_b = openpiv.tools.imread(fns[n+1])
    u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv( frame_a, frame_b, window_size=60, overlap=30, search_area_size=60, dt=1./25)
    cols, rows = openpiv.pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=60, overlap=30)



with rasterio.open(fns[0]) as ds:
    print(ds.transform)
    band = ds.read(1)
    shape = band.shape
    coli, rowi = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xi, yi = rasterio.transform.xy(ds.transform, rowi, coli)
    xi, yi = np.array(xi), np.array(yi)
    xq, yq = rasterio.transform.xy(ds.transform, rows, cols)
    xq, yq = np.array(xq), np.array(yq)
    plt.pcolormesh(xi, yi, band)

    # show(ds.read(1), transform=ds.transform)
#
#
# frame_cv = cv2.imread(fns[0])
#
# plt.imshow(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
plt.quiver(xq, yq, u, v, color='b')
plt.show()
