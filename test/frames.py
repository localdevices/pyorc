import OpenRiverCam
import cv2
import os

fn = os.path.abspath("../../PIV.TUDelft/Example notebooks/example_video.mp4")
print(fn)
dst = r"/home/hcwinsemius/OpenRiverCam/with_lens"

lens_pars = {
    "k1": -10.0e-6,
    "c": 2,
    "f": 8.0,
}
logger = OpenRiverCam.log.start_logger(True, False)
# do frame extraction
n = 0
for t, buf in OpenRiverCam.io.frames(fn, dst, start_frame=100, lens_pars=lens_pars):
    print(t)
    n += 1
    buf.seek(0)
    dst_fn = os.path.join(dst, '_{:04d}.jpg'.format(n))
    with open(dst_fn, 'wb') as f:
        f.write(buf.read())

print("Done")
# read one frame back and plot
img = cv2.imread(dst_fn, 0)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
