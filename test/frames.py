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
t, fns = OpenRiverCam.io.frames(fn, dst, lens_pars=lens_pars, logger=logger)

# read one frame back and plot
img = cv2.imread(fns[-1], 0)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
