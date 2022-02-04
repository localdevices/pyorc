import pyorc
import cv2
import os

fn = os.path.abspath("/home/hcwinsemius/Media/projects/OpenRiverCam/example_video.mp4")
# fn = os.path.abspath("/home/hcwinsemius/Downloads/1080p_25fps_4M_CBR.mkv")
print(fn)
dst = r"/home/hcwinsemius/Media/projects/OpenRiverCam/with_lens_color"
prefix = "frame"
lens_pars = {
    "k1": -10.0e-6,
    "c": 2,
    "f": 8.0,
}
# lens_pars = {
#     "k1": 0.,
#     "c": 2,
#     "f": 1.,
# }

logger = pyorc.log.start_logger(True, False)
# do frame extraction
n = 0
t = 0
if not (os.path.isdir(dst)):
    os.makedirs(dst)
for _t, img in pyorc.io.frames(
    fn, dst, start_frame=0, grayscale=False, lens_pars=lens_pars
):

    # print(_t, int((_t - t)*1000))

    n += 1
    ret, im_en = cv2.imencode(".jpg", img)
    dst_fn = os.path.join(
        dst, "{:s}_{:04d}_{:06d}.jpg".format(prefix, n, int(_t * 1000))
    )

    # dst_fn = os.path.join(dst, "_{:04d}.jpg".format(n))
    with open(dst_fn, "wb") as f:
        f.write(im_en)
    t = _t
print("Done")
# read one frame back and plot
img = cv2.imread(dst_fn, 0)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
