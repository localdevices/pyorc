import os.path
from pyorc import CameraConfig, Video
import matplotlib.pyplot as plt

def camera_config(
        video_file,
        cam_config_file,
        lens_position=None,
        corners=None,
        frame_sample=0.,
        **kwargs
):
    """
    Recipe for preparing a configuration file

    Parameters
    ----------
    video_file : str,
        Path to file with sample video containing objective of interest
    cam_config_file : str,
        Path to output file containing json camera config
    **kwargs: dict,
        Keyword arguments to pass to pyorc.CameraConfig (height and width are added on-the-fly from `video_file`)

    Returns
    -------
    None

    """
    # prepare file name for geographical and objective overview
    fn_geo = f"{os.path.splitext(cam_config_file)[0]}_geo.jpg"
    fn_cam = f"{os.path.splitext(cam_config_file)[0]}_cam.jpg"
    # open video for dimensions
    video = Video(video_file, start_frame=frame_sample, end_frame=frame_sample + 1)
    # extract first frame
    img = video.get_frame(frame_sample)
    img_rgb = video.get_frame(frame_sample, method="rgb")
    kwargs["height"], kwargs["width"] = img.shape
    # prepare camera config
    cam_config = CameraConfig(**kwargs)
    # write to output file
    if lens_position is not None:
        # set lens position assuming the same crs as the gcps
        cam_config.set_lens_position(*lens_position, crs=kwargs["gcps"]["crs"])
    if corners is not None:
        cam_config.set_bbox_from_corners(corners)
    cam_config.to_file(cam_config_file)
    if kwargs["crs"] is not None:
        ax = cam_config.plot(tiles="GoogleTiles", tiles_kwargs={"style": "satellite"})
    else:
        ax = cam_config.plot()
        ax.axis("equal")
    ax.figure.savefig(fn_geo)
    plt.close("all")
    f = plt.figure()
    ax = plt.axes()
    ax.imshow(img_rgb)
    cam_config.plot(ax=ax, camera=True)
    f.savefig(fn_cam)#, bbox_inches="tight")