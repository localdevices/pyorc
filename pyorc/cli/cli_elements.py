
import pyorc
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import Divider, Size
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

path_effects = [
    patheffects.Stroke(linewidth=2, foreground="w"),
    patheffects.Normal(),
]

corner_labels = [
    "upstream-left",
    "downstream-left",
    "downstream-right",
    "upstream-right"
]
class BaseSelect:
    def __init__(self, img, dst, buffer=0.0002, zoom_level=19):
        fig = plt.figure(figsize=(16, 9), frameon=False, facecolor="black")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        # fig = plt.figure(figsize=(12, 7))
        tiler = getattr(cimgt, "GoogleTiles")(style="satellite")
        xmin = np.array(dst)[:, 0].min()
        xmax = np.array(dst)[:, 0].max()
        ymin = np.array(dst)[:, 1].min()
        ymax = np.array(dst)[:, 1].max()
        extent = [xmin - buffer, xmax + buffer, ymin-buffer, ymax + buffer]
        # extent = [4.5, 4.51, 51.2, 51.21]
        ax_geo = fig.add_axes([0., 0., 1, 1], projection=tiler.crs)
        ax_geo.set_extent(extent, crs=ccrs.PlateCarree())
        ax_geo.add_image(tiler, zoom_level, zorder=1)
        ax_geo.set_visible(False)
        ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
        ax.set_facecolor("k")
        ax.set_position([0, 0, 1, 1])
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Left: add point, right: remove point, close: store in .src")
        # # make empty plot
        # self.p, = ax.plot([], [], "o", color="w", markeredgecolor="k", zorder=3)
        # TODO: ensure all coordinates are first transformed to latlon for plotting purposes
        # TODO: if no crs provided, then provide a normal axes with equal lengths on x and y axis
        self.p_geo = ax_geo.plot(*list(zip(*dst)), "o", color="w", markeredgecolor="k", zorder=3, transform=ccrs.PlateCarree())
        transform = ccrs.PlateCarree()._as_mpl_transform(ax_geo)
        for n, _pt in enumerate(dst):
            pt = ax_geo.annotate(
                n + 1,
                xytext=(4, 4),
                xy=_pt,
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="w"),
                    patheffects.Normal(),
                ],
                xycoords=transform
            )

        self.fig = fig
        self.ax_geo = ax_geo
        self.ax = ax  # add axes
        self.pts_t = []
        self.add_buttons()
        self.press = False
        self.move = False
        self.press_event = self.ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.release_event = self.ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.close_event = self.ax.figure.canvas.mpl_connect("close_event", self.on_close)
        self.move_event = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.required_clicks = None  # required clicks should be defined by the respective child class
        self.src = []
        self.dst = dst

    def add_buttons(self):
        h = [Size.Fixed(0.5), Size.Fixed(0.6)]
        v = [Size.Fixed(0.95), Size.Fixed(0.3)]
        v2 = [Size.Fixed(1.45), Size.Fixed(0.3)]
        v3 = [Size.Fixed(1.95), Size.Fixed(0.3)]

        divider1 = Divider(self.fig, (0, 0, 1, 1), h, v, aspect=False)
        divider2 = Divider(self.fig, (0, 0, 1, 1), h, v2, aspect=False)
        divider3 = Divider(self.fig, (0, 0, 1, 1), h, v3, aspect=False)
        self.ax_button1 = self.fig.add_axes(divider1.get_position(),
                                  axes_locator=divider1.new_locator(nx=1, ny=1))
        self.ax_button2 = self.fig.add_axes(divider2.get_position(),
                                  axes_locator=divider2.new_locator(nx=1, ny=1))
        self.ax_button3 = self.fig.add_axes(divider3.get_position(),
                                  axes_locator=divider3.new_locator(nx=1, ny=1))
        self.button1 = Button(self.ax_button1, 'Camera')
        self.button2 = Button(self.ax_button2, 'Map')
        self.button3 = Button(self.ax_button3, 'Done')
        self.button1.on_clicked(self.switch_to_ax)
        self.button2.on_clicked(self.switch_to_ax_geo)
        self.button3.on_clicked(self.close_window)
        self.button3.set_active(False)
        self.button3.label.set_color("gray")

    def close_window(self, event):
        # close window
        plt.close(self.fig)
        # exectute close event
        self.on_close(event)

    def on_close(self, event):
        print('disconnecting callback')
        self.ax.figure.canvas.mpl_disconnect(self.press_event)
        self.ax.figure.canvas.mpl_disconnect(self.release_event)
        self.ax.figure.canvas.mpl_disconnect(self.close_event)
        # check if the amount of src points and dst points is equal. If not return error
        assert len(self.src) == self.required_clicks, f"Discarding result because {len(self.src)} points were " \
                                                      f"selected, while {self.required_clicks} points are needed."

    def on_press(self, event):
        self.press = True
        self.move = False

    def on_move(self, event):
        # needed to detect movement and decide if a click if not a drag
        if self.press:
            self.move = True

    def on_click(self, event):
        # only if ax is visible and click was within the window
        if self.ax.get_visible() and event.inaxes == self.ax:
            if event.button is MouseButton.RIGHT:
                self.on_right_click(event)
            elif event.button is MouseButton.LEFT:
                if len(self.src) < self.required_clicks:
                    self.on_left_click(event)
                # check if enough points are collected to enable the Done button
            if len(self.src) == self.required_clicks:
                self.button3.set_active(True)
                self.button3.label.set_color("k")
            else:
                self.button3.set_active(False)
                self.button3.label.set_color("grey")
        self.ax.figure.canvas.draw()

    def on_left_click(self, event):
        if event.xdata is not None:
            print(f"Storing coordinate x: {event.xdata} y: {event.ydata} to src")
            self.src.append([int(np.round(event.xdata)), int(np.round(event.ydata))])
            self.p.set_data(*list(zip(*self.src)))
            pt = self.ax.annotate(
                len(self.src),
                xytext=(4, 4),
                xy=(event.xdata, event.ydata),
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="w"),
                    patheffects.Normal(),
                ],
            )
            self.pts_t.append(pt)

    def on_release(self, event):
        if self.press and not self.move:
            self.on_click(event)
        self.press = False
        self.move = False

    def on_right_click(self, event):
        # remove the last added point
        if len(self.pts_t) > 0:
            self.pts_t[-1].remove()
            del self.pts_t[-1]
        if len(self.src) > 0:
            del self.src[-1]
            if len(self.src) > 0:
                self.p.set_data(*list(zip(*self.src)))
            else:
                self.p.set_data([], [])

    def switch_to_ax(self, event):
        plt.sca(self.ax)
        self.ax.set_visible(True)
        self.ax_geo.set_visible(False)
        # plt.plot([1, 2, 3], [1, 2, 3])
        self.ax.figure.canvas.draw()

    # Define function for switching to ax2
    def switch_to_ax_geo(self, event):
        plt.sca(self.ax_geo)
        self.ax.set_visible(False)
        self.ax_geo.set_visible(True)
        # plt.plot([1, 2, 3], [3, 2, 1])
        self.ax_geo.figure.canvas.draw()


class AoiSelect(BaseSelect):
    """
    Selector tool to provide source GCP coordinates to pyOpenRiverCam
    """

    def __init__(self, img, src, dst, camera_config):
        super(AoiSelect, self).__init__(img, dst)
        # make empty plot
        self.camera_config = camera_config
        self.p_gcps, = self.ax.plot(*list(zip(*src)), "o", color="w", markeredgecolor="k", zorder=3)
        self.pts_t_gcps = [
            self.ax.annotate(
                n + 1,
                xytext=(4, 4),
                xy=xy,
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="w"),
                    patheffects.Normal(),
                ],
            ) for n, xy in enumerate(src)
        ]
        # self.pts_t.append(pt)

        self.p, = self.ax.plot([], [], "o", markersize=10, color="c", markeredgecolor="w", zorder=3)
        xloc = self.ax.get_xlim()[0] + 50
        yloc = self.ax.get_ylim()[-1] + 50
        self.title = self.ax.text(
            xloc,
            yloc,
            "Select corners of AOI in the following order with left and right as seen looking "
            "downstream:\nUpstream-left\nDownstream-left\nDownstream-right\nUpstream-right",
            size=12,
            va="top",
            path_effects=path_effects
        )

        # # TODO: if no crs provided, then provide a normal axes with equal lengths on x and y axis
        self.required_clicks = 4
        self.cam_config_cam = None
        self.cam_config_geo = None
        plt.show(block=True)

    def on_left_click(self, event):
        if event.xdata is not None:
            print(f"Storing coordinate x: {event.xdata} y: {event.ydata} to src")
            self.src.append([int(np.round(event.xdata)), int(np.round(event.ydata))])
            self.p.set_data(*list(zip(*self.src)))
            pt = self.ax.annotate(
                corner_labels[len(self.src) - 1],
                xytext=(4, 4),
                xy=(event.xdata, event.ydata),
                textcoords="offset points",
                zorder=4,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="w"),
                    patheffects.Normal(),
                ],
            )
            self.pts_t.append(pt)
            # check if all points are complete
            if len(self.src) == self.required_clicks:
                self.camera_config.set_bbox_from_corners(self.src)
                self.camera_config.plot_bbox(ax=self.ax, camera=True)
            else:
                # remove plot if present
                if self.cam_config_cam is not None:
                    self.cam_config_cam.remove()
                    self.cam_config_geo.remove()




class GcpSelect(BaseSelect):
    """
    Selector tool to provide source GCP coordinates to pyOpenRiverCam
    """

    def __init__(self, img, dst):
        super(GcpSelect, self).__init__(img, dst)
        # make empty plot
        self.p, = self.ax.plot([], [], "o", color="w", markeredgecolor="k", zorder=3)
        xloc = self.ax.get_xlim()[0] + 50
        yloc = self.ax.get_ylim()[-1] + 50
        self.title = self.ax.text(
            xloc,
            yloc,
            "Select location of control points in the right order (check locations on map view)",
            size=12,
            path_effects=path_effects
        )

        # # TODO: if no crs provided, then provide a normal axes with equal lengths on x and y axis
        self.required_clicks = len(self.dst)

