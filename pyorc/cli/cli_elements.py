
import pyorc
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patheffects
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import Divider, Size
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


class gcp_select:
    """
    Selector tool to provide source GCP coordinates to pyOpenRiverCam
    """

    def __init__(self, img, dst, button=None):
        fig = plt.figure(figsize=(12, 7))
        tiler = getattr(cimgt, "GoogleTiles")(style="satellite")
        extent = [4.5, 4.51, 51.2, 51.21]
        ax_geo = fig.add_axes([0.2, 0.1, 0.7, 0.8], projection=tiler.crs)
        ax_geo.set_extent(extent, crs=ccrs.PlateCarree())
        ax_geo.add_image(tiler, 16, zorder=1)
        ax_geo.set_visible(False)
        ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
        # fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Left: add point, right: remove point, close: store in .src")
        # make empty plot
        self.p, = ax.plot([], [], "o", color="w", markeredgecolor="k", zorder=3)
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
        self.src = []
        self.dst = dst
        plt.show(block=True)

    def add_buttons(self):
        h = [Size.Fixed(0.5), Size.Fixed(0.6)]
        v = [Size.Fixed(0.95), Size.Fixed(0.3)]
        v2 = [Size.Fixed(1.95), Size.Fixed(0.3)]
        v3 = [Size.Fixed(2.95), Size.Fixed(0.3)]

        divider1 = Divider(self.fig, (0, 0, 1, 1), h, v, aspect=False)
        divider2 = Divider(self.fig, (0, 0, 1, 1), h, v2, aspect=False)
        divider3 = Divider(self.fig, (0, 0, 1, 1), h, v3, aspect=False)
        self.ax_button1 = self.fig.add_axes(divider1.get_position(),
                                  axes_locator=divider1.new_locator(nx=1, ny=1))
        self.ax_button2 = self.fig.add_axes(divider2.get_position(),
                                  axes_locator=divider2.new_locator(nx=1, ny=1))
        self.ax_button3 = self.fig.add_axes(divider3.get_position(),
                                  axes_locator=divider3.new_locator(nx=1, ny=1))
        self.button1 = Button(self.ax_button1, 'Show camera')
        self.button2 = Button(self.ax_button2, 'Show map')
        self.button3 = Button(self.ax_button3, 'Done')
        self.button1.on_clicked(self.switch_to_ax)
        self.button2.on_clicked(self.switch_to_ax_geo)
        self.button3.on_clicked(self.close_window)
        self.button3.set_active(False)
        self.button3.label.set_color("gray")
        # self.button3.drawon()

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
        assert len(self.src) == len(
            self.dst), f"Discarding result because {len(self.src)} points were selected, while {len(self.dst)} points are needed."

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
                if len(self.src) < len(self.dst):
                    self.on_left_click(event)
                # check if enough points are collected to enable the Done button
            if len(self.src) == len(self.dst):
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
