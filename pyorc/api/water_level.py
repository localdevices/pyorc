"""WaterLevel module for pyorc."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry

MODES = Literal["camera", "geographic", "cross_section"]


class WaterLevel:
    """Water Level functionality."""

    def __str__(self):
        return str(self)

    def __repr__(self):
        return self

    def __init__(self, camera_config, cross_section: Union[gpd.GeoDataFrame, List[List]]):
        """Initialize WaterLevel class for establishing estimated water levels."""
        # if cross_section is a GeoDataFrame, check if it has a CRS, if yes, convert coordinates to crs of CameraConfig
        if isinstance(cross_section, gpd.GeoDataFrame):
            if cross_section.crs is not None and camera_config.crs is not None:
                cross_section.to_crs(camera_config.crs, inplace=True)
            elif cross_section.crs is not None or camera_config.crs is not None:
                raise ValueError("if a CRS is used, then both camera_config and cross_section must have a CRS.")
            g = cross_section.geometry
            x, y, z = g.x, g.y, g.z
        else:
            x, y, z = list(map(list, zip(*cross_section, strict=False)))

        x_diff = np.concatenate((np.array([0]), np.diff(x)))
        y_diff = np.concatenate((np.array([0]), np.diff(y)))
        # estimate distance from left to right bank
        s = np.cumsum((x_diff**2 + y_diff**2) ** 0.5)

        self.x = x  # x-coordinates in local projection or crs
        self.y = y  # y-coordinates in local projection or crs
        self.z = z  # z-coordinates in local projection or crs
        self.s = s  # distance from left to right bank (only used for interpolation funcs).
        self.camera_config = camera_config

    @property
    def cs_points(self):
        """Return cross section as list of shapely.geometry.Point."""
        return

    @property
    def cs_linestring(self):
        """Return cross section as shapely.geometry.Linestring."""
        return

    @property
    def cs_angle(self):
        """Average angle orientation that cross section makes in geographical space.

        Zero means ....
        """
        return

    def get_xyz(self, h: float, camera: bool = False) -> List[geometry.Point]:
        """Retrieve list of xyz points, where cross section is crossed by value z. Multiple crossings may be found."""
        z = self.camera_config.h_to_z(h)
        if z > self.z.max() or z < self.z.min():
            raise ValueError(
                "Value of water level is lower (higher) than the minimum (maximum) value found in the cross section"
            )
        return []

    def get_waterline(self, h: float, length=0.5, offset=0.0, camera: bool = False) -> List[geometry.LineString]:
        """Retrieve waterlines over the cross section, perpendicular to the orientation of the cross section.

        Returns a 2D LineString if camera is True, 3D if False

        """
        return []

    def get_planar_surface(
        self, h: float, length: float = 0.5, offset: Optional[float] = None, camera: bool = False
    ) -> geometry.Polygon:
        """Retrieve a planar water surface for a given water level, as a geometry.Polygon.

        Returns a 2D Polygon if camera is True, 3D if False

        """
        wls = self.get_waterline(h=h)
        if len(wls) != 2:
            raise ValueError("Amount of water line crossings must be 2 for a planar surface estimate.")

    def get_wetted_surface(self, h: float, camera: bool = False) -> geometry.Polygon:
        """Retrieve a wetted surface for a given water level, as a geometry.Polygon."""
        wls = self.get_xyz(h=h)
        if len(wls) != 2:
            raise ValueError("Amount of water line crossings must be 2 for a planar surface estimate.")

    def get_waterline_pol(
        self, h: float, length: float = 0.5, padding: Tuple[float] = (0.5, 0.5), offset: Optional[float] = None
    ) -> List[geometry.Polygon]:
        """Get vertically oriented polygons, with orientation along the waterline and with provided vertical spacing.

        Returns a 2D Polygon if camera is True, 3D if False

        """
        return []

    def plot(self, mode=MODES, h: Optional[float] = None, ax: Optional[mpl.axes.Axes] = None) -> mpl.axes.Axes:
        """Plot the situation.

        Plotting can be done geographic (planar), from cross section perspective in the camera perspective or 3D.
        """
        if not ax:
            if mode == "3d":
                f, ax = plt.subplots(1, 1, projection="3d")
            else:
                f, ax = plt.subplots(1, 1)
        return ax

    def detect_wl(self, img: np.ndarray, step: float = 0.05):
        """Attempt to detect the water line level along the cross section, using a provided pre-treated image."""
        if len(img.shape) == 3:
            # flatten image first
            img = img.mean(axis=2)
        assert (
            img.shape[0] == self.camera_config.height
        ), f"Image height {img.shape[0]} is not the same as camera_config height {self.camera_config.height}"
        assert (
            img.shape[1] == self.camera_config.width
        ), f"Image width {img.shape[1]} is not the same as camera_config width {self.camera_config.width}"
        for _ in np.arange(self.z.min(), self.z.max(), step):
            # TODO implement detection
            pass

            # TODO: do an optimization
        z = None
        # finally, return water level:
        return self.camera_config.h_to_z(z)
