"""WaterLevel module for pyorc."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from shapely import affinity, geometry
from shapely.ops import polygonize

MODES = Literal["camera", "geographic", "cross_section"]


class CrossSection:
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
    def interp_x(self) -> interp1d:
        """Linear interpolation function for x-coordinates."""
        return interp1d(self.s, self.x, kind="linear", fill_value="extrapolate")

    @property
    def interp_y(self) -> interp1d:
        """Linear interpolation function for x-coordinates."""
        return interp1d(self.s, self.y, kind="linear", fill_value="extrapolate")

    @property
    def cs_points(self) -> List[geometry.Point]:
        """Cross section as list of shapely.geometry.Point."""
        return [geometry.Point(_x, _y, _z) for _x, _y, _z in zip(self.x, self.y, self.z, strict=False)]

    @property
    def cs_points_yz(self) -> List[geometry.Point]:
        """Cross section as list of shapely.geometry.Point."""
        return [geometry.Point(_s, _z) for _s, _z in zip(self.s, self.z, strict=False)]

    @property
    def cs_linestring(self) -> geometry.LineString:
        """Cross section as shapely.geometry.Linestring."""
        return geometry.LineString(self.cs_points)

    @property
    def cs_linestring_yz(self) -> geometry.LineString:
        """Cross section perpendicular to flow direction (YZ) as shapely.geometry.Linestring."""
        return geometry.LineString(self.cs_points_yz)

    @property
    def cs_angle(self):
        """Average angle orientation that cross section makes in geographical space.

        Zero means left to right direction, positive means counter-clockwise, negative means clockwise.
        """
        point1_xy = self.cs_points[0].coords[0][:-1]
        point2_xy = self.cs_points[-1].coords[0][:-1]
        diff_xy = np.array(point2_xy) - np.array(point1_xy)
        return np.arctan2(diff_xy[1], diff_xy[0])

    def get_cs_waterlevel(self, h: float, yz=False) -> geometry.LineString:
        """Retrieve LineString of water surface at cross section at a given water level."""
        # get water level in camera config vertical datum
        z = self.camera_config.h_to_z(h)
        if yz:
            return geometry.LineString(zip(self.s, [z] * len(self.s), strict=False))
        return geometry.LineString(zip(self.x, self.y, [z] * len(self.x), strict=False))

    def get_csl_point(self, h: float, camera: bool = False) -> List[geometry.Point]:
        """Retrieve list of points, where cross section (cs) touches the land (l). Multiple points may be found."""
        # get water level in camera config vertical datum
        z = self.camera_config.h_to_z(h)
        if z > np.array(self.z).max() or z < np.array(self.z).min():
            raise ValueError(
                "Value of water level is lower (higher) than the minimum (maximum) value found in the cross section"
            )
        cs_waterlevel = self.get_cs_waterlevel(h, yz=True)
        # get crossings and turn into list of geometry.Point
        cross_yz = cs_waterlevel.intersection(self.cs_linestring_yz)

        # make cross_yz iterable
        if isinstance(cross_yz, geometry.Point):
            cross_yz = [cross_yz]
        elif isinstance(cross_yz, geometry.MultiPoint):
            cross_yz = list(cross_yz.geoms)
        else:
            raise ValueError(
                "Cross section is not crossed by water level. Check if water level is within the cross section."
            )
        # find xyz real-world coordinates and turn in to POINT Z list
        cross = [geometry.Point(self.interp_x(c.xy[0]), self.interp_y(c.xy[0]), c.xy[1]) for c in cross_yz]

        if camera:
            coords = [[p.x, p.y, p.z] for p in cross]
            coords_proj = self.camera_config.project_points(coords, swap_y_coords=True)
            cross = [geometry.Point(p[0], p[1]) for p in coords_proj]
        return cross

    def get_csl_line(self, h: float, length=0.5, offset=0.0, camera: bool = False) -> List[geometry.LineString]:
        """Retrieve waterlines over the cross section, perpendicular to the orientation of the cross section.

        Returns a 2D LineString if camera is True, 3D if False

        """
        z = self.camera_config.h_to_z(h)
        # acquire angle perpendicular to cross section
        angle_perp = self.cs_angle + np.pi / 2
        # retrieve xyz coordinates of cross-section land crossings
        csl_points = self.get_csl_point(h=h)

        # move points
        csl_points = [
            affinity.translate(p, xoff=np.cos(angle_perp) * offset, yoff=np.sin(angle_perp) * offset)
            for p in csl_points
        ]

        # make horizontally oriented lines of the required length (these are only xy at this stage)
        csl_lines = [
            geometry.LineString([geometry.Point(p.x - length / 2, p.y), geometry.Point(p.x + length / 2, p.y)])
            for p in csl_points
        ]
        # rotate in counter-clockwise perpendicular direction to the orientation of the cross section itself
        csl_lines = [
            affinity.rotate(l, angle_perp, origin=p, use_radians=True)
            for l, p in zip(csl_lines, csl_points, strict=False)
        ]

        if camera:
            # transform to 2D projected lines, make list of lists of coordinates
            coords_lines = [[[_x, _y, z] for _x, _y in l.coords] for l in csl_lines]
            # project list of lists
            coords_lines_proj = [self.camera_config.project_points(cl, swap_y_coords=True) for cl in coords_lines]
            # turn list of lists into list of LineString
            csl_lines = [geometry.LineString([geometry.Point(_x, _y) for _x, _y in l]) for l in coords_lines_proj]
        else:
            # add a z-coordinate and return
            csl_lines = [geometry.LineString([geometry.Point(_x, _y, z) for _x, _y in l.coords]) for l in csl_lines]
        return csl_lines

    def get_csl_pol(
        self,
        h: float,
        length: float = 0.5,
        padding: Tuple[float, float] = (-0.5, 0.5),
        offset: float = 0.0,
        camera: bool = False,
    ) -> List[geometry.Polygon]:
        """Get horizontal polygon from cross-section land line towards water or towards land.

        Returns a 2D Polygon if camera is True, 3D if False

        """
        # retrieve water line(s)
        csl = self.get_csl_line(h=h, length=length, offset=offset)
        if len(padding) != 2:
            raise ValueError(f"padding must contain two values (provided: {len(padding)}")
        if padding[1] <= padding[0]:
            raise ValueError("First value of padding must be smaller than second")
        csl_pol_bounds = [
            [
                affinity.translate(
                    line, xoff=np.cos(self.cs_angle) * padding[0], yoff=np.sin(self.cs_angle) * padding[0]
                ),
                affinity.translate(
                    line, xoff=np.cos(self.cs_angle) * padding[1], yoff=np.sin(self.cs_angle) * padding[1]
                ),
            ]
            for line in csl
        ]
        csl_pol_coords = [
            list(lines[0].coords) + list(lines[1].coords[::-1]) + [lines[0].coords[0]] for lines in csl_pol_bounds
        ]
        if camera:
            csl_pol_coords = [
                self.camera_config.project_points(coords, swap_y_coords=True) for coords in csl_pol_coords
            ]
        polygons = [geometry.Polygon(coords) for coords in csl_pol_coords]
        return polygons

    def get_planar_surface(
        self, h: float, length: float = 0.5, offset: float = 0.0, camera: bool = False
    ) -> geometry.Polygon:
        """Retrieve a planar water surface for a given water level, as a geometry.Polygon.

        Returns a 2D Polygon if camera is True, 3D if False

        """
        wls = self.get_csl_line(h=h, offset=offset, length=length, camera=camera)
        if len(wls) != 2:
            raise ValueError("Amount of water line crossings must be 2 for a planar surface estimate.")
        return geometry.Polygon(list(wls[0].coords) + list(wls[1].coords[::-1]))

    def get_wetted_surface_yz(self, h: float) -> geometry.Polygon:
        """Retrieve a wetted surface for a given water level, as a geometry.Polygon."""
        wl = self.get_cs_waterlevel(h=h, yz=True)
        # create polygon by making a union
        pol = list(polygonize(wl.union(self.cs_linestring_yz)))
        if len(pol) == 0:
            raise ValueError("Water level is not crossed by cross section and therefore undefined.")
        elif len(pol) > 1:
            raise ValueError("Water level is crossed by multiple polygons.")
        else:
            pol = pol[0]
        return pol

    def get_wetted_surface(self, h: float, camera: bool = False) -> geometry.Polygon:
        """Retrieve a wetted surface for a given water level, as a geometry.Polygon."""
        pol = self.get_wetted_surface_yz(h=h)
        coords = [[self.interp_x(p[0]), self.interp_y(p[0]), p[1]] for p in pol.exterior.coords]
        if camera:
            coords_proj = self.camera_config.project_points(coords, swap_y_coords=True)
            return geometry.Polygon(coords_proj)
        else:
            return geometry.Polygon(coords)

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

    def plot_cs(self, ax=None, camera=False, **kwargs):
        """Plot the cross section."""
        if not ax:
            ax = plt.axes()
        if ax.name == "3d" and not camera:
            # map 3d coordinates to x, y, z
            x, y, z = zip(*[(c[0], c[1], c[2]) for c in self.cs_linestring.coords], strict=False)
            p = ax.plot(x, y, z, **kwargs)
        else:
            if camera:
                # extract pixel locations
                pix = self.camera_config.project_points(list(map(list, self.cs_linestring.coords)), swap_y_coords=True)
                # map to x and y arrays
                x, y = zip(*[(c[0], c[1]) for c in pix], strict=False)
            else:
                x, y = zip(*[(c[0], c[1]) for c in self.cs_linestring_yz.coords], strict=False)
            p = ax.plot(x, y, **kwargs)
        return p

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
