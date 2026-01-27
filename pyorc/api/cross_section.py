"""WaterLevel module for pyorc."""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from shapely import affinity, force_2d, force_3d, geometry
from shapely.affinity import rotate, translate
from shapely.ops import polygonize, split

from .cameraconfig import CameraConfig, cv, plot_helpers

MODES = Literal["camera", "geographic", "cross_section"]
PATH_EFFECTS = [
    patheffects.Stroke(linewidth=3, foreground="w"),
    patheffects.Normal(),
]
LINE_COLOR = "#385895"
PLANAR_KWARGS = {"color": "c", "alpha": 0.5}
BOTTOM_KWARGS = {"color": "brown", "alpha": 0.1}
BANK_OPTIONS = {"far", "near", "both"}
WETTED_KWARGS = {
    "alpha": 0.15,
    "linewidth": 2.0,
    "facecolor": LINE_COLOR,
    "path_effects": PATH_EFFECTS,
    "edgecolor": "w",
    "zorder": 1,
}


def _make_angle_lines(csl_points, angle_perp, length, offset):
    """Make lines at cross section points, perpendicular to cross section orientation using angle and offset."""
    # move points
    csl_points = [
        affinity.translate(p, xoff=np.cos(angle_perp) * offset, yoff=np.sin(angle_perp) * offset) for p in csl_points
    ]

    # make horizontally oriented lines of the required length (these are only xy at this stage)
    csl_lines = [
        geometry.LineString([geometry.Point(p.x - length / 2, p.y), geometry.Point(p.x + length / 2, p.y)])
        for p in csl_points
    ]
    # rotate in counter-clockwise perpendicular direction to the orientation of the cross section itself
    csl_lines = [affinity.rotate(l, angle_perp, origin=p, use_radians=True) for l, p in zip(csl_lines, csl_points)]
    return csl_lines


def _histogram(data, bin_size: int = 5, normalize=False):
    """Create a histogram with predefined bin size."""
    bin_size = int(bin_size)
    if not data.dtype == np.uint8:
        raise ValueError("Image data must be of type uint8.")
    if not (bin_size >= 5 and bin_size <= 20):
        raise ValueError("Bin size must be between 5 and 20")
    bins = np.arange(0, 256, bin_size)
    # Compute histogram counts
    counts, edges = np.histogram(data, bins=bins)

    # Normalize counts if required
    if normalize and np.sum(counts) > 0:
        bin_widths = np.diff(edges)
        counts = counts / (np.sum(counts) * bin_widths)  # Normalize to sum to 1 over the distribution

    # Calculate bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, edges, counts


def _histogram_union(edges, hist1, hist2):
    """Measures the union of two normalized histograms and turns into score. They must have the same bins.

    A score of 0 means the histograms are entirely different, score of 1 means they are identical.
    """
    bin_chunks = edges[1:] - edges[:-1]
    # take the piecewise maximum of histograms
    hist_max = np.maximum(hist1, hist2)
    # integrate
    union = (bin_chunks * hist_max).sum()
    # normalize score: 1 least optimal, 0 most optimal.
    return 2 - union


class CrossSection:
    """Water Level functionality."""

    def __str__(self):
        return str(self.cs_linestring)

    def __repr__(self):
        return str(self.cs_linestring)

    def __init__(self, camera_config: CameraConfig, cross_section: Union[gpd.GeoDataFrame, List[List]]):
        """Initialize a cross-section representation.

        The object is geographically aware through the camera configuration.
        If a GeoDataFrame is supplied as `cross_section` input, CRS consistency
        between the cross-section and camera configuration is guaranteed.

        Parameters
        ----------
        camera_config : pyorc.CameraConfig
            Defines the camera configuration, including potential CRS used for processing
            geographic data consistency.
        cross_section : GeoDataFrame or list of lists
            Cross-sectional data representing coordinate information. Can either be a GeoDataFrame
            with CRS potentially set or a list-of-lists where each inner list contains [x, y, z] coordinates.

        Attributes
        ----------
        x : list of float
            Represents the x-coordinates of the cross-section. For GeoDataFrame input, these are
            converted into the CRS of the camera configuration if applicable.
        y : list of float
            Represents the y-coordinates of the cross-section. For GeoDataFrame input, these are
            converted into the CRS of the camera configuration if applicable.
        z : list of float
            Represents the z-coordinates of the cross-section.
        s : numpy.ndarray
            Represents the horizontal cumulative distances calculated between consecutive points
            in the cross-section, used primarily for spatial interpolation.
        l : numpy.ndarray
            Represents the cumulative spatial coordinates, tracking both horizontal and vertical
            distances over the cross-section, from the left-most to the right-most section.
            used primarily for spatial interpolation and needed where pure horizontal or vertical lines are experienced.
        camera_config : object
            Associated camera configuration object indicating CRS and related configuration applied
            to handle projection and compatibility with cross-section data.

        Raises
        ------
        ValueError
            Raised if the CRS is defined for only one of `camera_config` or `cross_section`,
            as CRS for both or neither must be specified for consistency.
        TypeError
            Raised if `cross_section` is neither of type GeoDataFrame nor a list of lists structure.

        """
        # if cross_section is a GeoDataFrame, check if it has a CRS, if yes, convert coordinates to crs of CameraConfig
        if isinstance(cross_section, gpd.GeoDataFrame):
            crs_cs = getattr(cross_section, "crs", None)
            crs_cam = getattr(camera_config, "crs", None)
            if crs_cs is not None and crs_cam is not None:
                cross_section.to_crs(camera_config.crs, inplace=True)
            elif crs_cs is not None or crs_cam is not None:
                raise ValueError("if a CRS is used, then both camera_config and cross_section must have a CRS.")
            g = cross_section.geometry
            x, y, z = g.x.values, g.y.values, g.z.values
        else:
            x, y, z = list(map(list, zip(*cross_section)))

        x_diff = np.concatenate((np.array([0]), np.diff(x)))
        y_diff = np.concatenate((np.array([0]), np.diff(y)))
        z_diff = np.concatenate((np.array([0]), np.diff(z)))
        # estimate distance from left to right bank
        s = np.cumsum((x_diff**2 + y_diff**2) ** 0.5)
        lens_position_xy = camera_config.estimate_lens_position()[0:2]
        d = ((x - lens_position_xy[0]) ** 2 + (y - lens_position_xy[1]) ** 2) ** 0.5
        # estimate length coordinates
        l = np.cumsum(np.sqrt(x_diff**2 + y_diff**2 + z_diff**2))

        self.x = np.array(x)  # x-coordinates in local projection or crs
        self.y = np.array(y)  # y-coordinates in local projection or crs
        self.z = np.array(z)  # z-coordinates in local projection or crs
        self.s = s  # horizontal distance from left to right bank (only used for interpolation funcs).
        self.l = l  # length distance (following horizontal and vertical) over cross section from left to right
        self.d = d  # horizontal distance between lens horizontal position and cross section point
        self.camera_config = camera_config

    @property
    def interp_x(self) -> interp1d:
        """Linear interpolation function for x-coordinates, using l as input."""
        return interp1d(self.l, self.x, kind="linear", fill_value="extrapolate")

    @property
    def interp_y(self) -> interp1d:
        """Linear interpolation function for y-coordinates, using l as input."""
        return interp1d(self.l, self.y, kind="linear", fill_value="extrapolate")

    @property
    def interp_z(self) -> interp1d:
        """Linear interpolation function for z-coordinates, using l as input."""
        return interp1d(self.l, self.z, kind="linear", fill_value="extrapolate")

    @property
    def interp_d(self) -> interp1d:
        """Linear interpolation function for distances (d) to lens, using l as input."""
        return interp1d(self.l, self.d, kind="linear", fill_value="extrapolate")

    @property
    def interp_x_from_s(self) -> interp1d:
        """Linear interpolation function for x-coordinates, using s as input."""
        return interp1d(self.s, self.x, kind="linear", fill_value="extrapolate")

    @property
    def interp_y_from_s(self) -> interp1d:
        """Linear interpolation function for y-coordinates, using s as input.."""
        return interp1d(self.s, self.y, kind="linear", fill_value="extrapolate")

    @property
    def interp_z_from_s(self) -> interp1d:
        """Linear interpolation function for z-coordinates, using s as input.."""
        return interp1d(self.s, self.z, kind="linear", fill_value="extrapolate")

    @property
    def interp_s_from_l(self) -> interp1d:
        """Linear interpolation function for s-coordinates, from left to right bank, using l as input."""
        return interp1d(self.l, self.s, kind="linear", fill_value="extrapolate")

    @property
    def cs_points(self) -> List[geometry.Point]:
        """Return cross-section as list of shapely.geometry.Point."""
        return [geometry.Point(_x, _y, _z) for _x, _y, _z in zip(self.x, self.y, self.z)]

    @property
    def cs_points_sz(self) -> List[geometry.Point]:
        """Return cross-section perpendicular to flow direction (SZ) as list of shapely.geometry.Point."""
        return [geometry.Point(_s, _z) for _s, _z in zip(self.s, self.z)]

    @property
    def cs_linestring(self) -> geometry.LineString:
        """Return cross-section as shapely.geometry.Linestring."""
        return geometry.LineString(self.cs_points)

    @property
    def cs_linestring_sz(self) -> geometry.LineString:
        """Return cross-section perpendicular to flow direction (SZ) as shapely.geometry.Linestring."""
        return geometry.LineString(self.cs_points_sz)

    @property
    def cs_angle(self):
        """Average angle orientation that cross-section makes in geographical space.

        Zero means left to right direction, positive means counter-clockwise, negative means clockwise.
        """
        point1_xy = self.cs_points[0].coords[0][:-1]
        point2_xy = self.cs_points[-1].coords[0][:-1]
        diff_xy = np.array(point2_xy) - np.array(point1_xy)
        return np.arctan2(diff_xy[1], diff_xy[0])

    @property
    def distance_camera(self):
        """Estimate distance of mean coordinate of cross section to camera position."""
        coord_mean = np.mean(self.cs_linestring.coords, axis=0)
        return np.sum((self.camera_config.estimate_lens_position() - coord_mean) ** 2) ** 0.5

    @property
    def idx_closest_point(self):
        """Determine index of point in cross-section, closest to the camera."""
        return 0 if self.d[0] < self.d[-1] else len(self.d) - 1

    @property
    def idx_farthest_point(self):
        """Determine index of point in cross-section, farthest from the camera."""
        return 0 if self.d[0] > self.d[-1] else len(self.d) - 1

    @property
    def within_image(self):
        """Check if any of the points of the cross section fall inside the image objective."""
        # check if cross section is visible within the image objective
        pix = self.camera_config.project_points(np.array(list(map(list, self.cs_linestring.coords))), within_image=True)
        # check which points fall within the image objective
        within_image = np.all(
            [
                pix[:, 0] >= 0,
                pix[:, 0] < self.camera_config.width,
                pix[:, 1] >= 0,
                pix[:, 1] < self.camera_config.height,
            ],
            axis=0,
        )
        # check if there are any points within the image objective and return result
        return bool(np.any(within_image))

    def get_cs_waterlevel(self, h: float, sz=False, extend_by=None) -> geometry.LineString:
        """Retrieve LineString of water surface at cross-section at a given water level.

        Parameters
        ----------
        h : float
            water level [m].
        sz : bool, optional
            If set, return water level line in y-z projection, by default False.
        extend_by : float, optional
            If set, the line will be extended left and right using the defined float in meters

        Returns
        -------
        geometry.LineString
            horizontal line at water level (2d if `sz`=True, 3d if `yz`=False)

        """
        # get water level in camera config vertical datum
        z = self.camera_config.h_to_z(h)
        if sz:
            if extend_by is None:
                s_coords = self.s
            else:
                s_coords = np.concatenate([[-np.abs(extend_by)], self.s, [self.s[-1] + np.abs(extend_by)]])
            return geometry.LineString(zip(s_coords, [z] * len(s_coords)))
        if extend_by is not None:
            alpha = np.arctan((self.x[1] - self.x[0]) / (self.y[1] - self.y[0]))
            x_coords = np.concatenate(
                [
                    [self.x[0] - np.cos(alpha) * np.abs(extend_by)],
                    self.x,
                    [self.x[-1] + np.cos(alpha) * np.abs(extend_by)],
                ]
            )
            y_coords = np.concatenate(
                [
                    [self.y[0] - np.sin(alpha) * np.abs(extend_by)],
                    self.y,
                    [self.y[-1] + np.sin(alpha) * np.abs(extend_by)],
                ]
            )
        else:
            x_coords = self.x
            y_coords = self.y

        return geometry.LineString(zip(x_coords, y_coords, [z] * len(x_coords)))

    def get_csl_point(
        self, h: Optional[float] = None, l: Optional[float] = None, camera: bool = False, swap_y_coords: bool = False
    ) -> List[geometry.Point]:
        """Retrieve list of points, where cross-section (cs) touches the land (l).

        Multiple points may be found.

        Parameters
        ----------
        h : float, optional
            water level [m], if provided, s must not be provided.
        l : float
            coordinate of distance from left to right bank, including height [m], if provided, h must not be provided.
        camera : bool, optional
            If set, return 2D projected points, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        List[shapely.geometry.Point]
            List of points, where water line touches land, can be only one or two points.

        """
        if h is not None and l is not None:
            raise ValueError("Only one of h or l can be provided.")
        if h is None and l is None:
            raise ValueError("One of h or l must be provided.")
        # get water level in camera config vertical datum
        if l is not None:
            if l < 0 or l > self.l[-1]:
                raise ValueError(
                    "Value of l is lower (higher) than the minimum (maximum) value found in the cross section"
                )
            cross = [geometry.Point(self.interp_x(l), self.interp_y(l), self.interp_z(l))]
        else:
            z = self.camera_config.h_to_z(h)
            if z > np.array(self.z).max() or z < np.array(self.z).min():
                raise ValueError(
                    "Value of water level is lower (higher) than the minimum (maximum) value found in the cross section"
                )
            cs_waterlevel = self.get_cs_waterlevel(h, sz=True)
            # get crossings and turn into list of geometry.Point
            cross_sz = cs_waterlevel.intersection(self.cs_linestring_sz)

            # make cross_yz iterable
            if isinstance(cross_sz, geometry.Point):
                cross_sz = [cross_sz]
            elif isinstance(cross_sz, geometry.MultiPoint):
                cross_sz = list(cross_sz.geoms)
            else:
                raise ValueError(
                    "Cross section is not crossed by water level. Check if water level is within the cross section."
                )
            # sort the cross_sz points to go from s=0 to s max.
            cross_sz = sorted(cross_sz, key=lambda p: p.x)
            # find xyz real-world coordinates and turn in to POINT Z list
            cross = [
                geometry.Point(self.interp_x_from_s(c.xy[0]), self.interp_y_from_s(c.xy[0]), c.xy[1]) for c in cross_sz
            ]

        if camera:
            coords = [[p.x, p.y, p.z] for p in cross]
            coords_proj = self.camera_config.project_points(coords, swap_y_coords=swap_y_coords)
            cross = [geometry.Point(p[0], p[1]) for p in coords_proj]
        return cross

    def get_csl_line(
        self,
        h: Optional[float] = None,
        l: Optional[float] = None,
        length: float = 0.5,
        offset: float = 0.0,
        camera: bool = False,
        swap_y_coords: bool = False,
    ) -> List[geometry.LineString]:
        """Retrieve waterlines over the cross-section, perpendicular to the orientation of the cross-section.

        Returns a 2D LineString if camera is True, 3D if False

        Parameters
        ----------
        h : float, optional
            water level [m]
        l : float
            coordinate of distance from left to right bank, including height [m], if provided, h must not be provided.
        length : float, optional
            length of the waterline [m], by default 0.5
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected lines, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        List[shapely.geometry.LineString]
            List of lines perpendicular to cross section orientation, can be only one or two lines.

        """
        csl_points = self.get_csl_point(h=h, l=l)
        z = csl_points[0].z

        # acquire angle perpendicular to cross section
        angle_perp = self.cs_angle + np.pi / 2

        # retrieve lines perpendicular to cross section with offset and length
        csl_lines = _make_angle_lines(csl_points, angle_perp, length, offset)

        if camera:
            # transform to 2D projected lines, make list of lists of coordinates
            coords_lines = [[[_x, _y, z] for _x, _y in l.coords] for l in csl_lines]
            # project list of lists
            coords_lines_proj = [
                self.camera_config.project_points(cl, swap_y_coords=swap_y_coords) for cl in coords_lines
            ]
            # turn list of lists into list of LineString
            csl_lines = [geometry.LineString([geometry.Point(_x, _y) for _x, _y in l]) for l in coords_lines_proj]
        else:
            # add a z-coordinate and return
            csl_lines = [geometry.LineString([geometry.Point(_x, _y, z) for _x, _y in l.coords]) for l in csl_lines]
        return csl_lines

    def get_csl_pol(
        self,
        h: Optional[float] = None,
        l: Optional[float] = None,
        length: float = 0.5,
        padding: Tuple[float, float] = (-0.5, 0.5),
        offset: float = 0.0,
        camera: bool = False,
        swap_y_coords: bool = False,
    ) -> List[geometry.Polygon]:
        """Get horizontal polygon from cross-section land-line towards water or towards land.

        Returns a 2D Polygon if camera is True, 3D if False

        Parameters
        ----------
        h : float
            water level [m]
        l : float
            coordinate of distance from left to right bank, including height [m], if provided, h must not be provided.
        length : float, optional
            length of the waterline [m], by default 0.5
        padding : Tuple[float, float], optional
            amount of distance [m] to extend the polygon beyond the waterline, by default (-0.5, 0.5)
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygons, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        List[shapely.geometry.Polygon]
            List of lines perpendicular to cross section orientation, can be only one or two lines.


        """
        # retrieve water line(s)
        csl = self.get_csl_line(h=h, l=l, length=length, offset=offset)
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
            coords_expand = np.zeros((0, 3))
            for cn, coords in enumerate(csl_pol_coords):
                for n in range(0, len(coords) - 1):
                    new_coords = np.linspace(coords[n], coords[n + 1], 100)
                    coords_expand = np.r_[coords_expand, new_coords]
                coords = coords_expand
                csl_pol_coords[cn] = coords
            csl_pol_coords = [
                self.camera_config.project_points(coords, swap_y_coords=swap_y_coords, within_image=True)
                for coords in csl_pol_coords
            ]
            csl_pol_coords = [coords[np.isfinite(coords[:, 0])] for coords in csl_pol_coords]
        polygons = [geometry.Polygon(coords) for coords in csl_pol_coords]
        return polygons

    def get_bbox_dry_or_wet(self, h: float, camera: bool = False, swap_y_coords: bool = False, dry: bool = False):
        """Get dry part of bounding box, by extending the water line perpendicular to the cross section.

        Parameters
        ----------
        h : float
            water level [m]
        camera : bool, optional
            If set, return 2D projected polygon, by default False. Note that any plotting does not provide shading or
            ray tracing hence plotting a 2D polygon is not recommended.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        dry : bool, optional
            If set, the dry parts will be returned, by default False, thus returning wet.

        Returns
        -------
        shapely.geometry.MultiPolygon
            multipolygon representing the dry or wet part of the bounding box of the camera config (2d if camera=True,
            3d if camera=False).

        See Also
        --------
        CrossSection.plot_bbox_dry_wet : Plot the dry and wet parts of the bounding box resulting from this method.



        """
        if self.camera_config.bbox is None:
            raise ValueError("CameraConfig must have a bounding box to use this method.")
        z_water = self.camera_config.h_to_z(h)  # 92.1 is exactly hitting a local peak
        geom_plan_2d = force_2d(self.get_planar_surface(h=h, length=10000))

        if dry:
            # dry pols
            pols = force_3d(self.camera_config.bbox.difference(geom_plan_2d), z=z_water)
        else:
            # wet pol(s)
            pols = force_3d(self.camera_config.bbox.intersection(geom_plan_2d), z=z_water)
        if isinstance(pols, geometry.MultiPolygon):
            pols = pols.geoms
        else:
            pols = [pols]
        pols_proj = []
        for pol in pols:
            # coords = [[self.interp_x_from_s(p[0]), self.interp_y_from_s(p[0]), p[1]] for p in pol.exterior.coords]
            coords = [[*p] for p in pol.exterior.coords]
            if camera:
                coords_proj = self.camera_config.project_points(coords, swap_y_coords=swap_y_coords, within_image=True)
                pols_proj.append(geometry.Polygon(coords_proj))
            else:
                pols_proj.append(geometry.Polygon(coords))
        return geometry.MultiPolygon(pols_proj)

    def get_bottom_surface(
        self, length: float = 2.0, offset: float = 0.0, camera: bool = False, swap_y_coords: bool = False
    ):
        """Retrieve a bottom surface polygon for the entire cross section, expanded over a length.

        Returns a 2D Polygon if camera is True, 3D if False. Useful in particular for 3d situation plots.

        Parameters
        ----------
        length : float, optional
            length of the cross-section lateral expansion (from up to downstream) [m], by default 2.0.
        offset : float, optional
            perpendicular offset of the expanded cross section from the original cross section coordinates [m],
            by default 0.0.
        camera : bool, optional
            If set, return 2D projected polygon, by default False. Note that any plotting does not provide shading or
            ray tracing hence plotting a 2D polygon is not recommended.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        shapely.geometry.Polygon
            polygon representing the cross-section expanded over a certain length (2d if camera=True,
            3d if camera=False).

        See Also
        --------
        CrossSection.plot_bottom_surface : Plot the bottom surface resulting from this method.

        """
        csl_points = [self.cs_points[0], self.cs_points[-1]]
        angle_perp = self.cs_angle + np.pi / 2
        # retrieve lines perpendicular to cross section with offset and length
        csl_lines = _make_angle_lines(csl_points, self.cs_angle + np.pi / 2, length, offset)
        csl_line_points = [
            [geometry.Point(_x, _y, z) for _x, _y in l.coords]
            for l, z in zip(csl_lines, [self.cs_points[0].z, self.cs_points[-1].z])
        ]
        # retrieve lines of displaced cross sections
        csl_displaced = [
            [
                affinity.translate(p, xoff=np.cos(angle_perp) * (offset + l), yoff=np.sin(angle_perp) * (offset + l))
                for p in self.cs_points
            ]
            for l in [length / 2, -length / 2]
        ]
        # put together a full polygon
        all_points = csl_line_points[0] + csl_displaced[0] + csl_line_points[1][::-1] + csl_displaced[1][::-1]
        if camera:
            coords = np.array([list(p.coords[0]) for p in all_points])

            all_points = self.camera_config.project_points(coords, swap_y_coords=swap_y_coords, within_image=True)
            all_points = all_points[np.isfinite(all_points[:, 0])]
            all_points = [geometry.Point(*p) for p in all_points]

        return geometry.Polygon(all_points)

    def get_planar_surface(
        self, h: float, length: float = 2.0, offset: float = 0.0, camera: bool = False, swap_y_coords=False
    ) -> Union[geometry.Polygon, geometry.MultiPolygon]:
        """Retrieve a planar water surface for a given water level, as a geometry.Polygon.

        Returns a 2D Polygon if camera is True, 3D if False

        Parameters
        ----------
        h : float
            water level [m]
        length : float, optional
            length of the waterline [m], by default 2.0
        offset : float, optional
            perpendicular offset of the waterline from the cross section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        shapely.geometry.Polygon
            rectangular horizontal polygon representing the planar water surface (2d if camera=True,
            3d if camera=False).

        See Also
        --------
        CrossSection.plot_planar_surface : Plot the planar surface resulting from this method.

        """
        # Get the water level crossing points (not camera projected for validation)
        csl_points = self.get_csl_point(h=h, camera=False)
        if len(csl_points) < 2:
            raise ValueError(
                f"Cross section must have at least two points to estimate a planar surface ({len(csl_points)} found)."
            )
        wls = self.get_csl_line(h=h, offset=offset, length=length, camera=camera, swap_y_coords=swap_y_coords)

        valid_pairs = []
        for p1, p2, l1, l2 in zip(csl_points[0:-1], csl_points[1:], wls[0:-1], wls[1:]):
            # get s-coordinates
            s1 = self.cs_linestring.project(geometry.Point(p1.x, p1.y))
            s2 = self.cs_linestring.project(geometry.Point(p2.x, p2.y))

            # check bottom elevation at midpoint
            s_mid = (s1 + s2) / 2
            z_bottom_mid = self.interp_z_from_s(s_mid)
            if z_bottom_mid < p1.z:
                valid_pairs.append((l1, l2))
        if len(valid_pairs) == 0:
            raise ValueError(
                "No valid water level crossings found. Check if water level is within the cross section and if "
                "the cross section is crossed by water."
            )

        # build the polygons
        polygons = []
        for l1, l2 in valid_pairs:
            pol = geometry.Polygon(list(l1.coords) + list(l2.coords[::-1]))
            if pol.is_valid and not pol.is_empty:
                polygons.append(pol)

        if len(polygons) == 0:
            raise ValueError(
                "No valid polygons found. Check if water level is within the cross section and if "
                "the cross section is crossed by water."
            )
        elif len(polygons) == 1:
            return polygons[0]
        else:
            return geometry.MultiPolygon(polygons)
        # if len(wls) != 2:
        #     raise ValueError("Amount of water line crossings must be 2 for a planar surface estimate.")
        # return geometry.Polygon(list(wls[0].coords) + list(wls[1].coords[::-1]))

    def get_wetted_surface_sz(
        self, h: float, perimeter: bool = False
    ) -> Union[geometry.MultiPolygon, geometry.MultiLineString]:
        """Retrieve a wetted surface or perimeter perpendicular to flow direction (SZ) for a water level.

        This returns a `geometry.MultiPolygon` when a surface is requested (`perimeter=False`), and
        `geometry.MultiLineString` when a perimeter is requested (`perimeter=True`).

        This is a useful method for instance to estimate m2 wetted surface or m wetted perimeter length for a given
        water level in the cross section.

        Parameters
        ----------
        h : float
            water level [m]
        perimeter : bool, optional
            If set to True, return a linestring with the wetted perimeter instead.

        Returns
        -------
        geometry.Polygon
            Wetted surface as a polygon, in Y-Z projection.

        """

        def avg_y(line):
            """Compute average y-coordinate of a line."""
            ys = [p[1] for p in line.coords]
            return sum(ys) / len(ys)

        wl = self.get_cs_waterlevel(
            h=h, sz=True, extend_by=0.1
        )  # extend a small bit to guarantee crossing with the bottom coordinates
        zl = wl.xy[1][0]
        # create polygon by making a union
        bottom_points = self.cs_points_sz
        # add a point left and right slightly above the level if the level is below the water level
        if bottom_points[0].y < zl:
            bottom_points.insert(0, geometry.Point(bottom_points[0].x, zl + 0.1))
        if bottom_points[-1].y < zl:
            bottom_points.append(geometry.Point(bottom_points[-1].x, zl + 0.1))
        bottom_line = geometry.LineString(bottom_points)
        if perimeter:
            wl_z = wl.coords[0][-1]
            split_segments = split(bottom_line, wl)
            filtered = []
            for seg in split_segments.geoms:
                seg_z = avg_y(seg)
                if seg_z < wl_z:
                    # segment is below water level, add to perimeter
                    filtered.append(seg)

            return geometry.MultiLineString(filtered)
            # return wetted_perim
        pol = list(polygonize(wl.union(bottom_line)))
        if len(pol) == 0:
            # create infinitely small polygon at lowest z coordinate
            lowest_z = min(self.z)
            lowest_s = self.s[list(self.z).index(lowest_z)]
            # make an infinitely small polygon around the lowest point in the cross section
            pol = [geometry.Polygon([(lowest_s, lowest_z)] * 3)]
        elif len(pol) > 1:
            # detect which polygons have their average z coordinate below the defined water level
            pol = [p for p in pol if p.centroid.xy[1][0] < zl]
            # raise ValueError("Water level is crossed by multiple polygons.")
        # else:
        #     pol = pol[0]
        return geometry.MultiPolygon(pol)

    def get_wetted_surface(self, h: float, camera: bool = False, swap_y_coords=False) -> geometry.Polygon:
        """Retrieve a wetted surface for a given water level, as a geometry.Polygon.

        Parameters
        ----------
        h : float
            water level [m]
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)

        Returns
        -------
        geometry.Polygon
            Wetted surface as a polygon (2d if camera=True, 3d if camera=False).


        """
        pols = self.get_wetted_surface_sz(h=h)
        pols_proj = []
        for pol in pols.geoms:
            coords = [[self.interp_x_from_s(p[0]), self.interp_y_from_s(p[0]), p[1]] for p in pol.exterior.coords]
            if camera:
                coords_proj = self.camera_config.project_points(coords, swap_y_coords=swap_y_coords)
                pols_proj.append(geometry.Polygon(coords_proj))
            else:
                pols_proj.append(geometry.Polygon(coords))
        return geometry.MultiPolygon(pols_proj)

    def get_line_of_interest(self, bank: BANK_OPTIONS = "far") -> List[float]:
        """Retrieve the points of interest within the cross-section for water level detection.

        This may be all points, points only at the far-bank or closest-bank of the camera position.

        Parameters
        ----------
        bank: Literal["far", "near", "both"], optional
            Select relevant part of cross section. If "both", select the full cross-section. If "far", select
            only the furthest part from the deepest point in the cross section, if any. If "near", select only
            the "nearest".

        Returns
        -------
        list of 2 floats
            start and end point of the line of interest in l-coordinates.

        """
        if bank == "both":
            return self.l.min(), self.l.max()
        elif bank == "far":
            start_point = self.l[self.idx_farthest_point]

        elif bank == "near":
            start_point = self.l[self.idx_closest_point]
        else:
            raise ValueError(f"bank must be one of {BANK_OPTIONS}, not {bank}")
        # find l values at lowest point

        l_lowest = self.l[np.where(self.z == self.z.min())]
        # find which is closest to start_point
        end_point = l_lowest[np.argmin(np.abs(l_lowest - start_point))]

        # sort from low to high
        return tuple(np.sort(np.array([start_point, end_point])))
        # fin l-value closest to

    def get_histogram_score(
        self,
        x: float,
        img: np.array,
        bin_size: float = 5.0,
        offset: float = 0.0,
        padding: float = 0.5,
        length: float = 2.0,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
        min_samples: int = 50,
    ):
        """Retrieve a histogram score for a given l-value."""
        l = x[0]
        if min_z is not None:
            if self.interp_z(l) < min_z:
                # return worst score
                return 2.0 + np.abs(self.interp_z(l) - min_z)
        elif max_z is not None:
            if self.interp_z(l) > max_z:
                return 2.0 + np.abs(self.interp_z(l) - max_z)
        pol1 = self.get_csl_pol(l=l, offset=offset, padding=(0, padding), length=length, camera=True)[0]
        pol2 = self.get_csl_pol(l=l, offset=offset, padding=(-padding, 0), length=length, camera=True)[0]
        # get intensity values within polygons
        ints1 = cv.get_polygon_pixels(img, pol1)
        ints2 = cv.get_polygon_pixels(img, pol2)
        if ints1.size < min_samples or ints2.size < min_samples:
            # return a strong penalty score value if there are too few samples
            return 2.0
        _, _, norm_counts1 = _histogram(ints1, normalize=True, bin_size=bin_size)
        bin_centers, bin_edges, norm_counts2 = _histogram(ints2, normalize=True, bin_size=bin_size)
        return _histogram_union(bin_edges, norm_counts1, norm_counts2)

    def plot(
        self,
        h: Optional[float] = None,
        length: float = 2.0,
        offset: float = 0.0,
        camera: bool = False,
        ax: Optional[mpl.axes.Axes] = None,
        cs_kwargs: Dict = None,
        planar=True,
        bottom=True,
        wetted=True,
        swap_y_coords=False,
        planar_kwargs: Dict = PLANAR_KWARGS,
        bottom_kwargs: Dict = BOTTOM_KWARGS,
        wetted_kwargs: Dict = WETTED_KWARGS,
    ) -> mpl.axes.Axes:
        """Plot the cross-section situation.

        This will create a plot of the cross section, with some up to downstream length in a 3D axes. It can
        be combined with a plot of the camera configuration, with `cross_section.camera_config.plot(mode="3d")`.

        Parameters
        ----------
        mode : Literal["camera", "3d"]
            manner in which plotting should be done.
        h : float, optional
            water level [m]. If not provided, the water level is taken from the camera config
            `cross_section.camera_config.gcps["h_ref"]`.
        length : float, optional
            length of the waterline [m], by default 2.0
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        cs_kwargs : dict, optional
            keyword arguments used to make the line plot of the cross section.
        planar : bool, optional
            whether to plot the planar, by default True.
        bottom : bool, optional
            whether to plot the bottom surface, by default True.
        wetted : bool, optional
            whether to plot the wetted surface, by default True.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        planar_kwargs : dict, optional
            keyword arguments used to make the polygon plot of the water plane. If not provided, a set of defaults
            will be used that give a natural look.
        bottom_kwargs : dict, optional
            keyword arguments used to make the polygon plot of the bottom surface. If not provided, a set of defaults
            will be used that give a natural look.
        wetted_kwargs : dict, optional
            keyword arguments used to make the polygon plot of the wetted surface. If not provided, a set of defaults
            will be used that give a natural look.

        Returns
        -------
        mpl.axes.Axes
            The developed axes object with all data

        """
        if not cs_kwargs:
            cs_kwargs = {}
        if not ax:
            if not camera:
                ax = plt.axes(projection="3d")
            else:
                ax = plt.axes()
        if h is None:
            h = self.camera_config.gcps["h_ref"]
            warnings.warn(
                "No water level is provided. Camera configuration reference water level is used.", stacklevel=2
            )
        if planar:
            self.plot_planar_surface(
                h=h, length=length, offset=offset, camera=camera, ax=ax, swap_y_coords=swap_y_coords, **planar_kwargs
            )
        if bottom:
            self.plot_bottom_surface(
                length=length, offset=offset, camera=camera, ax=ax, swap_y_coords=swap_y_coords, **bottom_kwargs
            )
        if wetted:
            self.plot_wetted_surface(h=h, camera=camera, ax=ax, swap_y_coords=swap_y_coords, **wetted_kwargs)
        self.plot_cs(ax=ax, camera=camera, swap_y_coords=swap_y_coords, **cs_kwargs)
        # ax.set_aspect("equal")
        return ax

    def plot_cs(self, ax=None, camera=False, swap_y_coords: bool = False, **kwargs):
        """Plot the cross section.

        Parameters
        ----------
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        **kwargs : dict, optional
            keyword arguments used to make the line plot of the cross-section.

        Returns
        -------
        mpl mappable

        """
        if not ax:
            if camera:
                ax = plt.axes()
            else:
                ax = plt.axes(projection="3d")
        if ax.name == "3d" and not camera:
            # map 3d coordinates to x, y, z
            x, y, z = zip(*[(c[0], c[1], c[2]) for c in self.cs_linestring.coords])
            p = ax.plot(x, y, z, **kwargs)
        else:
            if camera:
                # extract pixel locations
                pix = self.camera_config.project_points(
                    list(map(list, self.cs_linestring.coords)), within_image=True, swap_y_coords=swap_y_coords
                )
                # map to x and y arrays
                x, y = zip(*[(c[0], c[1]) for c in pix if np.isfinite(c[0])])
            else:
                x, y = zip(*[(c[0], c[1]) for c in self.cs_linestring_sz.coords])
            p = ax.plot(x, y, **kwargs)
        return p

    def plot_planar_surface(
        self,
        h: float,
        length: float = 2.0,
        offset: float = 0.0,
        camera: bool = False,
        swap_y_coords=False,
        ax=None,
        **kwargs,
    ) -> mpl.axes.Axes:
        """Plot the planar surface for a given water level.

        Parameters
        ----------
        h : float, optional
            water level [m]. If not provided, the water level is taken from the camera config
            `cross_section.camera_config.gcps["h_ref"]`.
        length : float, optional
            length of the waterline [m], by default 2.0
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        **kwargs : dict, optional
            keyword arguments used to make the polygon plot of the planar surface.

        Returns
        -------
        plt.axes

        """
        try:
            surf = self.get_planar_surface(
                h=h, length=length, offset=offset, swap_y_coords=swap_y_coords, camera=camera
            )
            if camera:
                p = plot_helpers.plot_polygon(surf, ax=ax, label="surface", **kwargs)
            else:
                p = plot_helpers.plot_3d_polygon(surf, ax=ax, label="surface", **kwargs)
            return p.axes
        except Exception:
            warnings.warn(
                "Cannot plot planar surface as there are too many crossings",
                stacklevel=2,
            )

    def plot_bbox_dry_wet(
        self,
        h: float,
        camera: bool = False,
        ax=None,
        swap_y_coords: bool = False,
        kwargs_wet: Optional[Dict] = None,
        kwargs_dry: Optional[Dict] = None,
    ):
        """Plot the bounding box of the dry and wet surfaces for a given water level."""
        if kwargs_wet is None:
            kwargs_wet = dict(color="b", label="wet")
        if kwargs_dry is None:
            kwargs_dry = dict(color="g", label="dry")
        if ax is None:
            if camera:
                ax = plt.axes()
            else:
                ax = plt.axes(projection="3d")
        dry_pols = self.get_bbox_dry_or_wet(h=h, dry=True, camera=camera, swap_y_coords=swap_y_coords)
        wet_pols = self.get_bbox_dry_or_wet(h=h, dry=False, camera=camera, swap_y_coords=swap_y_coords)
        if camera:
            _ = plot_helpers.plot_polygon(wet_pols, ax=ax, **kwargs_wet)
            p = plot_helpers.plot_polygon(dry_pols, ax=ax, **kwargs_dry)
        else:
            _ = plot_helpers.plot_3d_polygon(wet_pols, ax=ax, **kwargs_wet)
            p = plot_helpers.plot_3d_polygon(dry_pols, ax=ax, **kwargs_dry)
        return p.axes

    def plot_bottom_surface(
        self, length: float = 2.0, offset: float = 0.0, camera: bool = False, ax=None, swap_y_coords=False, **kwargs
    ) -> mpl.axes.Axes:
        """Plot the bottom surface for a given water level.

        Parameters
        ----------
        length : float, optional
            length of the waterline [m], by default 2.0
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        **kwargs : dict, optional
            keyword arguments used to make the polygon plot of the bottom surface.

        Returns
        -------
        plt.axes

        """
        surf = self.get_bottom_surface(length=length, offset=offset, camera=camera, swap_y_coords=swap_y_coords)
        if camera:
            p = plot_helpers.plot_polygon(surf, ax=ax, label="bottom", **kwargs)
        else:
            p = plot_helpers.plot_3d_polygon(surf, ax=ax, label="bottom", **kwargs)
        return p.axes

    def plot_wetted_surface(self, h: float, camera: bool = False, swap_y_coords=False, ax=None, **kwargs):
        """Plot the wetted surface for a given water level.

        Parameters
        ----------
        h : float, optional
            water level [m]. If not provided, the water level is taken from the camera config
            `cross_section.camera_config.gcps["h_ref"]`.
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        **kwargs : dict, optional
            keyword arguments used to make the polygon plot of the wetted surface.

        Returns
        -------
        plt.axes

        """
        surf = self.get_wetted_surface(h=h, camera=camera, swap_y_coords=swap_y_coords)
        if camera:
            p = plot_helpers.plot_polygon(surf, ax=ax, label="wetted", **kwargs)
        else:
            p = plot_helpers.plot_3d_polygon(surf, ax=ax, label="wetted", **kwargs)
        return p.axes

    def plot_water_level(
        self,
        h: float,
        length: float = 0.5,
        offset: float = 0.0,
        camera: bool = False,
        swap_y_coords: bool = False,
        add_text: bool = True,
        ax: Optional[mpl.axes.Axes] = None,
        **kwargs,
    ) -> mpl.axes.Axes:
        """Plot the water level at user provided value `h`.

        Parameters
        ----------
        h : float, optional
            water level [m]. If not provided, the water level is taken from the camera config
            `cross_section.camera_config.gcps["h_ref"]`.
        length : float, optional
            length of the waterline [m], by default 2.0
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        camera : bool, optional
            If set, return 2D projected polygon, by default False.
        swap_y_coords : bool, optional
            if set, all y-coordinates are swapped so that they fit on a flipped version of the image. This is useful
            in case you plot on ascending y-coordinate axis background images (default: False)
        add_text : bool, optional
            Default True, determines whether to add a text label to the water level plot at the farthest line.
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None). If provided, user must take care to provide the correct
            projection. If `camera=False`, an axes must be provided with `projection="3d"`.
        **kwargs : dict, optional
            keyword arguments used to make the line plot of the water level, perpendicular to the cross-section.

        Returns
        -------
        plt.axes

        """
        if ax is None:
            if camera:
                ax = plt.axes()
            else:
                ax = plt.axes(projection="3d")
        lines = self.get_csl_line(h=h, length=length, offset=offset, camera=camera, swap_y_coords=swap_y_coords)
        if camera:
            plot_f = plot_helpers.plot_line
        else:
            plot_f = plot_helpers.plot_3d_line
        for l in lines:
            _ = plot_f(l, ax=ax, zorder=1, **kwargs)
        if add_text and camera:
            points = self.get_csl_point(h=h, camera=False)  # find real-world points
            lens_position_xy = self.camera_config.estimate_lens_position()[0:2]
            dists = [((p.x - lens_position_xy[0]) ** 2 + (p.y - lens_position_xy[1]) ** 2) ** 0.5 for p in points]
            points = self.get_csl_point(h=h, camera=True, swap_y_coords=swap_y_coords)  # find camera positions
            x, y = points[np.argmax(dists)].xy
            x, y = float(x[0]), float(y[0])

            # only plot text in 2D camera perspective at farthest point
            ax.text(
                x, y, "{:1.3f} m.".format(h), path_effects=PATH_EFFECTS, ha="center", va="bottom", size=12, zorder=2
            )
        return ax

    def rotate_translate(self, angle: Optional[float] = None, xoff: float = 0.0, yoff: float = 0.0, zoff: float = 0.0):
        """Rotate and translate cross section to match config.

        Parameters
        ----------
        angle : float, optional
            Rotation angle in radians (anti-clockwise) around the center of the bounding box
        xoff : float, optional
            Translation distance in x direction in m.
        yoff : float, optional
            Translation distance in y direction in m.
        zoff : float, optional
            Translation distance in z direction in m.

        """
        # Apply rotation if specified
        if angle is not None:
            # Get centroid as origin
            centroid = self.cs_linestring.centroid
            # Apply rotation around centroid
            new_line = rotate(
                self.cs_linestring,
                angle,
                origin=centroid,
                use_radians=True,
            )
        else:
            new_line = copy.deepcopy(self.cs_linestring)
        # now translate
        new_line = translate(new_line, xoff=xoff, yoff=yoff, zoff=zoff)
        # create a new cross section object
        return CrossSection(self.camera_config, list(new_line.coords))

    def _preprocess_level_range(
        self,
        min_h: Optional[float] = None,
        max_h: Optional[float] = None,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ):
        """Set minimum and maximum z-levels (if not provided) based on the camera config."""
        if min_z is None:
            if min_h is not None:
                min_z = self.camera_config.h_to_z(min_h)
                min_z = np.maximum(min_z, self.z.min())
        if max_z is None:
            if max_h is not None:
                max_z = self.camera_config.h_to_z(max_h)
                max_z = np.minimum(max_z, self.z.max())
        if min_z and max_z:
            # check for z_min < z_max
            if min_z > max_z:
                raise ValueError("Minimum water level is higher than maximum water level.")
        return min_z, max_z

    def _preprocess_l_range(self, l_min: float, l_max: float, ds_max=0.5, dz_max=0.02) -> List[float]:
        """Generate a list of evaluation points between l_min and l_max for water level detection.

        Controls on evaluation are that vertically (using `self.z`), points are at least `dz_max` meters apart and
        horizontally (using `self.s`), points are included if the distance between them exceeds `dz_min` meters.

        Parameters
        ----------
        l_min: float
            Minimum l value for evaluation.
        l_max: float
            Maximum l value for evaluation.
        ds_max : float, optional
            maximum step size between evaluation points in horizontal direction, by default 0.5.
        dz_max : float, optional
            maximum step size between evaluation points in vertical direction, by default 0.02.

        Returns
        -------
        Tuple[List[float], List[float]]
            lists of l-coordinates and associated z-coordinates for evaluation.

        """
        # # Initial list of evaluation points
        # l_range = []
        # z_range = []
        # Start with the first point within the range
        current_l = l_min
        last_z = None
        last_s = None

        # Add all points from self.l that lie within [l_min, l_max]
        valid_l_indices = (self.l >= l_min) & (self.l <= l_max)  # Logical filter
        l_range = list(self.l[valid_l_indices])
        z_range = list(self.z[valid_l_indices])

        # Iterate over the self.l values by interpolating within the range
        while current_l <= l_max:
            # Interpolate x, y, z, and s using the available properties
            # x = self.interp_x(current_l)
            # y = self.interp_y(current_l)
            z = self.interp_z(current_l)
            s = self.interp_s_from_l(current_l)

            # Append the point if it satisfies the required criteria
            if last_z is None or last_s is None or abs(z - last_z) >= dz_max or abs(s - last_s) >= ds_max:
                l_range.append(current_l)
                z_range.append(z)
                last_z = z
                last_s = s

            # Increment l
            current_l += 0.01  # Increment in steps small enough to meet any constraint

        # add the final l
        if current_l > l_max:
            l_range.append(l_max)
            z_range.append(self.interp_z(l_max))

        # sort l_range and z_range by incremental l_range
        sorted_indices = np.argsort(l_range)
        l_range = np.array(l_range)[sorted_indices]
        z_range = np.array(z_range)[sorted_indices]

        return l_range, z_range

    def _water_level_score_range(
        self,
        img: np.ndarray,
        bank: BANK_OPTIONS = "far",
        bin_size: int = 5,
        length: float = 2.0,
        padding: float = 0.5,
        offset: float = 0.0,
        ds_max: Optional[float] = 0.5,
        dz_max: Optional[float] = 0.02,
        min_h: Optional[float] = None,
        max_h: Optional[float] = None,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> float:
        """Evaluate a range of scores for water level detection.

        This computes the histogram score for a range of l values and returns all scores for further evaluation.
        The histogram score is a measure of dissimilarity (low means very dissimilar) of pixel intensity distributions
        left and right of hypothesized water lines. The evaluation points along the cross section (l-values) are
        determined by two parameters max_ds (maximum step in horizontal direction) and max_dz (maximum step in vertical
        direction).

        Parameters
        ----------
        img : np.ndarray
            image (uint8) used to estimate water level from.
        bank: Literal["far", "near", "both"], optional
            select from which bank to detect the water level. Use this if camera is positioned in a way that only
            one shore is clearly distinguishable and not obscured. Typically you will use "far" if the camera is
            positioned on one bank, aimed perpendicular to the flow. Use "near" if not the full cross section is
            visible, but only the part nearest the camera. And leave empty when both banks are clearly visible and
            approximately the same in distance (e.g. middle of a bridge). If not provided, the bank is detected based
            on the best estimate from both banks.
        bin_size : int, optional
            Size of bins for histogram calculation of the provided image intensities, default 5.
        length : float, optional
            length of the waterline [m], by default 2.0
        padding : float, optional
            amount of distance [m] to extend the polygon beyond the waterline, by default 0.5. Two polygons are drawn
            left and right of hypothesized water line at -padding and +padding.
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        ds_max : float, optional
            maximum step size between evaluation points in horizontal direction, by default 0.5.
        dz_max : float, optional
            maximum step size between evaluation points in vertical direction, by default 0.02.
        min_h : float, optional
            minimum water level to try detection [m]. If not provided, the minimum water level is taken from the
            cross section.
        max_h : float, optional
            maximum water level to try detection [m]. If not provided, the maximum water level is taken from the
            cross section.
        min_z : float, optional
            same as min_h but using z-coordinates instead of local datum, min_z overrules min_h
        max_z : float, optional
            same as max_z but using z-coordinates instead of local datum, max_z overrules max_h

        """
        l_min, l_max = self.get_line_of_interest(bank=bank)
        min_z, max_z = self._preprocess_level_range(min_h, max_h, min_z, max_z)
        l_range, z_range = self._preprocess_l_range(l_min=l_min, l_max=l_max, ds_max=ds_max, dz_max=dz_max)
        if len(img.shape) == 3:
            # flatten image first if it his a time dimension
            img = img.mean(axis=2)
        assert (
            img.shape[0] == self.camera_config.height
        ), f"Image height {img.shape[0]} is not the same as camera_config height {self.camera_config.height}"
        assert (
            img.shape[1] == self.camera_config.width
        ), f"Image width {img.shape[1]} is not the same as camera_config width {self.camera_config.width}"

        # gridded results
        results = [
            self.get_histogram_score(
                x=[l],
                img=img,
                bin_size=bin_size,
                offset=offset,
                padding=padding,
                length=length,
                min_z=min_z,
                max_z=max_z,
            )
            for l in l_range
        ]
        return l_range, z_range, results

    def detect_water_level(
        self,
        img: np.ndarray,
        bank: BANK_OPTIONS = "far",
        bin_size: int = 5,
        length: float = 2.0,
        padding: float = 0.5,
        offset: float = 0.0,
        min_h: Optional[float] = None,
        max_h: Optional[float] = None,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> float:
        """Detect water level from provided image through optimization.

        Water level detection is done by first detecting the water line along the cross-section by comparisons
        of distribution functions left and right of hypothesized water lines, and then looking up the water level
        associated with the water line location. A differential evolution optimization is used to find the optimum.

        Parameters
        ----------
        img : np.ndarray
            image (uint8) used to estimate water level from.
        bank: Literal["far", "near", "both"], optional
            select from which bank to detect the water level. Use this if camera is positioned in a way that only
            one shore is clearly distinguishable and not obscured. Typically you will use "far" if the camera is
            positioned on one bank, aimed perpendicular to the flow. Use "near" if not the full cross section is
            visible, but only the part nearest the camera. And leave empty when both banks are clearly visible and
            approximately the same in distance (e.g. middle of a bridge). If not provided, the bank is detected based
            on the best estimate from both banks.
        bin_size : int, optional
            Size of bins for histogram calculation of the provided image intensities, default 5.
        length : float, optional
            length of the waterline [m], by default 2.0
        padding : float, optional
            amount of distance [m] to extend the polygon beyond the waterline, by default 0.5. Two polygons are drawn
            left and right of hypothesized water line at -padding and +padding.
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0
        min_h : float, optional
            minimum water level to try detection [m]. If not provided, the minimum water level is taken from the
            cross section.
        max_h : float, optional
            maximum water level to try detection [m]. If not provided, the maximum water level is taken from the
            cross section.
        min_z : float, optional
            same as min_h but using z-coordinates instead of local datum, min_z overrules min_h
        max_z : float, optional
            same as max_z but using z-coordinates instead of local datum, max_z overrules max_h

        Returns
        -------
        float
            Most likely water level, according to optimization and scoring (most distinct intensity PDF)

        """
        l_min, l_max = self.get_line_of_interest(bank=bank)
        min_z, max_z = self._preprocess_level_range(min_h, max_h, min_z, max_z)
        if len(img.shape) == 3:
            # flatten image first if it his a time dimension
            img = img.mean(axis=2)
        assert (
            img.shape[0] == self.camera_config.height
        ), f"Image height {img.shape[0]} is not the same as camera_config height {self.camera_config.height}"
        assert (
            img.shape[1] == self.camera_config.width
        ), f"Image width {img.shape[1]} is not the same as camera_config width {self.camera_config.width}"

        opt = differential_evolution(
            self.get_histogram_score,
            popsize=50,
            bounds=[(l_min, l_max)],
            args=(img, bin_size, offset, padding, length, min_z, max_z),
            atol=0.01,  # one mm
        )
        z = self.interp_z(opt.x[0])
        h = self.camera_config.z_to_h(z)
        # warning if the optimum is on the edge of the search space for l
        if np.isclose(opt.x[0], l_min) or np.isclose(opt.x[0], l_max):
            warnings.warn(
                f"The detected water level is on the edge of the search space and may be wrong. "
                f"Water level is {h} m. at cross-section length {opt.x[0]}.",
                UserWarning,
                stacklevel=2,
            )
        return h

    def detect_water_level_s2n(
        self,
        img: np.ndarray,
        bank: BANK_OPTIONS = "far",
        bin_size: int = 5,
        length: float = 2.0,
        padding: float = 0.5,
        offset: float = 0.0,
        ds_max: Optional[float] = 0.5,
        dz_max: Optional[float] = 0.02,
        min_h: Optional[float] = None,
        max_h: Optional[float] = None,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> float:
        """Detect water level optically from provided image, through evaluation of a vector of locations.

        Water level detection is done by first detecting the water line along the cross-section by comparisons
        of distribution functions left and right of hypothesized water lines, and then looking up the water level
        associated with the water line location. Because a full vector of results is evaluated, the signal to noise
        ratio can be evaluated and used to understand the quality of the water level detection. Two parameters are used
        to control the spacing between l (location) coordinates: `ds_max` controls the maximum step size in horizontal
        direction, `dz_max` controls the maximum step size in vertical direction.

        Parameters
        ----------
        img : np.ndarray
            image (uint8) used to estimate water level from.
        bank: Literal["far", "near", "both"], optional
            select from which bank to detect the water level. Use this if camera is positioned in a way that only
            one shore is clearly distinguishable and not obscured. Typically you will use "far" if the camera is
            positioned on one bank, aimed perpendicular to the flow. Use "near" if not the full cross section is
            visible, but only the part nearest the camera. And leave empty when both banks are clearly visible and
            approximately the same in distance (e.g. middle of a bridge). If not provided, the bank is detected based
            on the best estimate from both banks.
        bin_size : int, optional
            Size of bins for histogram calculation of the provided image intensities, default 5.
        length : float, optional
            length of the waterline [m], by default 2.0.
        padding : float, optional
            amount of distance [m] to extend the polygon beyond the waterline, by default 0.5. Two polygons are drawn
            left and right of hypothesized water line at -padding and +padding.
        offset : float, optional
            perpendicular offset of the waterline from the cross-section [m], by default 0.0.
        ds_max : float, optional
            maximum step size between evaluation points in horizontal direction, by default 0.5.
        dz_max : float, optional
            maximum step size between evaluation points in vertical direction, by default 0.02.
        min_h : float, optional
            minimum water level to try detection [m]. If not provided, the minimum water level is taken from the
            cross section.
        max_h : float, optional
            maximum water level to try detection [m]. If not provided, the maximum water level is taken from the
            cross section.
        min_z : float, optional
            same as min_h but using z-coordinates instead of local datum, min_z overrules min_h.
        max_z : float, optional
            same as max_z but using z-coordinates instead of local datum, max_z overrules max_h.

        Returns
        -------
        float
            Most likely water level, according to optimization and scoring (most distinct intensity PDF).
        float
            Signal to noise ratio, calculated as the mean of all computed scores divided by the optimum (lowest) score
            found in the entire l-range.

        """
        l_range, z_range, results = self._water_level_score_range(
            img=img,
            bank=bank,
            bin_size=bin_size,
            length=length,
            padding=padding,
            offset=offset,
            ds_max=ds_max,
            dz_max=dz_max,
            min_h=min_h,
            max_h=max_h,
            min_z=min_z,
            max_z=max_z,
        )
        # find optimum (minimum score)
        idx = np.argmin(results)
        s2n = np.mean(results) / results[idx]
        z = z_range[idx]
        h = self.camera_config.z_to_h(z)
        # warning if the optimum is on the edge of the search space for l
        return h, s2n
