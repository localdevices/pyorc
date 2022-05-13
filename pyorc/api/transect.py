import numpy as np
import xarray as xr

from xarray.core import utils

from pyorc import helpers
from .plot import _Transect_PlotMethods
from .orcbase import ORCBase

@xr.register_dataset_accessor("transect")
class Transect(ORCBase):
    def __init__(self, xarray_obj):
        super(Transect, self).__init__(xarray_obj)

    def vector_to_scalar(self, v_x="v_x", v_y="v_y"):
        """
        Set "v_eff" and "v_dir" variables as effective velocities over cross-section, and its angle

        :param v_x: str, variable containing (t, points) time series in cross section with x-directional velocities.
        :param v_y: str, variable containing (t, points) time series in cross section with y-directional velocities.
        :return: DataArray(t, points), time series in points with velocities perpendicular to cross section.

        """
        xs = self._obj["x"].values
        ys = self._obj["y"].values
        # find points that are located on the area of interest
        idx = np.isfinite(xs)
        xs = xs[idx]
        ys = ys[idx]
        # compute per velocity vector in the dataset, what its angle is
        v_angle = np.arctan2(self._obj[v_x], self._obj[v_y])
        # compute the scalar value of velocity
        v_scalar = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5

        # compute difference in angle between velocity and perpendicular of cross section
        flow_dir = self._obj["v_dir"]
        angle_diff = v_angle - flow_dir
        # compute effective velocity in the flow direction (i.e. perpendicular to cross section)
        v_eff = np.cos(angle_diff) * v_scalar
        v_eff.attrs = {
            "standard_name": "velocity",
            "long_name": "velocity in perpendicular direction of cross section, measured by angle in radians, measured from up-direction",
            "units": "m s-1",
        }
        # set name
        v_eff.name = "v_eff_nofill"  # there still may be gaps in this series
        self._obj["v_eff_nofill"] = v_eff


    def get_xyz_perspective(self, M=None, xs=None, ys=None, mask_outside=True):
        """
        Get camera-perspective column, row coordinates from cross-section points.

        :return:
        """
        if xs is None:
            xs = self._obj.x.values
        if ys is None:
            ys = self._obj.y.values
        # compute bathymetry as measured in local height reference (such as staff gauge)
        if self.camera_config.gcps["h_ref"] is None:
            h_ref = 0.
        else:
            h_ref = self.camera_config.gcps["h_ref"]
        hs = self.camera_config.z_to_h(self._obj.zcoords).values
        # zs = (self._obj.zcoords - self.camera_config.gcps["z_0"] + h_ref).values
        if M is None:
            Ms = [self.camera_config.get_M(h, reverse=True) for h in hs]
        else:
            # use user defined M instead
            Ms = [M for _ in hs]
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        cols, rows = zip(*[
            helpers.xy_to_perspective(
                x,
                y,
                self.camera_config.resolution,
                M,
                reverse_y=self.camera_config.shape[0]
            ) for x, y, M in zip(xs, ys, Ms)
        ])

        # ensure y coordinates start at the top in the right orientation
        shape_y, shape_x = self.camera_shape
        rows = shape_y - np.array(rows)
        cols = np.array(cols)
        if mask_outside:
            # remove values that do not fit in the frames
            cols[np.any([cols < 0, cols > self.camera_shape[1]], axis=0)] = np.nan
            rows[np.any([rows < 0, rows > self.camera_shape[0]], axis=0)] = np.nan

        return cols, rows


    def get_river_flow(self, q_name="q", Q_name="river_flow"):
        """
        Integrate time series of depth averaged velocities [m2 s-1] into cross-section integrated flow [m3 s-1]
        estimating one or several quantiles over the time dimension. Depth average velocities must first have been
        estimated using get_q. A variable "Q" will be added to Dataset, with only "quantiles" as dimension.

        """
        if "q" not in self._obj:
            raise ValueError(f'Dataset must contain variable "{q_name}", which is the depth-integrated velocity [m2 s-1], perpendicular to cross-section. Create this with ds.transect.get_q')
        # integrate over the distance coordinates (s-coord)
        Q = self._obj[q_name].fillna(0.0).integrate(coord="scoords")
        Q.attrs = {
            "standard_name": "river_discharge",
            "long_name": "River Flow",
            "units": "m3 s-1",
        }
        # set name
        Q.name = "Q"
        self._obj[Q_name] = Q


    def get_q(self, v_corr=0.9, fill_method="zeros"):
        """
        Depth integrated velocity for quantiles of time series using a correction v_corr between surface velocity and
        depth-average velocity.

        :param v_corr: float, optional, correction factor (default: 0.9)
        :return: xr.Dataset, Transect for selected quantiles in time, including "q".
        """
        # aggregate to a limited set of quantiles
        assert(fill_method in ["zeros", "log_profile", "interpolate"]), f'fill_method must be "zeros", "log_profile", or "interpolate", instead "{fill_method}" given'
        ds = self._obj
        x = ds["xcoords"].values
        y = ds["ycoords"].values
        z = ds["zcoords"].values
        # add filled surface velocities with a logarithmic profile curve fit
        depth = self.camera_config.get_depth(z, self.h_a)
        if fill_method == "zeros":
            ds["v_eff"] = ds["v_eff_nofill"].fillna(0.)
        elif fill_method == "log_profile":
            ds["v_eff"] = helpers.velocity_fill(x, y, depth, ds["v_eff_nofill"], groupby="quantile")
        elif fill_method == "interpolate":
            depth = ds.zcoords*0 + self.camera_config.get_depth(ds.zcoords, self.h_a)
            # interpolate gaps in between known values
            ds["v_eff"] = ds["v_eff_nofill"].interpolate_na(dim="points")
            # anywhere where depth == 0, remove values
            ds["v_eff"] = ds["v_eff"].where(depth > 0)
            # fill NAs with zeros
            ds["v_eff"] = ds["v_eff"].fillna(0.)
        # compute q for both non-filled and filled velocities
        ds["q_nofill"] = helpers.depth_integrate(depth, ds["v_eff_nofill"], v_corr=v_corr, name="q_nofill")
        ds["q"] = helpers.depth_integrate(depth, ds["v_eff"], v_corr=v_corr, name="q")
        return ds

    plot = utils.UncachedAccessor(_Transect_PlotMethods)
