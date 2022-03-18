import numpy as np
import xarray as xr
from pyorc import helpers

class Transect(xr.Dataset):
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vector_to_scalar(self, angle_method=0, v_x="v_x", v_y="v_y"):
        """
        Turns velocity vectors into effective velocities over cross-section, by computing the perpendicular velocity component
        :param v_x: DataArray(t, points), time series in cross section points with x-directional velocities
        :param v_y: DataArray(t, points), time series in cross section points with y-directional velocities
        :param angle_method: if set to 0, then angle of cross section is determined with left to right bank coordinates,
            otherwise, it is determined per section
        :return: v_eff: DataArray(t, points), time series in cross section points with velocities perpendicular to cross section

        """
        xs = self["x"].values
        ys = self["y"].values
        # find points that are located on the area of interest
        idx = np.isfinite(xs)
        xs = xs[idx]
        ys = ys[idx]
        if angle_method == 0:
            x_left, y_left = xs[0], ys[0]
            x_right, y_right = xs[-1], ys[-1]
            angle_da = np.arctan2(x_right - x_left, y_right - y_left)
        else:
            # start with empty angle
            angle = np.zeros(ys.shape)
            angle_da = np.zeros(self["x"].shape)
            angle_da[:] = np.nan

            for n, (x, y) in enumerate(zip(xs, ys)):
                # determine the angle of the current point with its neighbours
                # check if we are at left bank
                # first estimate left bank angle
                undefined = True  # angle is still undefined
                m = 0
                while undefined:
                    # go one step to the left
                    m -= 1
                    if n + m < 0:
                        # we are at the left bank, so angle with left neighbour is non-existing.
                        x_left, y_left = xs[n], ys[n]
                        # angle_left = np.nan
                        undefined = False
                    else:
                        x_left, y_left = xs[n + m], ys[n + m]
                        if not ((x_left == x) and (y_left == y)):
                            # profile points are in another pixel, so estimate angle
                            undefined = False
                            angle_left = np.arctan2(x - x_left, y - y_left)

                # estimate right bank angle
                undefined = True  # angle is still undefined
                m = 0
                while undefined:
                    # go one step to the left
                    m += 1
                    if n + m >= len(xs) - 1:
                        angle_right = np.nan
                        undefined = False
                    else:
                        x_right, y_right = xs[n + m], ys[n + m]
                        if not ((x_right == x) and (y_right == y)):
                            # profile points are in another pixel, so estimate angle
                            undefined = False
                            angle_right = np.arctan2(x_right - x, y_right - y)
                angle[n] = np.nanmean([angle_left, angle_right])
            # add angles to array meant for data array
            angle_da[idx] = angle

        # compute angle of flow direction (i.e. the perpendicular of the cross section) and add as DataArray to ds_points
        flow_dir = angle_da - 0.5 * np.pi

        # compute per velocity vector in the dataset, what its angle is
        v_angle = np.arctan2(self[v_x], self[v_y])
        # compute the scalar value of velocity
        v_scalar = (self[v_x] ** 2 + self[v_y] ** 2) ** 0.5

        # compute difference in angle between velocity and perpendicular of cross section
        angle_diff = v_angle - flow_dir
        # compute effective velocity in the flow direction (i.e. perpendicular to cross section
        v_eff = np.cos(angle_diff) * v_scalar
        v_eff.attrs = {
            "standard_name": "velocity",
            "long_name": "velocity in perpendicular direction of cross section, measured by angle in radians, measured from up-direction",
            "units": "m s-1",
        }
        # set name
        v_eff.name = "v_eff_nofill"  # there still may be gaps in this series
        self["v_eff_nofill"] = v_eff

    def get_xyz_perspective(self, reverse_y=None):
        z = (self.zcoords - self.camera_config.gcps["z_0"] + self.camera_config.gcps["h_ref"]).values
        Ms = [self.camera_config.get_M_reverse(depth) for depth in z]
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        cols, rows = zip(*[helpers.xy_to_perspective(x, y, self.camera_config.resolution, M, reverse_y=self.camera_config.shape[0]) for x, y, M in zip(self.x.values, self.y.values, Ms)])


        # xp, yp = helpers.xy_to_perspective(*np.meshgrid(x, np.flipud(y)), self.camera_config.resolution, M)
        # dirty trick to ensure y coordinates start at the top in the right orientation
        shape_y, shape_x = self.camera_shape
        rows = shape_y - np.array(rows)
        cols = np.array(cols)
        return cols, rows


    def get_river_flow(self):
        """
        Integrates time series of depth averaged velocities [m2 s-1] into cross-section integrated flow [m3 s-1]
        estimating one or several quantiles over the time dimension.
    
        :return: ds: xarray dataset, including variable "Q" which is cross-sectional integrated river flow [m3 s-1] for one or several quantiles.
            The time dimension no longer exists because of the quantile mapping, the point dimension no longer exists because of the integration over width
        """

        if "q" not in self:
            raise ValueError('Dataset must contain variable "q", which is the depth-integrated velocity [m2 s-1], perpendicular to cross-section. Create this with Transect.get_q')
        # integrate over the distance coordinates (s-coord)
        Q = self["q"].fillna(0.0).integrate(coord="scoords")
        Q.attrs = {
            "standard_name": "river_discharge",
            "long_name": "River Flow",
            "units": "m3 s-1",
        }
        # set name
        Q.name = "Q"
        self["river_flow"] = Q


    def get_q(self, v_corr=0.9, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        # aggregate to a limited set of quantiles
        ds = self.quantile(quantiles, dim="time", keep_attrs=True)
        x = ds["xcoords"].values
        y = ds["ycoords"].values
        z = ds["zcoords"].values
        z_0 = self.camera_config.gcps["z_0"]
        h_ref =  self.camera_config.gcps["h_ref"]
        h_a = ds.h_a
        # add filled surface velocities with a logarithmic profile curve fit
        ds["v_eff"] = helpers.velocity_fill(x, y, z, ds["v_eff_nofill"], z_0, h_ref, h_a, groupby="quantile")
        # compute q for both non-filled and filled velocities
        ds["q_nofill"] = helpers.depth_integrate(z, ds["v_eff_nofill"], z_0, h_ref, h_a, v_corr=v_corr, name="q_nofill")
        ds["q"] = helpers.depth_integrate(z, ds["v_eff"], z_0, h_ref, h_a, v_corr=v_corr, name="q")
        return ds


    def plot(self):
        """

        :return:
        """
        raise NotImplementedError