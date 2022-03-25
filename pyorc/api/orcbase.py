import numpy as np
import xarray as xr
from pyorc import helpers

class ORCBase(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def camera_config(self):
        if not(hasattr(self, "_camera_config")):
            #  first set the camera config and shape
            self.set_camera_config()
        return self._camera_config

    @camera_config.setter
    def camera_config(self, cam_config):
        if isinstance(cam_config, str):
            # convert into a camera_config object
            from pyorc import get_camera_config
            self._camera_config = get_camera_config(cam_config)
        else:
            self._camera_config = cam_config

    @property
    def camera_shape(self):
        return self._camera_shape

    @camera_shape.setter
    def camera_shape(self, cam_shape):
        if isinstance(cam_shape, str):
            self._camera_shape = helpers.deserialize_attr(self._obj, "camera_shape", np.array)
        else:
            self._camera_shape = self._obj.camera_shape

    def set_camera_config(self):
        # set the camera config
        self.camera_config = self._obj.camera_config
        self.camera_shape = self._obj.camera_shape


    def add_xy_coords(
        self,
        xy_coord_data,
        coords,
        attrs_dict
    ):
        """
        add coordinate variables with x and y dimensions (2d) to existing xr.Dataset.

        :param xy_coord_data: list, one or several arrays with 2-dimensional coordinates
        :param coords: tuple with strings, indicating the dimensions of the data in xy_coord_data
        :param attrs_dict: list of dicts, containing attributes belonging to xy_coord_data, must have equal length as xy_coord_data
        :return: xr.Dataset, with added coordinate variables.
        """
        dims = tuple(coords.keys())
        xy_coord_data = [
            xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs=attrs,
                name=name
            ) for data, (name, attrs) in zip(xy_coord_data, attrs_dict.items())]
        # assign the coordinates
        frames_coord = self._obj.assign_coords({
            k: (dims, v) for k, v in zip(attrs_dict, xy_coord_data)
        })
        # add the attributes (not possible with assign_coords
        for k, v in zip(attrs_dict, xy_coord_data):
            frames_coord[k].attrs = v.attrs
        # update the DataArray
        return frames_coord

