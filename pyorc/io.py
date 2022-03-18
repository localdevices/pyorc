import xarray as xr
from .api.frames import Frames
from .api.piv import Velocimetry
from .api.transect import Transect

def open_file(fn, *args, **kwargs):
    if not("chunks" in kwargs):
        # add at least the time chunk
        kwargs["chunks"] = {"time": 1}
    ds = xr.open_dataset(fn, *args, **kwargs)
    if "frames" in ds:
        # assume the dataset is a frames DataArray
        assert hasattr(ds["frames"], "camera_config"), f"File {fn} does not contain a camera_config attribute."
        return Frames(ds["frames"], attrs=ds["frames"].attrs)
    elif "v_x" in ds:
        assert hasattr(ds, "camera_config"), f"File {fn} does not contain a camera_config attribute."
        # this can be either a Transect or a Velocimetry dataset
        if "x" in ds.dims:
            return Velocimetry(ds, attrs=ds.attrs)
        elif "points" in ds.dims:
            return Transect(ds, attrs=ds.attrs)
        else:
            raise IOError(f"File {fn} seems to velocimetry information but lacks the right dimensions (x, y) or (point)")
    else:
        raise IOError(f"File {fn} is not a valid pyorc compatible file")



