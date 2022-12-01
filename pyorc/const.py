import operator
# attributes for PIV variables
PIV_ATTRS = {
    "v_x": {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flow element center velocity vector, x-component",
        "units": "m s-1",
        "coordinates": "lon lat",
    },
    "v_y": {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flow element center velocity vector, x-component",
        "units": "m s-1",
        "coordinates": "lon lat"
    },
    "s2n": {
        "standard_name": "ratio",
        "long_name": "signal to noise ratio",
        "units": "",
        "coordinates": "lon lat",
    },
    "corr": {
        "standard_name": "correlation_coefficient",
        "long_name": "correlation coefficient between frames",
        "units": "",
        "coordinates": "lon lat",
    }
}

# attributes for geographical coordinate variables
GEOGRAPHICAL_ATTRS = {
    "xs": {
        "axis": "X",
        "long_name": "x-coordinate in Cartesian system",
        "units": "m",
    },
    "ys": {
        "axis": "Y",
        "long_name": "y-coordinate in Cartesian system",
        "units": "m",
    },
    "lon": {
        "long_name": "longitude",
        "units": "degrees_east",
    },
    "lat": {
        "long_name": "latitude",
        "units": "degrees_north",
    }
}

# attributes for camera perspective coordinate values
PERSPECTIVE_ATTRS = {
    "xp": {
        "axis": "X",
        "long_name": "column in camera perspective",
        "units": "-"
    },
    "yp": {
        "axis": "Y",
        "long_name": "row in camera perspective",
        "units": "-"
    },
}

# typical arguments used for making an animation writer.
VIDEO_ARGS = {
    "fps": 10,
    "extra_args": ["-vcodec", "libx264"],
    "dpi": 120,
}

ANIM_ARGS = {
    "interval": 20,
    "blit": False
}

ENCODING_PARAMS = {
    "zlib": True,
    "dtype": "int16",
    "scale_factor": 0.01,
    "_FillValue": -9999
}

ENCODE_VARS = ["v_x", "v_y", "corr", "s2n"]
ENCODING = {k: ENCODING_PARAMS for k in ENCODE_VARS}

FIGURE_ARGS = {
    "figsize": (16, 9),
    "frameon": False,
}


CLASSIFY_MOVING_CAM = [
    {
        "method": "kmeans",
        "op": operator.ge
    },
    {
        "method": "dist",
        "q_threshold": 0.5,
        "op": operator.ge
    }
]

CLASSIFY_STANDING_CAM = [
    {
        "method": "kmeans",
        "op": operator.le
    },
    {
        "method": "dist",
        "q_threshold": 0.8,
        "op": operator.le
    }
]

CLASSIFY_CAM = {
    "fixed": CLASSIFY_STANDING_CAM,
    "moving": CLASSIFY_MOVING_CAM
}
