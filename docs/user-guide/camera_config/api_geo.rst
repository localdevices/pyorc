.. _camera_config_api_geo:

Below, we show what the configuration would look like if we would add the Rijksdriehoek projection to our camera
configuration. You can see that the code is converted into a Well-Known-Text format, so that it can also easily be
stored in a generic text (json) format.

.. code-block:: python

    import pyorc
    cam_config = pyorc.CameraConfig(height=1080, width=1920, crs=32631)
    cam_config

    {
        "height": 1080,
        "width": 1920,
        "crs": "PROJCRS[\"WGS 84 / UTM zone 31N\",BASEGEOGCRS[\"WGS 84\",ENSEMBLE[\"World Geodetic System 1984 ensemble\",MEMBER[\"World Geodetic System 1984 (Transit)\"],MEMBER[\"World Geodetic System 1984 (G730)\"],MEMBER[\"World Geodetic System 1984 (G873)\"],MEMBER[\"World Geodetic System 1984 (G1150)\"],MEMBER[\"World Geodetic System 1984 (G1674)\"],MEMBER[\"World Geodetic System 1984 (G1762)\"],MEMBER[\"World Geodetic System 1984 (G2139)\"],ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1]],ENSEMBLEACCURACY[2.0]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]],ID[\"EPSG\",4326]],CONVERSION[\"UTM zone 31N\",METHOD[\"Transverse Mercator\",ID[\"EPSG\",9807]],PARAMETER[\"Latitude of natural origin\",0,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",3,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"Scale factor at natural origin\",0.9996,SCALEUNIT[\"unity\",1],ID[\"EPSG\",8805]],PARAMETER[\"False easting\",500000,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"(E)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"(N)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]],USAGE[SCOPE[\"Engineering survey, topographic mapping.\"],AREA[\"Between 0\u00b0E and 6\u00b0E, northern hemisphere between equator and 84\u00b0N, onshore and offshore. Algeria. Andorra. Belgium. Benin. Burkina Faso. Denmark - North Sea. France. Germany - North Sea. Ghana. Luxembourg. Mali. Netherlands. Niger. Nigeria. Norway. Spain. Togo. United Kingdom (UK) - North Sea.\"],BBOX[0,0,84,6]],ID[\"EPSG\",32631]]",
        "resolution": 0.05,
        "window_size": 10,
        "dist_coeffs": [
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ]
        ],
        "camera_matrix": [
            [
                1920.0,
                0.0,
                960.0
            ],
            [
                0.0,
                1920.0,
                540.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ]
    }
