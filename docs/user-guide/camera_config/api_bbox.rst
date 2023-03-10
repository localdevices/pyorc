.. _camera_config_api_bbox:

The area of interest can theoretically be provided directly, simply by providing
a ``shapely.geometry.Polygon`` with 5 bounding points as follows (pseudo-code):

.. code-block:: python

    cam_config.bbox = Polygon(...)

However, this is quite risky, as you are then responsible
for ensuring that the area of interest is rectangular, has exactly 4 corners and fits in the FOV. Currently, there are no checks
and balances in place, to either inform the user about wrongfully supplied Polyons, or Polygons that are entirely
outside of the FOV.

Therefore, a much more intuitive approach is to use ``set_bbox_from_corners``. You simply supply 4 approximate
corner points of the area of interest *within the camera FOV*. **pyorc** will then find the best planar bounding box
around these roughly chosen corner points and return this for you. A few things to bear in mind while choosing these:

* Ensure you provide the corner points in the right order. So no diagonal order, but always along the expected Polygon
  bounds.
* If you intend to process multiple videos with the same camera configuration, ensure you choose the points wide
  enough so that with higher water levels, they will likely still give a good fit around the water body of interest.
* *Important*: if water follows a clear dominant flow direction (e.g. in a straight relatively uniform section) then
  you may use the angular filter later on, to remove spurious velocities that are not in the flow direction. In order
  to make the area of interest flow direction aware, ensure to provide the points in the following order:

    - upstream left-bank
    - downstream left-bank
    - downstream right-bank
    - upstream right-bank

  where left and right banks are defined as if you are looking in downstream direction.

Below we show how the corners are provided to the existing ``cam_config``.

.. code-block::

    corners = [
        [255, 118],
        [1536, 265],
        [1381, 1019],
        [88, 628]
    ]
    cam_config.set_bbox_from_corners(corners)

This yields the bounding box shown in the figure above, which is the same as the one shown in the perspective below.
You can see that the rectangular area is chosen such that the chosen corner points at least fit in the bounding box,
and the orientation is chosen such that it follows the middle line between the chosen points as closely as possible.

.. image:: ../../_images/wark_cam_config_persp.jpg
