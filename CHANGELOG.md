## [0.8.8] = 2025-07-14
### Added
### Changed
### Deprecated
### Removed
### Fixed
- issues with plotting cross sections in edge cases with water levels equal to lowest point or above highest point
- fixed problem with situations where either the cross section data from a shapefile or the camera configuration
  did not contain CRS information. This is now correctly parsed when creating a `CrossSection` instance.

## [0.8.7] = 2025-06-30
### Added
- CLI option `--cross_wl`
- A new 3-point bounding box estimation, ideal for selecting a bounding box in strongly oblique cases. First select left
  bank, then right bank, then a point up- or downstream from the selected line for the size of the bounding box.
- `rvec` and `tvec` are written to camera configuration after fitting.
### Changed
- CLI option `--cross_wl` now replaces `--cross` for optical water level estimation. `--cross` is only used for
  discharge calculation
- Plotting in camera objective is accelerated
- water level detection with `CrossSection.detect_water_level` can now consume either `min_z` and `max_z` (for levels
  in the original coordinate system) or `min_h` and `max_h` (for levels using the local `h_ref` datum if provided).
  These minimum and maximum levels can be used to pre-condition the level range that the algoithm will seek in.
- Pose fitting can now be performed with a pre-defined camera matrix and set of distortion coefficients. This is very
  useful when a user has already pre-calibrated these parameters. It will improve the fit of the `rvec` and `tvec`,
  i.e. rotation and translation vectors.
- Changed code coverage reporting from Codecov to sonarqube for code coverage reports. Codecov caused issues with
  numba code.
### Deprecated
### Removed
### Fixed
- incorrect estimation of optical water level if `bank="near"` was used. This resulted in only a smaller portion of the
  cross section being used. Now the full nearby side is used.

## [0.8.6] = 2025-05-16
### Added
- added options `--k1`, `--k2` and `--focal_length` to command line interface for cases where
  focal length, and distortion coefficients are already known
### Changed
- optimization of intrinsics can now also be done with partly already known data. If k1 and k2 are known
  these can be passed as camera
- Debug messaging increased in `service.velocimetry`.
### Deprecated
### Removed
### Fixed


## [0.8.5] = 2025-03-25
### Added
- option `--cross` can now also be provided at service level. Only relevant
  for applications that want to run pyorc command line programmatically.
### Changed
### Deprecated
### Removed
### Fixed
- Small bug fix in plotting routines, causing `mode="camera"` to result in flipped results
  over the y-axis of the local projection.

## [0.8.4] = 2025-03-17
### Added
### Changed
- modified the `Frames.project(method="numpy")` to only map indices once.
  This severely reduces the resources required for the method and accelerates
  the workflow when little resources are available. `method="numpy"` is now the
  recommended method and therefore now also the default. Results are identical to
  the original numpy-based method.
### Deprecated
### Removed
### Fixed

## [0.8.3] = 2025-03-16
### Added
### Changed
### Deprecated
### Removed
### Fixed
- hotfix with missing "velocity_flow_subprocess" in service module

## [0.8.2] = 2025-03-16
### Added
### Changed
### Deprecated
### Removed
### Fixed
- fixed problem with plotting under shallow angles in `mode="camera"`

## [0.8.1] = 2025-03-14
### Added
- One can now also use a single image as input on the command-line interface for camera calibration.

### Changed
### Deprecated
### Removed
### Fixed
- Problem with using `zip(..., strict=False)`. This is a default ruff correction but breaks many functions in
  Python 3.9, still used a lot on Raspberry Pi.
- Water level plot in transect plot sections would plot up-side-down. This is now corrected.

## [0.8.0] = 2025-03-13
### Added
- Through a new class `CrossSection`, a user can perform water level detection on a provided image frame.
- Plot functionalities of `CrossSection` are extended.
- Documentation of new `CrossSection` class with user guide and API description.
- New `mode="3d"` option for plotting `CameraConfig` objects. This gives a 3d rotatable perspective plot of the situation.
  3d plots of `CrossSection` objects are default, and can be combined with `CameraConfig.plot(mode="3d")`

### Changed
- class `CrossSection` interpolates all coordinates over the length of the cross section, not over the left-right
  width. This ensures that also if entirely vertical profile parts are found, unique coordinates are returned from
  selected locations in the cross section.

### Deprecated
- `pyorc.Frames.get_piv` option `engine=openpiv` is now deprecated.

### Removed
### Fixed
- `CameraConfig.rvec` and `CameraConfig.tvec` can now be immutable properties. This makes the solution more stable and prevents
  unnecessary iterations in estimating rvec and tvec.
- `CameraConfig.rvec` and `CameraConfig.tvec` are now always defined in the coordinate system of the camera
  configuration, not in coordinates relative to the ground control point mean. This makes rvec and tvec directly
  usable without knowledge of the original control points.

### Security

## [0.7.2] = 2025-02-14
### Added
- New class CrossSection. This is to prepare for water level estimation functionalities.
  It provides several geometrical operations along a cross section. This is documented yet.
  and may change significantly in the near future.

### Changed
- `cli.cli_utils.get_gcps_optimized_fit` now export rotation and translation vectors also, for
  later use in front end functionalities (e.g. show pose of camera interactively).
### Deprecated
### Removed
### Fixed
### Security

## [0.7.1] - 2024-12-13
### Added
### Changed
progress bars while reading can be configured with new flag `progress`

### Deprecated
### Removed
### Fixed
Reading of last frame in video often got errors. This is now more robust
Writing with `frames.to_video` became very slow with the latest video reader, this has been fixed, it is now very fast.

### Security


## [0.7.0] - 2024-12-10
### Added
`get_piv` now uses several engines, `engine="numba"` is a lot fastr
### Changed
Reading frames is now a lot more efficient as they are read in bulks (20 by default). As a result, very large videos
can be processed efficiently.
### Deprecated
openpiv is still default, but may become deprecated in future versions.
### Removed
### Fixed
### Security


## [0.6.1] - 2024-09-26
### Added
### Changed
### Deprecated
### Removed
### Fixed
- fixing `rasterio` to version <1.4.0 to allow 2d array transforms
### Security


## [0.6.0] - 2024-09-20
### Added
A logo with modifications in trademark guidelines in TRADEMARK.md and README.md.
Logo is also shown in the online documentation on https://localdevices.github.io/pyorc
### Changed
`Frames.project` with `method="numpy"` is improved so that it also works well in heavily undersampled areas.
`Video` instances defaulting with `lazy=False`. This in most cases increases the speed of video treatment significantly.
For large videos with large memory requirements, videos can be opened with `lazy=True`.
### Deprecated
### Removed
### Fixed
The legacy `setup.py` has been replaced by a `pyproject.toml` using flit installer.
### Security


## [0.5.6] - 2024-06-28
### Added
### Changed
`Frames.project` with `method="numpy"` is improved so that it also works well in heavily undersampled areas.
### Deprecated
`Video` instances will default with `lazy=False` in v0.6.0. A warning message will appear for now
### Removed
### Fixed
### Security


## [0.5.5] - 2024-05-15
### Added
### Changed
### Deprecated
### Removed
### Fixed
Plots of videos that are projected with `method="numpy"` were not correctly projected in commnad-line use, as they were still using `cv` as method. Now, `method="numpy"` is forced on plots as well, when using the command-line interface (direct use of the API requires the user to ensure the projection method is correct).

Projection with `method="numpy"` sometimes results in missing values in the result when parts of the objective are outside the camera view. This consequently results in only zero values when difference filtering is applied. This has been fixed.
### Security


## [0.5.4] - 2024-04-19
### Added
`Video` instances can now be made with `lazy=False` (default is `True`). In this case, frames of videos are read in one go instead of lazily per frame. This consumes more memory but is a lot faster. For tested workflows from opening videos to writing and masking of velocities, this reduces the compute time with about 20%.
### Changed
### Deprecated
### Removed
### Fixed
### Security


## [0.5.3] - 2023-11-10
### Added
`frames.project` now has a `method` option which allows for choosing projection using opencv-methods (`method="cv"`)
which is currently still the default, or numpy-like operations (`method="numpy"`). `method="numpy"` is new and we
may change the default behaviour to this option in a later release as it is much more robust in cases with a lot
of lens distortion and where part of the area of interest is outside of the field of view.

Videos can now be rotated, e.g. if a camera was setup with vertical orientation, without having a metadata tag for this.
Video rotation can be defined in the camera configuration, on the CLI with `--rotation` using either `90`, `180` or `270`
as measured in degrees, or for individual single videos as an additional input argument to `pyorc.Video`
e.g. `pyorc.Video(..., rotation=90)`.
### Changed
Some default values for estimating the intrinsic lens parameters from control points are changed. We now estimate the
first two barrel distortion coefficients if enough information for constraining them is available.
### Deprecated
### Removed
### Fixed
In some cases, ground control points were not correctly projected from real-world to camera coordinates. This seemed
to be an issue only on windows machines. This is now fixed.
### Security


## [0.5.2] - 2023-08-16
### Added
`--lowmem` option added in the CLI for very large videos and/or low resource devices.
### Changed
Calling of the service level now works through one function call that executes all actions within the service.
All inputs to these functions MUST be in deserialized form. This generalizes the approach by which a service is
executed which was necessary for getting ready for nodeOpenRiverCam (not released yet), which will be a shell around
pyopenrivercam for scalable computation across a cloud. API / CLI users do not notice a difference.
### Deprecated
### Removed
### Fixed
Notebook 02 in the examples folder contained a deprecation error with the stabilize option for opening videos. This
has been corrected and functionality description improved.
### Security

## [0.5.1] - 2023-06-27
### Added
### Changed
### Deprecated
### Removed
### Fixed
- removed the strict cartopy dependency. This enables pip installation for users that are not interested in
  geographical plotting. Enables also installation on raspi platforms (only 64-bit!)
- Transects sometimes gave infinite discharge when areas with zero depth received a small velocity. This has now
  been resolved.
### Security


## [0.5.0] - 2023-05-24
### Added
- make it a lot easier to get well-calibrated ground control and lens parameters at the same time. we now do this
  by optimizing the lens'  focal length and (if enough ground control is provided) barrel distortion whilst fitting
  the perspective to the user-provided ground control points.
- provide the fitted ground control points in the interface so that the user can immediately see if the ground control
  points are well fitted or if anything seems to be wrong with one or more control points.
- feature stabilization on command line which consequently provided user-interfacing to select non-moving areas by
  pointing and clicking.
### Changed
- Much-improved stabilization for non-stable videos
- stabilization can also be configured in CameraConfig level to accomodate slightly moving fixed rigs
- h_a can be provided on command-line instead of in recipe
### Deprecated
### Removed
- old velocimetry.filter_... methods are now entirely removed
### Fixed
### Security


## [0.4.0] - 2023-03-10
### Added
The most notable change is that the code now includes an automatically installed command-line interface. This
will facilitate a much easier use by a large user group. Also the documentation is fully updated to include all
functionalities for both command-line users and API users. In detail we have the following additions:
- First release of a command-line interface for the entire process of camera configuration, processing and preparing
  outputs and figures.
- Service layer that makes it easy for developers to connect pyorc to apps such as GUIs or dashboards.
- Full user guide with description of both the command-line interface and API.
### Changed
- Small modifications and additions in the API to accomodate the command-line interface building.
### Deprecated
### Removed
### Fixed
- Bug fixes in the video objects causing videos to sometimes not open properly because of missing frames
### Security


## [0.3.3] - 2023-01-24
### Added
- Documentation for velocimetry and masking
### Changed
- filtering changed into mask as subclass pyorc.Velocimetry.mask
### Deprecated
- all functions starting with pyorc.Velocimetry.filter....
### Removed
### Fixed
### Security


## [0.3.2] - 2023-01-05
### Added
### Changed
### Deprecated
### Removed
### Fixed
- scipy 1.10.0 release causing regression error in xarray.Dataset.interp
### Security

## [0.3.1] - 2022-12-02
### Added
- perspective transform with 6 gcps
### Changed
### Deprecated
### Removed
### Fixed
- several small bugs
### Security


## [0.3.0] - 2022-11-13
### Added
- Video.set_lens_calibration added automated camera calibration with chessboard
- User guide
- Improved pytest code coverage

### Changed
- several API modifications to accommodate lens calibration and 6-point orthorectification
- CameraConfig format changed
- CameraConfig.lens_parameters no longer used (replaced by camera_matrix and dist_coeffs)
- CameraConfig.gcps extended orthorectification with 6(+)-point x, y, z perspective option
- CameraConfig.set_corners changed into set_bbox_from_corners (setting a POLYGON as property bbox)

### Deprecated
### Removed
- CameraConfig.set_lens_parameters

### Fixed
### Security

## [0.2.4] - 2022-09-16
### Added
- pyorc.Video added stabilize option for video stabilization
- Start with user guide
- Improved pytest code coverage

### Changed

### Deprecated
### Removed
### Fixed

### Security


## [0.2.3] - 2022-08-10
### Added
### Changed
- pyorc.transect.get_q added method="log_interp" using a log-depth normalized velocity and linear interpolation

### Deprecated
### Removed
### Fixed
- pyorc.transect.get_q improved method="log_fit" (former "log_profile"), so that it works if no dry bed points are found

### Security


## [0.2.2] - 2022-08-01
### Added
- pytest

### Changed
### Deprecated
### Removed
### Fixed
- small improvements and bug fixes

### Security

## [0.2.1] - 2022-06-22
### Added
- Documentation

### Changed
- docstrings (numpy format, and completed)

### Deprecated
### Removed
### Fixed
- small improvements and bug fixes

### Security

## [0.2.0] - 2022-05-18
### Added
- API for entire library
- Example notebooks (+binders)
- Conda environment

### Changed
- Restructured code
- Data model in xarray (with lazy dask workflows)

### Deprecated
### Removed
### Fixed
### Security

## [0.1.0] - 2022-01-21
### Added
- First release

### Changed
### Deprecated
### Removed
### Fixed
### Security
