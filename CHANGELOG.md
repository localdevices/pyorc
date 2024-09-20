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
