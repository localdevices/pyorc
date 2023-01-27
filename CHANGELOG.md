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
