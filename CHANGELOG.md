# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.16.0] - 2026-10-23

### Changed

- **Python 3.11 is now the minimum supported version; Python 3.9 and 3.10 support has been removed.**
  Upgrade Python before installing this release, or keep ``gaitmap-datasets<0.16`` in legacy environments.
  Python 3.14 remains excluded by the current loader dependency chain.
- Shared runtime dependency minimums now match gaitmap: pandas 3, NumPy 2.3.5, SciPy 1.16.3,
  joblib 1.5.2, and tpcp 3.0.0. NumPy is now an explicit dependency.
- Shared development dependency ranges now match gaitmap. CI tests Python 3.11 through 3.13 and
  the lowest direct dependency versions on Python 3.11.
- Raised the h5py minimum to 3.11.0 and the development Matplotlib minimum to 3.9.0 for the NumPy 2 stack.
  h5py documents NumPy 2 support in its [3.11 release notes](https://docs.h5py.org/en/stable/whatsnew/3.11.html).

### Fixed

- Kluge 2017 preprocessing now selects marker columns per foot without the removed pandas ``groupby(axis=1)`` API.
  Per-foot visibility masking, IMU coordinates, and reference stride selection remain unchanged.

- Egait 2013 loader properties now pass the selected participant identifier to their file helpers, rather than
  the tpcp group-label tuple. This restores IMU, stride-border, and GAITRite reference loading.

## [0.15.0] - 2026-03-07

### Added

- Dataset classes now expose a `download(...)` entry point. Automatic downloads are available for
  `EgaitAdidas2014`, `StairAmbulationHealthy2021Full`, `StairAmbulationHealthy2021PerTest`,
  `SensorPositionComparison2019Segmentation`, and `SensorPositionComparison2019Mocap`.
- Datasets without automatic download support now raise dataset-specific instructions that explain how to obtain the
  data manually.

### Changed

- The supported Python range is now explicitly `3.9` to `3.13`. Python `3.14` is not declared as supported yet,
  because the current `imucal` release chain does not support it.
- The minimum supported `tpcp` version is now `2.0`.
- `nilspodlib` is now required at `>=4.1.2`, which restores a clean dependency setup for Python `3.9`.
- The `c3d` requirement now allows `0.6+`, which lets the resolver use current NumPy `2.x` releases where supported.
- Dataset classes now validate their configured root folders earlier and raise dataset-specific error messages when the
  selected path does not match the expected folder structure.
  ([#11](https://github.com/mad-lab-fau/gaitmap-datasets/issues/11),
  [#21](https://github.com/mad-lab-fau/gaitmap-datasets/pull/21))

### Development

- The development workflow now uses `uv` for environment management, locking, task execution, and package builds.
- The package now uses a `src/` layout and `uv_build`.
- Development dependency upper bounds were removed and the lockfile was refreshed against current compatible releases.

## [0.14.0] - 2023-07-21

### Fixed

- The convert event methods for the EgaitAddidas2014 dataset had a bug that the offset was not taken into account 
  correctly.
- For two participants of the EgaitAddidas2014 dataset the offset between the systems was for some reason not consistent
  across all strides. We decided to ignore this, but warn the user about it.

### Changed

- The `mocap_sampling_rate_hz` method of the Kluge2017 dataset is now called `mocap_sampling_rate_hz_` 
  (with trailing underscore) to be consistent with the other datasets.

### Added

- SensorPositionComparison2019 now has a method to get the marker trajectories per stride. 

## [0.13.0] - 2023-07-13

### Fixed
- The egait segmentation validation dataset contained strides that had negative duration.
  We now remove them during loading.

## [0.12.0] - 2023-04-27

### Added
- New labels for the EgaitSegmentationValidation dataset have been created and are distributed with all future versions 
  of the dataset.
  These new labels are now the default labels used by the dataset.
  The old labels can be accessed using the `segmented_stride_list_original_` property.

## [0.11.0] - 2023-04-21

### Fixed
- Fixed further potential cases where the index of datasets was not deterministically sorted.


### Changed
- Config files are[CHANGELOG.md](CHANGELOG.md) frozen now

## [0.10.0] - 2023-03-10

### Fixed
- Fixed an issue that the index of `EgaitParameterValidation2013` was not deterministically sorted.

### Changed

- Removed pydantic as dependency and switched to custom logic to set configs.
  The use-facing API remains the same.

## [0.9.0] - 2023-02-02

## Added

- The `EgaitParameterValidation2013` dataset now has the option to exclude participant "P52", which appears to have 
  non-sensical imu data.
- The `EgaitAdidas2014` dataset was added.
  The dataset is not publicly available yet, but will be soon and contains data from healthy participants walking
  through a Vicon motion capture system.
- A `convert_segmented_stride_list` was added to convert segmented event list to min_vel eventlists.

## Changed

- The method `convert_with_padding` of the `SensorPositionComparison2019` dataset is now called 
  `convert_events_with_padding`.

## [0.8.0] - 2023-30-01

### Added
- It is now possible to create a global config file containing the local path to the datasets.
  This allows you to use the datasets without having to specify the path to the dataset every time.
  Further, it means that you don't have to modify source code (just a config file) to run the examples and tests during 
  development.
  See the README for more details.
  (https://github.com/mad-lab-fau/gaitmap-datasets/pull/8)

## [0.7.0] - 2023-23-01

### Added
- Support for the new alternative calibration files for the `egait_parameter_validation_2013` dataset.
  Basically, we found out that the original calibration files were not that good and added new ones to the public 
  dataset, that should result in significantly better results when calculating spatial gait parameters.
- All datasets now have images visualizing the applied coordinate system transformations.
- The stair pyshoe dataset has now a new index column describing the number of levels the participant walked during a 
  single trail.

## [0.6.0] - 2022-21-12

### Added
- The PyShoe Dataset (https://github.com/utiasSTARS/pyshoe)

### Changed
- The sensor position dataset now reports the mocap distance in m instead of mm to be consistent with the use of SI 
  units.

## [0.5.0] - 2022-13-12

Changed package name mad-datasets to gaitmap-datasets!
All old releases will still be available under the old name.
All new releases will be available under the new name.

## [0.4.0] - 2022-12-12

### Added

- Support for the sensor position dataset (https://zenodo.org/record/5747173). This replaces the dedicated package that
  existed for this dataset before and makes some usability improvements.
  Note, when switching from the old package (https://github.com/mad-lab-fau/sensor_position_dataset_helper), you might 
  see small changes in the final results of calculated parameters.
  This is because the way padding is handled was changed to be more consistent and less confusing.
  (https://github.com/mad-lab-fau/gaitmap-datasets/pull/5)

### Changed

- Added faster version to perform the initial alignment with the sensor data frame.
  We replaced performing the actual rotations with simple column swaps when possible.
  (https://github.com/mad-lab-fau/gaitmap-datasets/pull/5)

### Fixed

- Upper version constrains for most packages are removed
- Loading the StairAmbulationHealthy2021 dataset now works with nilspodlib>=3.6

## [0.3.0] - 2022-07-07 

### Changed

- The StairAmbulationDataset loader was adapted to work with the official public version of the dataset.
- The script to generate the pressure events for the StairAmbulationDataset can now get the dataset path as a cli
  argument.
