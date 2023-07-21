# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.14.0] - unreleased

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
