# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 

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
