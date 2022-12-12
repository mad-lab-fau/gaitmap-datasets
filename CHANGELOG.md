# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - unreleased

### Added

- Support for the sensor position dataset (https://zenodo.org/record/5747173). This replaces the dedicated package that
  existed for this dataset before and makes some usability improvements.
  Note, when switching from the old package (https://github.com/mad-lab-fau/sensor_position_dataset_helper), you might 
  see small changes in the final results of calculated parameters.
  This is because the way padding is handled was changed to be more consistent and less confusing.
  (https://github.com/mad-lab-fau/mad-datasets/pull/5)

### Changed

- Added faster version to perform the initial alignment with the sensor data frame.
  We replaced performing the actual rotations with simple column swaps when possible.
  (https://github.com/mad-lab-fau/mad-datasets/pull/5)

### Fixed

- Upper version constrains for most packages are removed
- Loading the StairAmbulationHealthy2021 dataset now works with nilspodlib>=3.6

## [0.3.0] - 2022-07-07 

### Changed

- The StairAmbulationDataset loader was adapted to work with the official public version of the dataset.
- The script to generate the pressure events for the StairAmbulationDataset can now get the dataset path as a cli
  argument.
