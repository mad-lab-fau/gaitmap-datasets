# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - unreleased

### Fixed

- Uper version constrains for most packages are removed
- Loading the StairAmbulationHealthy2021 dataset now works with nilspodlib>=3.6

## [0.3.0] - 2022-07-07 

### Changed

- The StairAmbulationDataset loader was adapted to work with the official public version of the dataset.
- The script to generate the pressure events for the StairAmbulationDataset can now get the dataset path as a cli
  argument.
