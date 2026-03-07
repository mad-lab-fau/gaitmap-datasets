"""Tests for dataset-specific path validation errors."""

import pytest

from gaitmap_datasets import (
    EgaitAdidas2014,
    EgaitParameterValidation2013,
    EgaitSegmentationValidation2014,
    Kluge2017,
    SensorPositionComparison2019Mocap,
    SensorPositionComparison2019Segmentation,
    StairAmbulationHealthy2021Full,
    StairAmbulationHealthy2021PerTest,
)
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Hallway, PyShoe2019Stairs, PyShoe2019Vicon


@pytest.mark.parametrize(
    ("dataset_class", "expected_fragments"),
    [
        (
            EgaitParameterValidation2013,
            ["EgaitParameterValidation2013", "ValidationRawData", "GoldStandard_GaitRite"],
        ),
        (
            EgaitSegmentationValidation2014,
            ["EgaitSegmentationValidation2014", "calibrationFiles", "GoldStandard_StrideBorders"],
        ),
        (
            SensorPositionComparison2019Segmentation,
            ["SensorPositionComparison2019", "`data` and `calibrations`", "nested `data` folder"],
        ),
        (
            SensorPositionComparison2019Mocap,
            ["SensorPositionComparison2019", "`data` and `calibrations`", "nested `data` folder"],
        ),
        (
            EgaitAdidas2014,
            ["EgaitAdidas2014", "`data` and `calibrations`", "nested `data` folder"],
        ),
        (
            PyShoe2019Vicon,
            ["PyShoe2019", "vicon/processed", "dataset root folder"],
        ),
        (
            PyShoe2019Hallway,
            ["PyShoe2019", "hallway", "dataset root folder"],
        ),
        (
            PyShoe2019Stairs,
            ["PyShoe2019", "stairs/test", "dataset root folder"],
        ),
        (
            StairAmbulationHealthy2021PerTest,
            ["StairAmbulationHealthy2021", "`healthy` and `calibrations`", "nested `healthy` folder"],
        ),
        (
            StairAmbulationHealthy2021Full,
            ["StairAmbulationHealthy2021", "`healthy` and `calibrations`", "nested `healthy` folder"],
        ),
        (
            Kluge2017,
            ["Kluge2017", "raw_data_export.mat", "subjects.csv", "stride_parameters.csv"],
        ),
    ],
)
def test_dataset_classes_raise_user_friendly_errors_for_wrong_root(dataset_class, expected_fragments, tmp_path):
    """Dataset classes fail early with dataset-specific root-folder guidance."""
    dataset = dataset_class(tmp_path)

    with pytest.raises(ValueError, match="expected dataset structure") as exc:
        dataset.create_index()

    message = str(exc.value)
    for fragment in expected_fragments:
        assert fragment in message
