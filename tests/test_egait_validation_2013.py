from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from gaitmap_datasets.egait_parameter_validation_2013 import EgaitParameterValidation2013
from gaitmap_datasets.egait_parameter_validation_2013.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_gaitrite_parameters,
    get_segmented_stride_list,
)
from gaitmap_datasets.utils.consts import SF_COLS

base_dir = Path("/home/arne/Documents/repos/work/datasets/eGaIT_database")

HERE = Path(__file__).parent


def test_get_all_participants():
    assert len(get_all_participants(base_dir=base_dir)) == 101


def test_get_all_data_for_participant():
    data = get_all_data_for_participant("P115", use_alternative_calibrations=False, base_dir=base_dir)

    reference_data = {
        "left_sensor": pd.read_csv(HERE / "egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
        "right_sensor": pd.read_csv(
            HERE / "egait_validation_2013_test_data/P115_E4_right_calibrated.csv", header=0
        ).drop(columns=["n_samples"]),
    }

    assert list(data.keys()) == ["left_sensor", "right_sensor"]

    for sensor in ["left_sensor", "right_sensor"]:
        assert data[sensor].shape == reference_data[sensor].shape
        # The loaded data is transformed to the common sensor frame definition
        # We test a couple of columns, but in general we trust that the transformation is correct
        if sensor == "left_sensor":
            assert_series_equal(data[sensor]["acc_x"], -reference_data[sensor]["acc_y"] * 9.81, check_names=False)
        else:
            assert_series_equal(data[sensor]["acc_x"], reference_data[sensor]["acc_y"] * 9.81, check_names=False)


def test_get_stride_borders():
    data = get_segmented_stride_list("P115", base_dir=base_dir)

    assert list(data.keys()) == ["left_sensor", "right_sensor"]

    for sensor in ["left_sensor", "right_sensor"]:
        assert data[sensor].columns.tolist() == ["start", "end"]
        assert data[sensor].index.name == "s_id"

    assert len(data["left_sensor"]) == 9
    assert len(data["right_sensor"]) == 8


def test_get_gaitrite_parameters():
    data = get_gaitrite_parameters("P115", base_dir=base_dir)

    assert list(data.keys()) == ["left_sensor", "right_sensor"]

    for sensor in ["left_sensor", "right_sensor"]:
        assert data[sensor].columns.tolist() == ["stride_length", "stride_time", "stance_time", "swing_time"]
        assert data[sensor].index.name == "s_id"
        # Test that the meter conversion worked
        assert (data[sensor].stride_length < 1).all()

    assert len(data["left_sensor"]) == 8
    assert len(data["right_sensor"]) == 7


class TestEgaitParameterValidation2013Dataset:
    def test_index(self):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        assert len(dataset) == 101

    def test_sampling_rate(self):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        assert dataset.sampling_rate_hz == 102.4

    def test_imu_data(self):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        imu_data = dataset.get_subset(participant="P115").data

        reference_data = get_all_data_for_participant("P115", base_dir=base_dir)

        for sensor in ["left_sensor", "right_sensor"]:
            assert_frame_equal(imu_data[sensor].reset_index(drop=True), reference_data[sensor])
            assert_series_equal(
                pd.Series(imu_data[sensor].index),
                pd.Series(reference_data[sensor].index / dataset.sampling_rate_hz),
                check_names=False,
            )
            assert imu_data[sensor].index.name == "time [s]"
            assert imu_data[sensor].columns.tolist() == SF_COLS

    def test_stride_borders(self):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        stride_list = dataset.get_subset(participant="P115").segmented_stride_list_

        reference_data = get_segmented_stride_list("P115", base_dir=base_dir)

        for sensor in ["left_sensor", "right_sensor"]:
            assert_frame_equal(stride_list[sensor], reference_data[sensor])

    def test_gait_parameters(self):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        parameters = dataset.get_subset(participant="P115").gaitrite_parameters_

        reference_data = get_gaitrite_parameters("P115", base_dir=base_dir)

        for sensor in ["left_sensor", "right_sensor"]:
            assert_frame_equal(parameters[sensor], reference_data[sensor])

    @pytest.mark.parametrize("attribute", ["data", "segmented_stride_list_", "gaitrite_parameters_"])
    def test_raises_error_if_not_single(self, attribute):
        dataset = EgaitParameterValidation2013(data_folder=base_dir)
        with pytest.raises(ValueError):
            getattr(dataset, attribute)

    def test_alternative_calibrations(self):
        # This is a stupid test that simple tests that if data is loaded with alternative calibrations it is different
        # from the data loaded with the default calibrations.
        dataset = EgaitParameterValidation2013(data_folder=base_dir, use_alternative_calibrations=False)
        imu_data = dataset.get_subset(participant="P115").data
        dataset_alternative = EgaitParameterValidation2013(data_folder=base_dir, use_alternative_calibrations=True)
        imu_data_alternative_calibrations = dataset_alternative.get_subset(participant="P115").data

        for sensor in ["left_sensor", "right_sensor"]:
            assert not imu_data[sensor].equals(imu_data_alternative_calibrations[sensor])
