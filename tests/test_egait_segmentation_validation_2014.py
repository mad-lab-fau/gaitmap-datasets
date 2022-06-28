from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from mad_datasets.egait_segmentation_validation_2014 import EgaitSegmentationValidation2014
from mad_datasets.egait_segmentation_validation_2014.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_segmented_stride_list,
)
from mad_datasets.utils.consts import SF_COLS

base_dir = Path("/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation")

HERE = Path(__file__).parent


def test_get_all_participants():
    participants = get_all_participants(base_dir=base_dir)
    cohorts, tests, participant_ids = zip(*participants)
    assert set(cohorts) == {"control", "pd", "geriatric"}
    assert set(tests) == {"4x10m", "free_walk"}
    assert len(participants) == 45


def test_get_data():
    # We test loading for **all** participants, because there are too many werid things that we need to check for
    for cohort, test, participant_id in get_all_participants(base_dir=base_dir):
        data = get_all_data_for_participant(participant_id, cohort, test, base_dir=base_dir)
        assert list(data.keys()) == ["left_sensor", "right_sensor"]
        for sensor in ["left_sensor", "right_sensor"]:
            if sensor == "right_sensor" and (cohort, test, participant_id) == ("control", "free_walk", "GA214026"):
                assert data[sensor] is None
            else:
                assert data[sensor].shape[0] > 0
                assert data[sensor].columns.tolist() == SF_COLS


def test_get_segmented_stride_list():
    # We test loading for **all** participants, because there are too many werid things that we need to check for
    for cohort, test, participant_id in get_all_participants(base_dir=base_dir):
        stride_list = get_segmented_stride_list(participant_id, cohort, test, base_dir=base_dir)
        assert list(stride_list.keys()) == ["left_sensor", "right_sensor"]
        for sensor in ["left_sensor", "right_sensor"]:
            if sensor == "right_sensor" and (cohort, test, participant_id) == ("control", "free_walk", "GA214026"):
                assert stride_list[sensor] is None
            else:
                assert stride_list[sensor].columns.tolist() == ["start", "end"]
                assert len(stride_list[sensor]) > 0


class TestDataset:
    @pytest.mark.parametrize("exclude", [True, False])
    def test_index(self, exclude):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir, exclude_incomplete_participants=exclude)

        assert dataset.index.shape[0] == 45 - (1 if exclude else 0)
        assert dataset.index.columns.tolist() == ["cohort", "test", "participant"]

    def test_sampling_rate(self):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir)
        assert dataset.sampling_rate_hz == 102.4

    def test_imu_data_missing(self):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir)

        # This is a participant, with missing data:
        subset = dataset.get_subset(participant="GA214026")
        assert len(subset.data) == 1

        # This is a participant, without missing data:
        subset = dataset.get_subset(participant="GA214030")
        imu_data = subset.data

        reference_data = get_all_data_for_participant("GA214030", "control", "free_walk", base_dir=base_dir)

        assert len(imu_data) == 2

        for sensor in ["left_sensor", "right_sensor"]:
            assert_frame_equal(imu_data[sensor].reset_index(drop=True), reference_data[sensor])
            assert_series_equal(
                pd.Series(imu_data[sensor].index),
                pd.Series(reference_data[sensor].index / dataset.sampling_rate_hz),
                check_names=False,
            )
            assert imu_data[sensor].index.name == "time [s]"
            assert imu_data[sensor].columns.tolist() == SF_COLS

    def test_correct_participant_excluded(self):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir, exclude_incomplete_participants=True)

        for p in dataset:
            assert len(p.data) == 2

    def test_stride_borders_missing(self):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir)

        # This is a participant, with missing data:
        subset = dataset.get_subset(participant="GA214026")
        assert len(subset.segmented_stride_list_) == 1

        # This is a participant, without missing data:
        subset = dataset.get_subset(participant="GA214030")
        stride_list = subset.segmented_stride_list_

        assert len(stride_list) == 2

        reference_data = get_segmented_stride_list("GA214030", "control", "free_walk", base_dir=base_dir)

        for sensor in ["left_sensor", "right_sensor"]:
            assert_frame_equal(stride_list[sensor], reference_data[sensor])

    @pytest.mark.parametrize("attribute", ["data", "segmented_stride_list_"])
    def test_raises_error_if_not_single(self, attribute):
        dataset = EgaitSegmentationValidation2014(data_folder=base_dir)
        with pytest.raises(ValueError):
            getattr(dataset, attribute)
