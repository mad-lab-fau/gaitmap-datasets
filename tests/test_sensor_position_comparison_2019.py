from typing import Type, Union

import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from pandas._testing import assert_frame_equal, assert_series_equal

import gaitmap_datasets.sensor_position_comparison_2019.helper as h
from gaitmap_datasets import config
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Mocap,
    SensorPositionComparison2019Segmentation,
)

base_dir = config().sensor_position_comparison_2019


@pytest.mark.parametrize("include_failed", [True, False])
def test_get_all_participants(include_failed):
    all_participants = h.get_all_participants(include_wrong_recording=include_failed, data_folder=base_dir)
    assert len(list(all_participants)) == 14 + int(include_failed)


def test_get_participant_metadata():
    metadata = h.get_metadata_participant("4d91", data_folder=base_dir)
    assert metadata["age"] == 28
    assert all([k in metadata for k in ["age", "bmi", "height", "weight", "shoe_size", "sensors"]])


def test_get_all_tests():
    tests = h.get_all_tests("4d91", data_folder=base_dir)
    assert len(list(tests)) == 7


def _convert_to_flat_str_index(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.to_flat_index()
    df.columns = ["__".join([str(k) for k in col]) for col in df.columns]
    return df


def test_get_mocap_data(snapshot):
    mocap_data = h.get_mocap_test("4d91", "fast_20", data_folder=base_dir)

    assert mocap_data.columns.get_level_values(1).unique().tolist() == ["x", "y", "z"]

    mocap_data = _convert_to_flat_str_index(mocap_data) * 1000

    snapshot.assert_match(mocap_data.describe(), name=f"example_mocap_summary")


def test_get_data(snapshot):
    imudata = h.get_imu_test("4d91", "fast_20", data_folder=base_dir)

    assert imudata.columns.get_level_values(1).unique().tolist() == [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "trigger",
    ]

    assert set(imudata.columns.get_level_values(0).unique().tolist()) == set(
        h.get_metadata_participant("4d91", data_folder=base_dir)["sensors"].keys()
    )

    imudata = _convert_to_flat_str_index(imudata)

    snapshot.assert_match(imudata.describe(), name=f"example_imu_summary")


class TestDatasetCommon:
    dataset_class: Union[Type[SensorPositionComparison2019Segmentation], Type[SensorPositionComparison2019Mocap]]

    @pytest.fixture(params=[SensorPositionComparison2019Segmentation, SensorPositionComparison2019Mocap], autouse=True)
    def _dataset_class(self, request):
        self.dataset_class = request.param

    def test_include_wrong_recording(self):
        dataset_false = self.dataset_class(data_folder=base_dir, include_wrong_recording=False)
        dataset_true = self.dataset_class(data_folder=base_dir, include_wrong_recording=True)

        assert len(dataset_false.groupby("participant")) == 14
        assert len(dataset_true.groupby("participant")) == 15

    def test_data_raises_error_if_not_single(self):
        dataset = self.dataset_class(data_folder=base_dir)
        with pytest.raises(ValueError):
            _ = dataset.data

        with pytest.raises(ValueError):
            _ = dataset.segmented_stride_list_

        with pytest.raises(ValueError):
            _ = dataset.segmented_stride_list_per_sensor_

        with pytest.raises(ValueError):
            _ = dataset.metadata

        if isinstance(dataset, SensorPositionComparison2019Mocap):
            with pytest.raises(ValueError):
                _ = dataset.marker_position_

            with pytest.raises(ValueError):
                _ = dataset.mocap_events_

    def test_segmented_stride_list(self):
        dataset = self.dataset_class(data_folder=base_dir)
        segmented_stride_list = dataset[0].segmented_stride_list_

        assert set(segmented_stride_list.keys()) == {"left", "right"}
        for v in segmented_stride_list.values():
            assert isinstance(v, pd.DataFrame)
            assert set(v.columns) == {"start", "end"}
            assert v.index.name == "s_id"

    def test_segmented_stride_list_per_sensor(self):
        dataset = self.dataset_class(data_folder=base_dir)
        segmented_stride_list_per_sensor = dataset[0].segmented_stride_list_per_sensor_
        segmented_stride_list = dataset[0].segmented_stride_list_

        assert set(segmented_stride_list_per_sensor.keys()) == {
            *h.get_foot_sensor("left", True),
            *h.get_foot_sensor("right", True),
        }

        for foot in ["left", "right"]:
            for s in h.get_foot_sensor(foot, True):
                assert_frame_equal(segmented_stride_list_per_sensor[s], segmented_stride_list[foot])


class TestMocapDataset:
    @pytest.mark.parametrize("include_failed", [True, False])
    def test_len(self, include_failed):
        dataset = SensorPositionComparison2019Mocap(data_folder=base_dir, include_wrong_recording=include_failed)
        assert len(dataset) == 7 * (14 + int(include_failed))

    def test_test_cut(self):
        dataset = SensorPositionComparison2019Mocap(
            data_folder=base_dir, include_wrong_recording=False, memory=Memory(".cache")
        )
        dataset.memory.clear(warn=False)
        subset = dataset.get_subset(participant="4d91")

        for test in subset:
            participant, test_name = test.group
            data = test.data
            metadata = test.metadata
            assert (
                data.shape[0]
                == metadata["imu_tests"][test_name]["stop_idx"] - metadata["imu_tests"][test_name]["start_idx"] + 1
            )

        dataset.memory.clear(warn=False)

    def test_mocap_event_list(self):
        dataset = SensorPositionComparison2019Mocap(data_folder=base_dir)
        event_list = dataset[0].mocap_events_

        assert set(event_list.keys()) == {"left", "right"}
        for v in event_list.values():
            assert isinstance(v, pd.DataFrame)
            assert set(v.columns) == {"start", "end", "ic", "tc", "min_vel"}
            assert v.index.name == "s_id"

    def test_padding_samples(self):
        cache = Memory(".cache")
        cache.clear(warn=False)
        # We use a loop instead of parameterize to be able to clear the cache only after all paddings ran
        for padding_s in [0, 1, 2, 2.5]:
            dataset_no_padding = SensorPositionComparison2019Mocap(
                data_folder=base_dir, data_padding_s=0, memory=cache
            )[0]
            dataset = SensorPositionComparison2019Mocap(data_folder=base_dir, data_padding_s=padding_s, memory=cache)[0]

            assert dataset_no_padding.data_padding_imu_samples == 0
            assert dataset.data_padding_imu_samples == np.round(padding_s * dataset.sampling_rate_hz)

            data_no_padding = dataset_no_padding.data
            data = dataset.data

            expected_length = data_no_padding.shape[0] + 2 * dataset.data_padding_imu_samples
            assert data.shape[0] == expected_length

            # The padded values have negative times. This means, if we select 0 on the time axis, the value needs to be
            # identical
            assert_series_equal(data_no_padding.loc[0], data.loc[0])

            # We test that the timeaxis is still continuous
            assert all(data.index.to_series().diff().dropna().unique() == 1 / dataset.sampling_rate_hz)

            # Test that segmented stridelist is just shifted
            segmented_stride_list = dataset.segmented_stride_list_
            segmented_stride_list_no_padding = dataset_no_padding.segmented_stride_list_
            for foot in ["left", "right"]:
                assert_frame_equal(
                    segmented_stride_list[foot] - dataset.data_padding_imu_samples,
                    segmented_stride_list_no_padding[foot],
                )

            # Test that mocap data and mocap events are unchanged
            assert_frame_equal(dataset.marker_position_, dataset_no_padding.marker_position_)
            for foot in ["left", "right"]:
                assert_frame_equal(dataset.mocap_events_[foot], dataset_no_padding.mocap_events_[foot])

        cache.clear(warn=False)

    # We use values for which we know, that they will result in a padding without "rest"
    @pytest.mark.parametrize("padding_s", [0, 100 / 204.8, 400 / 204.8, 1000 / 204.8])
    def test_conversion_with_padding_imu(self, padding_s):
        dataset = SensorPositionComparison2019Mocap(data_folder=base_dir, data_padding_s=padding_s)

        # As this is in IMU samples, it means, that 0 refers to the first sample in the returned data (i.e. including padding)
        mock_stride_list = pd.DataFrame([{"start": 0, "end": 100}])

        # Convertion to time
        converted = dataset.convert_with_padding(mock_stride_list, from_time_axis="imu", to_time_axis="time")
        assert converted.iloc[0]["start"] == -padding_s

        assert converted.iloc[0]["end"] == (100 - dataset.data_padding_imu_samples) / dataset.sampling_rate_hz
        # Test length of converted stride
        assert converted.iloc[0]["end"] - converted.iloc[0]["start"] == 100 / dataset.sampling_rate_hz

        # Convertion to mocap
        converted = dataset.convert_with_padding(mock_stride_list, from_time_axis="imu", to_time_axis="mocap")
        assert converted.iloc[0]["start"] == -np.round(
            dataset.data_padding_imu_samples / dataset.sampling_rate_hz * dataset.mocap_sampling_rate_hz_
        )

        # Test length of converted stride
        assert converted.iloc[0]["end"] - converted.iloc[0]["start"] == np.round(
            100 / dataset.sampling_rate_hz * dataset.mocap_sampling_rate_hz_
        )

    # We use values for which we know, that they will result in a padding without "rest"
    @pytest.mark.parametrize("padding_s", [0, 100 / 204.8, 400 / 204.8, 1000 / 204.8])
    def test_conversion_with_padding_mocap(self, padding_s):
        dataset = SensorPositionComparison2019Mocap(data_folder=base_dir, data_padding_s=padding_s)

        # This is in mocap samples, so 0 refers to the first sample in the mocap data -> the start of the test -> no padding
        mock_event_list = pd.DataFrame([{"start": 0, "end": 100, "ic": 60, "tc": 80}])

        # Conversion to time (completely independent of padding)
        converted = dataset.convert_with_padding(mock_event_list, from_time_axis="mocap", to_time_axis="time")
        assert converted.iloc[0]["start"] == 0

        assert converted.iloc[0]["end"] == 100 / dataset.mocap_sampling_rate_hz_
        assert converted.iloc[0]["ic"] == 60 / dataset.mocap_sampling_rate_hz_
        assert converted.iloc[0]["tc"] == 80 / dataset.mocap_sampling_rate_hz_
        # Test length of converted stride
        assert converted.iloc[0]["end"] - converted.iloc[0]["start"] == 100 / dataset.mocap_sampling_rate_hz_

        # Conversion to imu
        converted = dataset.convert_with_padding(mock_event_list, from_time_axis="mocap", to_time_axis="imu")
        assert converted.iloc[0]["start"] == dataset.data_padding_imu_samples
        assert converted.iloc[0]["end"] == dataset.data_padding_imu_samples + np.round(
            100 / dataset.mocap_sampling_rate_hz_ * dataset.sampling_rate_hz
        )
        assert converted.iloc[0]["ic"] == dataset.data_padding_imu_samples + np.round(
            60 / dataset.mocap_sampling_rate_hz_ * dataset.sampling_rate_hz
        )
        assert converted.iloc[0]["tc"] == dataset.data_padding_imu_samples + np.round(
            80 / dataset.mocap_sampling_rate_hz_ * dataset.sampling_rate_hz
        )

        # Test length of converted stride
        assert converted.iloc[0]["end"] - converted.iloc[0]["start"] == np.round(
            100 / dataset.mocap_sampling_rate_hz_ * dataset.sampling_rate_hz
        )
