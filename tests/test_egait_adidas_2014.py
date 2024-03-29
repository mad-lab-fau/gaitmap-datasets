from pathlib import Path

import pytest

from gaitmap_datasets import EgaitAdidas2014, config
from gaitmap_datasets.egait_adidas_2014.helper import (
    get_all_participants_and_tests,
    get_mocap_data_for_participant_and_test,
    get_synced_stride_list,
)

base_dir = config().egait_adidas_2014

HERE = Path(__file__).parent


def test_get_all_participants():
    p_t = get_all_participants_and_tests(base_dir=base_dir)
    participants = list({c["participant"] for c in p_t})
    assert len(participants) == 20


def test_get_participants_and_tests():
    all_combis = get_all_participants_and_tests(base_dir=base_dir)
    assert len(all_combis) == 497


def test_get_mocap_data():
    shimmer3 = get_mocap_data_for_participant_and_test("015", "shimmer3", "normal", "normal", "1", base_dir=base_dir)
    shimmer2r = get_mocap_data_for_participant_and_test("015", "shimmer2r", "normal", "normal", "1", base_dir=base_dir)

    assert len(shimmer3) == len(shimmer2r) == 2
    # 7 seconds * 200 Hz, 2 feet * 2 angle_values * 3 directions
    assert shimmer3[1].shape == shimmer2r[1].shape == (7 * 200, 2 * 2 * 3)
    for sensor in ["left_sensor", "right_sensor"]:
        # 7 seconds * 200 Hz, 2 feet * 6 markers * 3 dimensions
        assert shimmer3[0][sensor].shape == shimmer2r[0][sensor].shape == (7 * 200, 6 * 3)


class TestDataset:
    def test_index(self):
        dataset = EgaitAdidas2014(data_folder=base_dir)
        assert dataset.index.shape[0] == 497
        assert dataset.index.columns.tolist() == [
            "participant",
            "sensor",
            "stride_length",
            "stride_velocity",
            "repetition",
        ]

    @pytest.mark.parametrize("sensor", ["shimmer3", "shimmer2r"])
    def test_sampling_rate(self, sensor):
        dataset = EgaitAdidas2014(data_folder=base_dir)
        subset = dataset.get_subset(participant="015", sensor=sensor)
        assert subset.sampling_rate_hz == (204.8 if sensor == "shimmer3" else 102.4)

    def test_data(self):
        trial = EgaitAdidas2014(data_folder=base_dir)[3]
        data = trial.data
        for sensor in ["left_sensor", "right_sensor"]:
            # The first time sample should be negative, as it is before the mocap data starts
            assert data[sensor].index[0] < 0
            # The last time sample should be positive, as it is after the mocap data ends
            assert data[sensor].index[-1] > 0

    def test_marker_position(self):
        trial = EgaitAdidas2014(data_folder=base_dir)[3]
        data = trial.marker_position_
        for sensor in ["left_sensor", "right_sensor"]:
            # The first sample should be 0, as it marks the start of the mocap data
            assert data[sensor].index[0] == 0

    def test_segmented_stride_list(self):
        trial = EgaitAdidas2014(data_folder=base_dir)[3]
        data = trial.segmented_stride_list_
        for sensor in ["left_sensor", "right_sensor"]:
            offset = trial.mocap_offset_s_[sensor] * trial.sampling_rate_hz
            # All values should be larger than the mocap offset, as all values are within the mocap data
            assert (data[sensor] > offset).all().all()

    def test_load_all(self):
        dataset = EgaitAdidas2014(data_folder=base_dir)
        with pytest.warns(UserWarning) as w:
            for p in dataset:
                assert len(p.data) > 0
                assert len(p.segmented_stride_list_) > 0
        assert len(w) == 2

    def test_stride_list_conversion(self):
        dataset = EgaitAdidas2014(data_folder=base_dir)[3]

        strides_mocap = get_synced_stride_list(*dataset.group, system="mocap", base_dir=dataset._data_folder_path)
        strides_imu = get_synced_stride_list(*dataset.group, system="imu", base_dir=dataset._data_folder_path)
        converted_mocap_to_imu = dataset.convert_events(strides_mocap, from_time_axis="mocap", to_time_axis="imu")
        converted_imu_to_mocap = dataset.convert_events(strides_imu, from_time_axis="imu", to_time_axis="mocap")

        for sensor in ["left_sensor", "right_sensor"]:
            assert converted_mocap_to_imu[sensor].equals(strides_imu[sensor])
            assert converted_imu_to_mocap[sensor].equals(strides_mocap[sensor].round(0).astype(int))

    def test_stride_list_conversion_roundtrip(self):
        dataset = EgaitAdidas2014(data_folder=base_dir)[3]

        stride_list = dataset.segmented_stride_list_
        roundtrip = dataset.convert_events(
            dataset.convert_events(stride_list, from_time_axis="imu", to_time_axis="mocap"),
            from_time_axis="mocap",
            to_time_axis="imu",
        )

        for sensor in ["left_sensor", "right_sensor"]:
            # The first sample should be 0, as it marks the start of the mocap data
            assert stride_list[sensor].equals(roundtrip[sensor])
