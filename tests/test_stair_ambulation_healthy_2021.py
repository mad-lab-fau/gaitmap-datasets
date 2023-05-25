from typing import Type, Union

import pytest
from joblib import Memory

from gaitmap_datasets import config
from gaitmap_datasets.stair_ambulation_healthy_2021 import (
    StairAmbulationHealthy2021Full,
    StairAmbulationHealthy2021PerTest,
)
from gaitmap_datasets.stair_ambulation_healthy_2021.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_all_participants_and_tests,
    get_participant_metadata,
    get_pressure_insole_events,
    get_segmented_stride_list,
)

base_dir = config().stair_ambulation_healthy_2021


def test_get_all_participants():
    all_participants = get_all_participants(base_dir=base_dir)
    assert len(all_participants) == 20
    assert all(p.startswith("subject_") for p in all_participants)


def test_get_participant_metadata():
    metadata = get_participant_metadata("subject_01", base_dir=base_dir)
    assert metadata["subject_id"] == "001"


def test_get_all_tests():
    all_tests = get_all_participants_and_tests(base_dir=base_dir)

    assert len(all_tests) == 20
    assert all(p.startswith("subject_") for p in all_tests)
    assert all(len(t) == 28 for t in all_tests.values())
    assert all(list(t.keys()) == ["start", "end", "part"] for tests in all_tests.values() for t in tests.values())


def test_get_segmented_stride_list():
    segmented_stride_list = get_segmented_stride_list(
        base_dir=base_dir, participant_folder_name="subject_01", part="part_1"
    )
    assert len(segmented_stride_list) == 2
    assert list(segmented_stride_list.keys()) == ["left_sensor", "right_sensor"]
    for _k, v in segmented_stride_list.items():
        assert v.columns.tolist() == ["start", "end", "type", "z_level"]


@pytest.mark.parametrize("pressure", [True, False])
def test_load_data(snapshot, pressure):
    data = get_all_data_for_participant("subject_01", part="part_1", return_pressure_data=pressure, base_dir=base_dir)

    snapshot_data = data.iloc[:100]
    for sensor in snapshot_data.columns.get_level_values(0):
        snapshot.assert_match(snapshot_data[sensor], name=sensor)
        snapshot.assert_match(data[sensor].describe(), name=f"{sensor}_summary")


@pytest.mark.parametrize("include_pressure_data", [True, False])
@pytest.mark.parametrize("include_hip_sensor", [True, False])
@pytest.mark.parametrize("include_baro_data", [True, False])
def test_load_data_exclude_columns(include_pressure_data, include_baro_data, include_hip_sensor):
    data = get_all_data_for_participant(
        "subject_01",
        part="part_1",
        return_pressure_data=include_pressure_data,
        return_baro_data=include_baro_data,
        return_hip_sensor=include_hip_sensor,
        base_dir=base_dir,
    )

    if include_pressure_data:
        assert all(c in data.columns.get_level_values(1) for c in StairAmbulationHealthy2021Full._PRESSURE_COLUMNS)
    else:
        assert all(c not in data.columns.get_level_values(1) for c in StairAmbulationHealthy2021Full._PRESSURE_COLUMNS)

    if include_baro_data:
        assert "baro" in data.columns.get_level_values(1)
    else:
        assert "baro" not in data.columns.get_level_values(1)

    if include_hip_sensor:
        assert "hip_sensor" in data.columns.get_level_values(0)
    else:
        assert "hip_sensor" not in data.columns.get_level_values(0)


def test_get_pressure_insole_events():
    pressure_events = get_pressure_insole_events(participant_folder_name="subject_01", part="part_1", base_dir=base_dir)

    assert len(pressure_events) == 2
    assert list(pressure_events.keys()) == ["left_sensor", "right_sensor"]
    for v in pressure_events.values():
        assert v.columns.tolist() == ["start", "end", "ic", "tc", "min_vel", "pre_ic"]
        assert v.index.name == "s_id"


class TestDatasetCommon:
    dataset_class: Union[Type[StairAmbulationHealthy2021PerTest], Type[StairAmbulationHealthy2021Full]]

    @pytest.fixture(params=[StairAmbulationHealthy2021PerTest, StairAmbulationHealthy2021Full], autouse=True)
    def _dataset_class(self, request):
        self.dataset_class = request.param

    @pytest.mark.parametrize("include_pressure_data", [True, False])
    @pytest.mark.parametrize("include_hip_sensor", [True, False])
    @pytest.mark.parametrize("include_baro_data", [True, False])
    def test_columns_in_output(self, include_hip_sensor, include_pressure_data, include_baro_data):
        dataset = self.dataset_class(
            data_folder=base_dir,
            include_hip_sensor=include_hip_sensor,
            include_baro_data=include_baro_data,
            include_pressure_data=include_pressure_data,
        )
        dataset = dataset[0]

        imu_data = dataset.data
        assert all(c in imu_data.columns.get_level_values(0) for c in ["left_sensor", "right_sensor"])

        if include_hip_sensor:
            assert "hip_sensor" in imu_data.columns.get_level_values(0)
            assert imu_data.shape[1] == 3 * 6
        else:
            assert "hip_sensor" not in imu_data.columns.get_level_values(0)
            assert imu_data.shape[1] == 2 * 6

        if include_pressure_data:
            pressure_data = dataset.pressure_data
            assert all(c in pressure_data.columns.get_level_values(0) for c in ["left_sensor", "right_sensor"])
            assert all(
                c in pressure_data.columns.get_level_values(1) for c in StairAmbulationHealthy2021Full._PRESSURE_COLUMNS
            )
            assert pressure_data.shape[1] == 2 * len(StairAmbulationHealthy2021Full._PRESSURE_COLUMNS)
        else:
            with pytest.raises(ValueError):
                _ = dataset.pressure_data

        if include_baro_data:
            baro_data = dataset.baro_data
            sensors = ["left_sensor", "right_sensor"]
            if include_hip_sensor:
                sensors.append("hip_sensor")
            assert all(c in baro_data.columns.get_level_values(0) for c in sensors)
            assert baro_data.shape[1] == len(sensors)
        else:
            with pytest.raises(ValueError):
                _ = dataset.baro_data

    def test_all_data_same_length(self):
        dataset = self.dataset_class(
            data_folder=base_dir, include_baro_data=True, include_pressure_data=True, include_hip_sensor=True
        )
        dataset = dataset[0]

        assert dataset.data.shape[0] == dataset.pressure_data.shape[0] == dataset.baro_data.shape[0]

    def test_data_access_raises_error_if_not_single(self):
        dataset = self.dataset_class(
            data_folder=base_dir, include_baro_data=True, include_pressure_data=True, include_hip_sensor=True
        )

        with pytest.raises(ValueError):
            _ = dataset.data

        with pytest.raises(ValueError):
            _ = dataset.pressure_data

        with pytest.raises(ValueError):
            _ = dataset.baro_data

    @pytest.mark.parametrize("include_z_level", [True, False])
    def test_include_stride_border_columns(self, include_z_level):
        dataset = self.dataset_class(data_folder=base_dir)
        dataset = dataset[0]

        stride_borders = dataset.get_segmented_stride_list_with_type(return_z_level=include_z_level)
        for borders in stride_borders.values():
            if include_z_level:
                assert borders.columns.to_list() == ["start", "end", "type", "z_level"]
            else:
                assert borders.columns.to_list() == ["start", "end"]

    @pytest.mark.parametrize(
        "filter",
        [None, ["level"], ["level", "ascending"], ["level", "descending"], ["level", "ascending", "descending"]],
    )
    def test_filter_stride_list(self, filter):
        dataset = self.dataset_class(data_folder=base_dir)
        dataset = dataset[1]
        stride_borders = dataset.get_segmented_stride_list_with_type(stride_type=filter, return_z_level=True)

        all_stride_types = ["level", "ascending", "descending", "slope_ascending", "slope_descending"]

        if filter is None:
            expected_types = all_stride_types
            other_types = []
        else:
            expected_types = filter
            other_types = list(set(all_stride_types) - set(filter))

        for v in stride_borders.values():
            types = v["type"].to_list()
            if isinstance(dataset, StairAmbulationHealthy2021Full):
                # For the "perTest" dataset, we can not ensure that all stride types exist in a single test
                for t in expected_types:
                    assert t in types
            for t in other_types:
                assert t not in types

    def test_segmented_stride_list_property(self):
        dataset = self.dataset_class(data_folder=base_dir)
        dataset = dataset[0]

        stride_borders_1 = dataset.get_segmented_stride_list_with_type(return_z_level=False)
        stride_borders_2 = dataset.segmented_stride_list_
        assert stride_borders_1.keys() == stride_borders_2.keys()

        for k, val in stride_borders_1.items():
            assert val.equals(stride_borders_2[k])

    def test_metadata_property(self):
        dataset = self.dataset_class(data_folder=base_dir)
        dataset = dataset[0]

        metadata = dataset.metadata
        assert metadata["subject_id"] == "001"


class TestStairAmbulationHealthy2021PerTest:
    def test_index_shape(self):
        dataset = StairAmbulationHealthy2021PerTest(base_dir)
        assert dataset.index.shape == (20 * 26, 2)

    def test_cut_test(self):
        dataset = StairAmbulationHealthy2021PerTest(
            base_dir, memory=Memory(".cache"), include_pressure_data=True, include_baro_data=True
        )
        dataset.memory.clear(warn=False)
        dataset = dataset.get_subset(participant="subject_03")

        for subset in dataset:
            participant, test = subset.index.iloc[0]
            test = get_all_participants_and_tests(base_dir=base_dir)[participant][test]
            test_len = test["end"] - test["start"]
            assert subset.data.shape[0] == subset.pressure_data.shape[0] == subset.baro_data.shape[0] == test_len
            assert subset.data.index[0] == subset.pressure_data.index[0] == subset.baro_data.index[0] == 0
            assert (
                subset.data.index[-1]
                == subset.pressure_data.index[-1]
                == subset.baro_data.index[-1]
                == (test_len - 1) / subset.sampling_rate_hz
            )

        dataset.memory.clear(warn=False)

    def test_cut_test_segmented_stride_list(self):
        dataset = StairAmbulationHealthy2021PerTest(base_dir, memory=Memory(".cache"))
        dataset.memory.clear(warn=False)
        dataset = dataset.get_subset(participant="subject_03")

        for subset in dataset:
            participant, test = subset.index.iloc[0]
            test = get_all_participants_and_tests(base_dir=base_dir)[participant][test]
            for stride_borders in subset.segmented_stride_list_.values():
                # We can not really test much here, as we substract the start of the test from the stride borders.
                assert (stride_borders + test["start"] <= test["end"]).all().all()

                assert (stride_borders["end"] > stride_borders["start"]).all()

    def test_cut_test_pressure_insole_event_list(self):
        dataset = StairAmbulationHealthy2021PerTest(base_dir, memory=Memory(".cache"))
        dataset.memory.clear(warn=False)
        dataset = dataset.get_subset(participant="subject_03")

        for subset in dataset:
            participant, test = subset.index.iloc[0]
            test = get_all_participants_and_tests(base_dir=base_dir)[participant][test]
            for stride_borders in subset.pressure_insole_event_list_.values():
                # We can not really test much here, as we substract the start of the test from the stride borders.
                assert (stride_borders.fillna(0) + test["start"] <= test["end"]).all().all()

                assert (stride_borders["end"] > stride_borders["tc"]).all()
                assert (stride_borders["tc"] > stride_borders["start"]).all()
                assert (stride_borders["ic"] > stride_borders["tc"]).all()
                assert (stride_borders["pre_ic"].fillna(-1) < stride_borders["start"]).all()
                assert (stride_borders["start"] == stride_borders["min_vel"]).all()


class TestStairAmbulationHealthy2021Full:
    def test_index_shape(self):
        dataset = StairAmbulationHealthy2021Full(base_dir)
        assert dataset.index.shape == (20 * 2, 2)

    @pytest.mark.parametrize("ignore_manual_session_markers", [True, False])
    def test_ignore_session(self, ignore_manual_session_markers):
        dataset = StairAmbulationHealthy2021Full(
            base_dir,
            memory=Memory(".cache"),
            ignore_manual_session_markers=ignore_manual_session_markers,
            include_pressure_data=True,
            include_baro_data=True,
        )
        dataset.memory.clear(warn=False)

        # For these participants, the session we know that the session data was cut
        for participant in [4, 22, 24]:
            subset = dataset.get_subset(index=dataset.index.iloc[participant : participant + 1])
            participant, part = subset.index.iloc[0]
            full_session = get_all_participants_and_tests(base_dir=base_dir)[participant][f"full_session_{part}"]
            full_session_length = full_session["end"] - full_session["start"]

            assert subset.data.shape[0] == subset.pressure_data.shape[0] == subset.baro_data.shape[0]

            if ignore_manual_session_markers:
                assert subset.data.shape[0] >= full_session_length
            else:
                assert subset.data.shape[0] == full_session_length

            assert subset.data.index[0] == subset.pressure_data.index[0] == subset.baro_data.index[0] == 0
            assert (
                subset.data.index[-1]
                == subset.pressure_data.index[-1]
                == subset.baro_data.index[-1]
                == (subset.data.shape[0] - 1) / subset.sampling_rate_hz
            )

        # for these participants cutting should not make a difference (just a subset of the data)
        for participant in [1, 2]:
            subset = dataset.get_subset(index=dataset.index.iloc[participant : participant + 1])
            participant, part = subset.index.iloc[0]
            full_session = get_all_participants_and_tests(base_dir=base_dir)[participant][f"full_session_{part}"]
            full_session_length = full_session["end"] - full_session["start"]

            assert (
                subset.data.shape[0]
                == subset.pressure_data.shape[0]
                == subset.baro_data.shape[0]
                == full_session_length
            )
            assert subset.data.index[0] == subset.pressure_data.index[0] == subset.baro_data.index[0] == 0
            assert (
                subset.data.index[-1]
                == subset.pressure_data.index[-1]
                == subset.baro_data.index[-1]
                == (full_session_length - 1) / subset.sampling_rate_hz
            )

        dataset.memory.clear(warn=False)

    def test_stride_borders_cut_to_session(self):
        dataset = StairAmbulationHealthy2021Full(
            base_dir,
            memory=Memory(".cache"),
            ignore_manual_session_markers=False,
        )
        dataset.memory.clear(warn=False)

        for participant in [1, 2, 4, 22, 24]:
            subset = dataset.get_subset(index=dataset.index.iloc[participant : participant + 1])
            participant, part = subset.index.iloc[0]
            full_session = get_all_participants_and_tests(base_dir=base_dir)[participant][f"full_session_{part}"]
            for stride_borders in subset.segmented_stride_list_.values():
                assert (stride_borders + full_session["start"] <= full_session["end"]).all().all()
        dataset.memory.clear(warn=False)

    def test_pressure_insole_event_list_cut_to_session(self):
        dataset = StairAmbulationHealthy2021Full(
            base_dir,
            memory=Memory(".cache"),
            ignore_manual_session_markers=False,
        )
        dataset.memory.clear(warn=False)

        for participant in [1, 2, 4, 22, 24]:
            subset = dataset.get_subset(index=dataset.index.iloc[participant : participant + 1])
            participant, part = subset.index.iloc[0]
            full_session = get_all_participants_and_tests(base_dir=base_dir)[participant][f"full_session_{part}"]
            for stride_borders in subset.pressure_insole_event_list_.values():
                # We can not really test much here, as we substract the start of the test from the stride borders.
                assert (stride_borders.fillna(0) + full_session["start"] <= full_session["end"]).all().all()

                assert (stride_borders["end"] > stride_borders["tc"]).all()
                assert (stride_borders["tc"] > stride_borders["start"]).all()
                assert (stride_borders["ic"] > stride_borders["tc"]).all()
                assert (stride_borders["pre_ic"].fillna(-1) < stride_borders["start"]).all()
                assert (stride_borders["start"] == stride_borders["min_vel"]).all()
        dataset.memory.clear(warn=False)

    def test_test_list(self):
        dataset = StairAmbulationHealthy2021Full(
            base_dir,
            ignore_manual_session_markers=False,
        )
        dataset = dataset[1]

        test_list = dataset.test_list
        assert test_list.index.name == "roi_id"
        assert test_list.columns.to_list() == ["start", "end"]
        assert test_list.shape == (19, 2)
