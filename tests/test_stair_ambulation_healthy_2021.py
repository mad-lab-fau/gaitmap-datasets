from pathlib import Path
from typing import Type, Union

import pytest

from mad_datasets.stair_ambulation_healthy_2021 import StairAmbulationHealthy2021Full, StairAmbulationHealthy2021PerTest
from mad_datasets.stair_ambulation_healthy_2021.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_all_participants_and_tests,
    get_participant_metadata,
)

base_dir = Path("/home/arne/Documents/repos/work/datasets/stair-ambulation-data-ba-liv")


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
    assert all([len(t) == 28 for t in all_tests.values()])
    assert all([list(t.keys()) == ["start", "end", "part"] for tests in all_tests.values() for t in tests.values()])


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
        dataset = dataset.get_subset(index=dataset.index.iloc[:1])

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
        dataset = dataset.get_subset(index=dataset.index.iloc[:1])

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


class TestStairAmbulationHealthy2021PerTest:
    def test_index_shape(self):
        dataset = StairAmbulationHealthy2021PerTest(base_dir)
        assert dataset.index.shape == (20 * 26, 2)


class TestStairAmbulationHealthy2021Full:
    def test_index_shape(self):
        dataset = StairAmbulationHealthy2021Full(base_dir)
        assert dataset.index.shape == (20 * 2, 2)
