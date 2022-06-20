from pathlib import Path

import pytest

from mad_datasets.stair_ambulation_healthy_2021._dataset import (
    StairAmbulationHealthy2021PerTest,
    StairAmbulationHealthy2021Full,
)
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
    assert all([len(t) == 27 for t in all_tests.values()])
    assert all([list(t.keys()) == ["start", "end", "part"] for tests in all_tests.values() for t in tests.values()])


@pytest.mark.parametrize("pressure", [True, False])
def test_load_data(snapshot, pressure):
    data = get_all_data_for_participant("subject_01", part=1, return_pressure_data=pressure, base_dir=base_dir)

    snapshot_data = data.iloc[:100]
    for sensor in snapshot_data.columns.get_level_values(0):
        snapshot.assert_match(snapshot_data[sensor], name=sensor)
        snapshot.assert_match(data[sensor].describe(), name=f"{sensor}_summary")


class TestStairAmbulationHealthy2021PerTest:
    def test_per_test_shape(self):
        dataset = StairAmbulationHealthy2021PerTest(base_dir)
        assert dataset.index.shape == (20 * 26, 2)


class TestStairAmbulationHealthy2021Full:
    def test_per_test_shape(self):
        dataset = StairAmbulationHealthy2021Full(base_dir)
        assert dataset.index.shape == (20 * 2, 2)
