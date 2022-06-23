from pathlib import Path

from mad_datasets.egait_segmentation_validation_2013.helper import get_all_participants

base_dir = Path("/home/arne/Documents/repos/work/datasets/eGaIT_database_segmentation")

HERE = Path(__file__).parent


def test_get_all_participants():
    participants = get_all_participants(base_dir=base_dir)
    cohorts, tests, participant_ids = zip(*participants)
    assert set(cohorts) == {"control", "pd", "geriatric"}
    assert set(tests) == {"4x10m", "free_walk"}
    assert len(participants) == 45
