from pathlib import Path

from mad_datasets.egait_segmentation_validation_2013.helper import get_all_participants, get_all_data_for_participant, \
    get_segmented_stride_list
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
