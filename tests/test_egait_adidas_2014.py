from pathlib import Path

from gaitmap_datasets import config
from gaitmap_datasets.egait_adidas_2014.helper import (
    get_all_participants_and_tests,
    get_mocap_data_for_participant_and_test,
)

base_dir = config().egait_adidas_2014

HERE = Path(__file__).parent


def test_get_all_participants():
    p_t = get_all_participants_and_tests(base_dir=base_dir)
    participants = list(set(c["participant"] for c in p_t))
    assert len(participants) == 20


def test_get_participants_and_tests():
    all_combis = get_all_participants_and_tests(base_dir=base_dir)
    assert len(all_combis) == 497


def test_get_mocap_data():
    shimmer3 = get_mocap_data_for_participant_and_test("015", "shimmer3", "normal", "normal", "1", base_dir=base_dir)
    shimmer2r = get_mocap_data_for_participant_and_test("015", "shimmer2r", "normal", "normal", "1", base_dir=base_dir)

    assert len(shimmer3) == len(shimmer2r) == 2
    # 7 seconds * 200 Hz, 2 feet * 6 markers * 3 dimensions
    assert shimmer3[0].shape == shimmer2r[0].shape == (7 * 200, 2 * 6 * 3)
    # 7 seconds * 200 Hz, 2 feet * 2 angle_values * 3 directions
    assert shimmer3[1].shape == shimmer2r[1].shape == (7 * 200, 2 * 2 * 3)
