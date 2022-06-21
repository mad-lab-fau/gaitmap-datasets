from pathlib import Path

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from mad_datasets.egait_validation_2013.egait_loading_helper import (
    SHIMMER2_DATA_LAYOUT,
    load_compact_cal_matrix,
    load_shimmer2_data,
)
from mad_datasets.egait_validation_2013.helper import (
    CALIBRATION_FILE_NAMES,
    get_all_participants,
    get_all_data_for_participant,
    get_segmented_stride_list,
    get_gaitrite_parameters,
)
from mad_datasets.utils.consts import SF_ACC
from mad_datasets.utils.data_loading import load_bin_file

base_dir = Path("/home/arne/Documents/repos/work/datasets/eGaIT_database")


def test_basic_data_loading():
    data = load_bin_file(Path("./egait_validation_2013_test_data/P115_E4_left.dat"), SHIMMER2_DATA_LAYOUT)
    # Reference data produced by the matlab import script
    reference_data = pd.read_csv("./egait_validation_2013_test_data/P115_E4_left.csv", header=0)
    assert_frame_equal(data.astype(int), reference_data.drop(columns=["n_samples"]))


def test_calibration():
    data = load_bin_file(Path("./egait_validation_2013_test_data/P115_E4_left.dat"), SHIMMER2_DATA_LAYOUT).astype(float)
    cal_matrix = load_compact_cal_matrix(Path("./egait_validation_2013_test_data/A917.csv"))
    calibrated_data = cal_matrix.calibrate_df(data, "a.u.", "a.u.")
    # Reference data produced by the matlab import script
    reference_data = pd.read_csv("./egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0)
    assert_frame_equal(calibrated_data, reference_data.drop(columns=["n_samples"]))


def test_data_loading_with_transformation():
    data = load_shimmer2_data(
        Path("./egait_validation_2013_test_data/P115_E4_left.dat"),
        Path("./egait_validation_2013_test_data/P115_E4_right.dat"),
        Path("./egait_validation_2013_test_data"),
        CALIBRATION_FILE_NAMES,
    )
    reference_data = {
        "left_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
        "right_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_right_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
    }

    for sensor in ["left_sensor", "right_sensor"]:
        # Ensure that we have a proper dtype and not uint from loading
        data[sensor] = data[sensor].astype(float)
        reference_data[sensor] = reference_data[sensor].astype(float)

        assert data[sensor].shape == reference_data[sensor].shape
        assert_series_equal(data[sensor]["gyr_x"], reference_data[sensor]["gyr_y"], check_names=False)
        assert_series_equal(data[sensor]["gyr_y"], reference_data[sensor]["gyr_x"], check_names=False)
        assert_series_equal(data[sensor]["gyr_z"], -reference_data[sensor]["gyr_z"], check_names=False)
        assert_frame_equal(data[sensor][SF_ACC], reference_data[sensor][SF_ACC])


def test_get_all_participants():
    assert len(get_all_participants(base_dir=base_dir)) == 101


def test_get_all_data_for_participant():
    data = get_all_data_for_participant("P115", base_dir=base_dir)

    reference_data = {
        "left_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
        "right_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_right_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
    }

    assert list(data.keys()) == ["left_sensor", "right_sensor"]

    # We only test the shape and the acc data. We verified the rest in a previous test
    for sensor in ["left_sensor", "right_sensor"]:
        assert data[sensor].shape == reference_data[sensor].shape
        assert_frame_equal(data[sensor][SF_ACC], reference_data[sensor][SF_ACC])


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
