from pathlib import Path

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from gaitmap_datasets.egait_parameter_validation_2013.helper import CALIBRATION_FILE_NAMES
from gaitmap_datasets.utils.consts import SF_ACC
from gaitmap_datasets.utils.data_loading import load_bin_file
from gaitmap_datasets.utils.egait_loading_helper import (
    SHIMMER_DATA_LAYOUT,
    load_compact_cal_matrix,
    load_shimmer2_data,
)

HERE = Path(__file__).parent


def test_basic_data_loading():
    data = load_bin_file(HERE / "egait_validation_2013_test_data/P115_E4_left.dat", SHIMMER_DATA_LAYOUT)
    # Reference data produced by the matlab import script
    reference_data = pd.read_csv(HERE / "egait_validation_2013_test_data/P115_E4_left.csv", header=0)
    assert_frame_equal(data.astype(int), reference_data.drop(columns=["n_samples"]))


def test_calibration():
    data = load_bin_file(HERE / "egait_validation_2013_test_data/P115_E4_left.dat", SHIMMER_DATA_LAYOUT).astype(float)
    cal_matrix = load_compact_cal_matrix(HERE / "egait_validation_2013_test_data/A917.csv")
    calibrated_data = cal_matrix.calibrate_df(data, "a.u.", "a.u.")
    # Reference data produced by the matlab import script
    reference_data = pd.read_csv(HERE / "egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0)
    reference_data[SF_ACC] *= 9.81
    assert_frame_equal(calibrated_data, reference_data.drop(columns=["n_samples"]))


def test_data_loading_with_transformation():
    reference_data = {
        "left_sensor": pd.read_csv(HERE / "egait_validation_2013_test_data/P115_E4_left_calibrated.csv", header=0).drop(
            columns=["n_samples"]
        ),
        "right_sensor": pd.read_csv(
            HERE / "egait_validation_2013_test_data/P115_E4_right_calibrated.csv", header=0
        ).drop(columns=["n_samples"]),
    }

    data_path = {
        "left_sensor": HERE / "egait_validation_2013_test_data/P115_E4_left.dat",
        "right_sensor": HERE / "egait_validation_2013_test_data/P115_E4_right.dat",
    }

    for sensor in ["left_sensor", "right_sensor"]:
        data = load_shimmer2_data(
            data_path[sensor],
            HERE / "egait_validation_2013_test_data" / CALIBRATION_FILE_NAMES[sensor],
        )
        reference_data[sensor] = reference_data[sensor].astype(float)

        assert data.shape == reference_data[sensor].shape
        assert_series_equal(data["gyr_x"], reference_data[sensor]["gyr_y"], check_names=False)
        assert_series_equal(data["gyr_y"], reference_data[sensor]["gyr_x"], check_names=False)
        assert_series_equal(data["gyr_z"], -reference_data[sensor]["gyr_z"], check_names=False)
        assert_frame_equal(data[SF_ACC], reference_data[sensor][SF_ACC] * 9.81)
