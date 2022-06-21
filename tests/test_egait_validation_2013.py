from pathlib import Path

import pandas as pd
from pandas._testing import assert_frame_equal

from mad_datasets.egait_validation_2013.helper import load_shimmer2_data, SHIMMER2_DATA_LAYOUT
from mad_datasets.utils.data_loading import load_bin_file

base_dir = Path("/home/arne/Documents/repos/work/datasets/eGaIT_database")


def test_basic_data_loading():
    data = load_bin_file(Path("./egait_validation_2013_test_data/P115_E4_left.dat"), SHIMMER2_DATA_LAYOUT)
    # Reference data produced by the matlab import script
    reference_data = pd.read_csv("./egait_validation_2013_test_data/P115_E4_left.csv", header=0)
    assert_frame_equal(data.astype(int), reference_data.drop(columns=["n_samples"]))


def test_data_loading_with_transformation():
    data = load_shimmer2_data(
        Path("./egait_validation_2013_test_data/P115_E4_left.dat"),
        Path("./egait_validation_2013_test_data/P115_E4_right.dat"),
    )
    reference_data = {
        "left_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_left.csv", header=0).drop(
            columns=["n_samples"]
        ),
        "right_sensor": pd.read_csv("./egait_validation_2013_test_data/P115_E4_right.csv", header=0).drop(
            columns=["n_samples"]
        ),
    }

    for sensor in ["left_sensor", "right_sensor"]:
        # Ensure that we have a proper dtype and not uint from loading
        data[sensor] = data[sensor].astype(float)
        reference_data[sensor] = reference_data[sensor].astype(float)

        assert data[sensor].shape == reference_data[sensor].shape
        assert data[sensor]["gyr_x"].equals(reference_data[sensor]["gyr_y"])
        assert data[sensor]["gyr_y"].equals(reference_data[sensor]["gyr_x"])
        assert data[sensor]["gyr_z"].equals(-reference_data[sensor]["gyr_z"])

