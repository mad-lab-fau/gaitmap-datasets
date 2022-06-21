import copy
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd

from mad_datasets.utils.data_loading import load_bin_file

COORDINATE_SYSTEM_TRANSFORMATION = (  # egait_lateral_shimmer2r
    {
        # [[-y -> +x], [+z -> +y], [-x -> +z]]
        "left_sensor": [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        # [[+y -> +x], [-z -> +y], [-x -> +z]]
        "right_sensor": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    },
)

SHIMMER2_DATA_LAYOUT = {
    "acc_x": np.uint16,
    "acc_y": np.uint16,
    "acc_z": np.uint16,
    "gyr_x": np.uint16,
    "gyr_y": np.uint16,
    "gyr_z": np.uint16,
}


def rename_sh2_axes(
    dataset: Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Transform shimmer2 axes to align acc and gyroscope.

    Parameters
    ----------
    dataset
        A "raw" shimmer 2 dataset.

    Returns
    -------
    dataset
        Dataset with renamed and transformed axis.

    """
    # we need to handle the gyro-coordinate system separately because it is left-handed and rotated against the
    # acc-coordinate system
    for sensor in ["left_sensor", "right_sensor"]:
        # Ensure that we have a proper dtype and not uint from loading
        dataset[sensor] = dataset[sensor].astype(float)
        gyr_x_original = copy.deepcopy(dataset[sensor]["gyr_x"])
        gyr_y_original = copy.deepcopy(dataset[sensor]["gyr_y"])
        gyr_z_original = copy.deepcopy(dataset[sensor]["gyr_z"])

        dataset[sensor]["gyr_x"] = gyr_y_original
        dataset[sensor]["gyr_y"] = gyr_x_original
        dataset[sensor]["gyr_z"] = -gyr_z_original

    return dataset


def load_shimmer2_data(
    left_sensor_path: Path, right_sensor_path: Path
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Load shimmer2 data from a file."""

    data = {
        "left_sensor": load_bin_file(left_sensor_path, SHIMMER2_DATA_LAYOUT),
        "right_sensor": load_bin_file(right_sensor_path, SHIMMER2_DATA_LAYOUT),
    }
    data = rename_sh2_axes(data)
    return data
