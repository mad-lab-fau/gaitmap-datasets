"""Helper to rotate sensor data."""

from typing import Dict, Literal, Optional, Union

import pandas as pd
from scipy.spatial.transform import Rotation

from mad_datasets.utils.consts import (
    BF_COLS,
    FSF_FBF_CONVERSION_LEFT,
    FSF_FBF_CONVERSION_RIGHT,
    SF_ACC,
    SF_COLS,
    SF_GYR,
)


def rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(dataset: pd.DataFrame, rotation: Union[Rotation, Dict[str, Rotation]]) -> pd.DataFrame:
    """Apply a rotation to acc and gyro data of a dataset.

    Parameters
    ----------
    dataset
        dataframe representing a multiple synchronised sensors.
    rotation
        In case a single rotation object is passed, it will be applied to all sensors of the dataset.
        If a dictionary of rotations is applied, the respective rotations will be matched to the sensors based on the
        dict keys.
        If no rotation is provided for a sensor, it will not be modified.

    Returns
    -------
    rotated dataset
        This will always be a copy. The original dataframe will not be modified.

    """
    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in dataset.columns.unique(level=0)}

    rotated_dataset = dataset.copy()
    original_cols = dataset.columns

    for key in rotation_dict.keys():
        test = rotate_sensor(dataset[key], rotation_dict[key], inplace=False)
        rotated_dataset[key] = test

    # Restore original order
    rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset


def convert_left_foot_to_fbf(data: pd.DataFrame):
    """Convert the axes from the left foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    """
    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[FSF_FBF_CONVERSION_LEFT[sf_col_name][1]] = FSF_FBF_CONVERSION_LEFT[sf_col_name][0] * data[sf_col_name]

    return result


def convert_right_foot_to_fbf(data: pd.DataFrame):
    """Convert the axes from the right foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    """
    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[FSF_FBF_CONVERSION_RIGHT[sf_col_name][1]] = FSF_FBF_CONVERSION_RIGHT[sf_col_name][0] * data[sf_col_name]

    return result


def convert_to_fbf(data: Union[pd.DataFrame, Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]]):
    """Convert the axes from the left foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    """
    result = {
        "left_sensor": convert_left_foot_to_fbf(data["left_sensor"]),
        "right_sensor": convert_right_foot_to_fbf(data["right_sensor"]),
    }
    if isinstance(data, pd.DataFrame):
        return pd.concat(result, axis=1)
    return result
