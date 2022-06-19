"""Helper to rotate sensor data."""

from typing import Dict, Optional, Union

import pandas as pd
from scipy.spatial.transform import Rotation

from mad_datasets.utils.consts import SF_ACC, SF_GYR


def _rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
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
        test = _rotate_sensor(dataset[key], rotation_dict[key], inplace=False)
        rotated_dataset[key] = test

    # Restore original order
    rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset
