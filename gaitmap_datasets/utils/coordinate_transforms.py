"""Helper to rotate sensor data."""

from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap_datasets.utils.consts import (
    BF_COLS,
    FSF_FBF_CONVERSION_LEFT,
    FSF_FBF_CONVERSION_RIGHT,
    SF_ACC,
    SF_COLS,
    SF_GYR,
)


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def flip_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Flip (same as rotate, but only 90 deg rots allowed) the data of a single sensor.

    Compared to normal rotations, this function can result in massive speedups!
    """
    if rotation is None:
        return data
    if rotation.single is False:
        raise ValueError("Only single rotations are allowed!")

    tol = 10e-9
    rot_matrix = rotation.as_matrix().squeeze()
    all_1 = np.allclose(np.abs(rot_matrix[~np.isclose(rot_matrix, 0, atol=tol)]).flatten(), 1, atol=tol)
    if not all_1:
        raise ValueError(
            "Only 90 deg rotations are allowed (i.e. 1 and -1 in the rotation matrix)! "
            f"The current matrix is:\n\n {rot_matrix}"
        )

    # Now that we know the rotation is valid, we round the values to make all further checks simpler
    rot_matrix = np.round(rot_matrix)

    if inplace is False:
        data = data.copy()

    orig_col_order = data.columns
    for sensor in ["acc", "gyr"]:
        cols = np.array({"acc": SF_ACC, "gyr": SF_GYR}[sensor])
        rename = {}
        mirror = []
        # We basically iterate over the rotation matrix and find which axis is transformed to which other axis.
        # If the entry is -1, we also mirror the axis.
        for col, row in zip(cols, rot_matrix):
            old_index = cols[np.abs(row).astype(bool)][0]
            rename[old_index] = col
            if np.sum(row) == -1:
                mirror.append(col)
        data = data.rename(columns=rename)
        data[mirror] *= -1
    data = data[orig_col_order]
    return data


def rotate_sensor(data: pd.DataFrame, rotation: Optional[Rotation], inplace: bool = False) -> pd.DataFrame:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def flip_dataset(dataset: pd.DataFrame, rotation: Union[Rotation, Dict[str, Rotation]]) -> pd.DataFrame:
    """Flip datasets (same as rotate, but only 90 deg rots allowed).

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
    flipped dataset
        This will always be a copy. The original dataframe will not be modified.

    """
    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in dataset.columns.unique(level=0)}

    rotated_dataset = dataset.copy()
    original_cols = dataset.columns

    for key in rotation_dict.keys():
        rotated_dataset[key] = flip_sensor(dataset[key], rotation_dict[key], inplace=False)

    # Restore original order
    rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset


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
        rotated_dataset[key] = rotate_sensor(dataset[key], rotation_dict[key], inplace=False)

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
