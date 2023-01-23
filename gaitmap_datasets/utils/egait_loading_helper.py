"""Helper to load data of the egait system (specifically the shimmer 2R system)."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from imucal import FerrarisCalibrationInfo

from gaitmap_datasets.utils.data_loading import load_bin_file

SHIMMER2_DATA_LAYOUT = {
    "acc_x": np.uint16,
    "acc_y": np.uint16,
    "acc_z": np.uint16,
    "gyr_x": np.uint16,
    "gyr_y": np.uint16,
    "gyr_z": np.uint16,
}


def transform_shimmer2_axes(dataset: pd.DataFrame) -> pd.DataFrame:
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
    # Ensure that we have a proper dtype and not uint from loading
    dataset = dataset.astype(float)
    gyr_x_original = copy.deepcopy(dataset["gyr_x"])
    gyr_y_original = copy.deepcopy(dataset["gyr_y"])
    gyr_z_original = copy.deepcopy(dataset["gyr_z"])

    dataset["gyr_x"] = gyr_y_original
    dataset["gyr_y"] = gyr_x_original
    dataset["gyr_z"] = -gyr_z_original

    return dataset


def calibrate_shimmer2_data(
    data: pd.DataFrame,
    calibration_file_path: Path,
) -> pd.DataFrame:
    """Calibrate shimmer2 data."""
    if calibration_file_path.is_file():
        cal_matrix = load_compact_cal_matrix(calibration_file_path)
    elif calibration_file_path.is_dir():
        cal_matrix = load_extended_calib(calibration_file_path)
    else:
        raise ValueError("Calibration file does not exist.")
    data = cal_matrix.calibrate_df(data, "a.u.", "a.u.")

    return data


def load_shimmer2_data(
    data_path: Path,
    calibration_file_path: Path,
) -> pd.DataFrame:
    """Load shimmer2 data from a file."""
    data = load_bin_file(data_path, SHIMMER2_DATA_LAYOUT)
    data = calibrate_shimmer2_data(data, calibration_file_path)
    data = transform_shimmer2_axes(data)
    return data


def load_compact_cal_matrix(path: Path) -> FerrarisCalibrationInfo:
    """Load a compact calibration matrix from a file."""
    cal_matrix = np.genfromtxt(path, delimiter=",")
    plus_g = cal_matrix[0]
    minus_g = cal_matrix[1]
    b_a = (plus_g + minus_g) / 2
    K_a = np.eye(3) * (plus_g - minus_g) / 2  # pylint: disable=invalid-name
    K_a /= 9.81  # convert to m/s^2  # pylint: disable=invalid-name
    R_a = np.eye(3)  # pylint: disable=invalid-name
    b_g = cal_matrix[2]

    # 2.731 is the digital conversion factor for the gyro
    K_g = np.eye(3) * 2.731  # pylint: disable=invalid-name
    R_g = np.eye(3)  # pylint: disable=invalid-name
    K_ga = np.zeros((3, 3))  # pylint: disable=invalid-name

    return FerrarisCalibrationInfo(
        b_a=b_a, K_a=K_a, R_a=R_a, b_g=b_g, K_g=K_g, R_g=R_g, K_ga=K_ga, from_acc_unit="a.u.", from_gyr_unit="a.u."
    )


def load_extended_calib(calib_folder: Path) -> FerrarisCalibrationInfo:
    """Convert calibration files in format *_acc.csv and *_gyr.csv into a FerrarisCalibrationInfo object.

    This calibration format is used by later iterations of the egait system.
    It contains more information than the short calibration matrix and if data in this format is available, it should be
    preferred over the short calibration matrix.
    """
    # Each folder is a calibration

    acc_calib_path = next(calib_folder.glob("*_acc.csv"))
    gyr_calib_path = next(calib_folder.glob("*_gyro.csv"))
    acc_cal_phct = pd.read_csv(acc_calib_path, header=None)
    gyr_cal_phct = pd.read_csv(gyr_calib_path, header=None)
    imucal_cal = {
        "K_a": acc_cal_phct[[4, 5, 6]].to_numpy() / 9.81,
        "R_a": acc_cal_phct[[1, 2, 3]].to_numpy(),
        "b_a": acc_cal_phct[0].to_numpy(),
        "K_g": gyr_cal_phct[[4, 5, 6]].to_numpy(),
        "R_g": gyr_cal_phct[[1, 2, 3]].to_numpy(),
        "K_ga": np.zeros((3, 3)),
        "b_g": gyr_cal_phct[0].to_numpy(),
        "acc_unit": "m/s^2",
        "gyr_unit": "deg/s",
        "from_acc_unit": "a.u.",
        "from_gyr_unit": "a.u.",
    }
    imucal_cal = FerrarisCalibrationInfo(
        **imucal_cal, comment=f"Date: {calib_folder.name}, Sensor Node: " f"{acc_calib_path.stem.split('_')[0]}"
    )
    return imucal_cal
