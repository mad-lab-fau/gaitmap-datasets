"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""

import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from scipy.spatial.transform import Rotation

from mad_datasets.utils.coordinate_transforms import flip_sensor
from mad_datasets.utils.egait_loading_helper import load_shimmer2_data

CALIBRATION_FILE_NAMES = {
    "left_sensor": "A917.csv",
    "right_sensor": "A6DF.csv",
}

COORDINATE_SYSTEM_TRANSFORMATION = {  # egait_lateral_shimmer2r
    # [[-y -> +x], [+z -> +y], [-x -> +z]]
    "left_sensor": [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    # [[+y -> +x], [-z -> +y], [-x -> +z]]
    "right_sensor": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
}


def _raw_data_folder(base_dir: Path) -> Path:
    """Return the relative path to the participant subfolder."""
    return base_dir / "ValidationRawData"


def _reference_stride_borders_folder(base_dir: Path) -> Path:
    """Return the relative path to the reference stride borders subfolder."""
    return base_dir / "GoldStandard_StrideBorders"


def _reference_stride_parameters_folder(base_dir: Path) -> Path:
    """Return the relative path to the reference stride parameters subfolder."""
    return base_dir / "GoldStandard_GaitRite"


def _calibration_folder(base_dir: Path) -> Path:
    """Return the relative path to the imu-calibration subfolder."""
    return base_dir


def get_all_participants(*, base_dir: Optional[Path] = None) -> List[str]:
    """Get the folder names of all participants."""
    return [f.name.split("_")[0] for f in _raw_data_folder(base_dir).glob("*_left.dat")]


def get_all_data_for_participant(
    participant_id: str, *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get all data for a participant."""
    all_data = {}
    for foot in ["left", "right"]:
        sensor = foot + "_sensor"
        data_path = _raw_data_folder(base_dir) / f"{participant_id}_E4_{foot}.dat"
        calibration_path = _calibration_folder(base_dir) / CALIBRATION_FILE_NAMES[sensor]
        data = load_shimmer2_data(data_path, calibration_path)
        data = flip_sensor(data, Rotation.from_matrix(COORDINATE_SYSTEM_TRANSFORMATION[sensor]))
        all_data[sensor] = data
    return all_data


def get_segmented_stride_list(
    participant_id: str, *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the list of all strides for a participant."""
    stride_borders = {}
    for foot in ["left", "right"]:
        stride_borders[f"{foot}_sensor"] = (
            pd.read_csv(
                _reference_stride_borders_folder(base_dir) / f"{participant_id}_E4_{foot}.txt", skiprows=8, header=0
            )
            .rename(columns={"Sart": "start", "Start": "start", "End": "end"})
            .rename_axis(index="s_id")
        )
    return stride_borders


def get_gaitrite_parameters(
    participant_id: str, *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the gaitrite parameters for a participant."""
    parameters = {}
    for foot in ["left", "right"]:
        parameters[f"{foot}_sensor"] = (
            pd.read_csv(
                _reference_stride_parameters_folder(base_dir) / f"{participant_id}_E4_{foot}.txt", skiprows=8, header=0
            )
            .rename(columns=lambda name: re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())
            .rename_axis(index="s_id")
            .assign(stride_length=lambda df_: df_["stride_length"] / 100.0)  # Convert stride length to meters
        )
    return parameters
