"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap_datasets.utils.coordinate_transforms import flip_sensor
from gaitmap_datasets.utils.egait_loading_helper import load_shimmer2_data

Cohorts = Literal["control", "pd", "geriatric"]
Tests = Literal["4x10m", "free_walk"]

_test_rename_dict = {"4x10m": "", "free_walk": "_4MW"}
_cohort_rename_dict = {"control": "Control", "pd": "PD", "geriatric": "Geriatric"}

COORDINATE_SYSTEM_TRANSFORMATION = {  # egait_lateral_shimmer2r
    # [[-y -> +x], [+z -> +y], [-x -> +z]]
    "left_sensor": [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    # [[+y -> +x], [-z -> +y], [-x -> +z]]
    "right_sensor": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
}


def _raw_data_folder(base_dir: Path, cohort: Cohorts, test: Tests) -> Path:
    """Return the relative path to the participant subfolder."""
    return base_dir / f"{_cohort_rename_dict[cohort]}s_RawDataValidation{_test_rename_dict[test]}"


def _reference_stride_borders_folder(base_dir: Path, cohort: Cohorts, test: Tests) -> Path:
    """Return the relative path to the reference stride borders subfolder."""
    return base_dir / f"{_cohort_rename_dict[cohort]}s_GoldStandard_StrideBorders{_test_rename_dict[test]}"


def _calibration_folder(base_dir: Path) -> Path:
    """Return the relative path to the imu-calibration subfolder."""
    return base_dir / "calibrationFiles"


def _get_sensor_node(stride_segmentation_file_path: Path) -> str:
    """Get the sensor node id by reading the respective entry in the file."""
    # The information is stored in the 4th line of the file.
    with stride_segmentation_file_path.open() as f:
        return f.readlines()[3].split(",")[1].strip().upper()


def _extract_participant_id(file_name: str, test: Tests) -> str:
    """Extract the participant id from the file name."""
    if test == "4x10m":
        return file_name.split("_")[0].upper()
    if test == "free_walk":
        for foot in ["left", "right"]:
            if f"{foot}foot" in file_name.lower():
                return file_name.lower().split(f"{foot}foot")[0].upper()
    raise ValueError("Invalid file format or test name")


def _find_files_for_participant(
    directory: Path, participant_id: str
) -> Dict[Literal["left_sensor", "right_sensor"], Optional[Path]]:
    """Find the file for a participant."""
    participant_id = participant_id.lower()
    files = {"left_sensor": None, "right_sensor": None}
    for file in directory.iterdir():
        if file.is_file():
            file_name = file.name.lower()
            if participant_id in file_name:
                if "left" in file_name:
                    files["left_sensor"] = file
                elif "right" in file_name:
                    files["right_sensor"] = file
        if all(files.values()):
            break
    return files


def get_all_participants(*, base_dir: Optional[Path] = None) -> List[Tuple[str, str, str]]:
    """Get the folder names of all participants."""
    all_participants = []
    for cohort in ["control", "pd", "geriatric"]:
        for test in ["4x10m", "free_walk"]:
            stride_border_folder = _reference_stride_borders_folder(base_dir, cohort, test)
            for file_name in stride_border_folder.glob("*.txt"):
                if file_name.is_file():
                    participant_id = _extract_participant_id(file_name.name, test)
                    all_participants.append((cohort, test, participant_id))
    return list(set(all_participants))


def get_all_data_for_participant(
    participant_id: str, cohort: Cohorts, test: Tests, *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the data for a participant."""
    raw_data_folder = _raw_data_folder(base_dir, cohort, test)
    files = _find_files_for_participant(raw_data_folder, participant_id)
    # We need to load the stride segmentation data as well, because it contains the information about the calibration
    # files.
    stride_segmentation_folder = _reference_stride_borders_folder(base_dir, cohort, test)

    all_data = {}
    for sensor, file_path in files.items():
        if file_path is None:
            all_data[sensor] = None
            continue
        stride_segmentation_file = stride_segmentation_folder / file_path.with_suffix(".txt").name
        calibration_file_path = _calibration_folder(base_dir) / f"{_get_sensor_node(stride_segmentation_file)}.csv"
        data = load_shimmer2_data(file_path, calibration_file_path)
        data = flip_sensor(data, Rotation.from_matrix(COORDINATE_SYSTEM_TRANSFORMATION[sensor]))
        all_data[sensor] = data
    return all_data


def get_segmented_stride_list(
    participant_id: str, cohort: Cohorts, test: Tests, *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the list of all strides for a participant."""
    stride_borders = {}
    stride_segmentation_folder = _reference_stride_borders_folder(base_dir, cohort, test)
    segmentation_files = _find_files_for_participant(stride_segmentation_folder, participant_id)
    for sensor, file_path in segmentation_files.items():
        if file_path is None:
            stride_borders[sensor] = None
            continue
        stride_borders[sensor] = (
            pd.read_csv(
                file_path,
                skiprows=8,
                header=0,
            )
            .rename(columns={"Start": "start", "End": "end"})
            .rename_axis(index="s_id")
        )
    return stride_borders
