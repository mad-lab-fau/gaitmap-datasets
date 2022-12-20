"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from imucal.management import CalibrationWarning
from nilspodlib import SyncedSession
from nilspodlib.exceptions import SynchronisationError, SynchronisationWarning
from scipy.spatial.transform import Rotation

from gaitmap_datasets.stair_ambulation_healthy_2021.pressure_sensor_helper import calibrate_analog_data
from gaitmap_datasets.utils.coordinate_transforms import flip_dataset

COORDINATE_SYSTEM_TRANSFORMATION = {  # stair_ambulation_instep_nilspodv2
    # [[-y -> +x], [+x -> +y], [+z -> +z]]
    "left_sensor": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    # [[+y -> +x], [-x -> +y], [+z -> +z]]
    "right_sensor": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    # [[-z -> +x], [+x -> +y], [-y -> +z]]
    "hip_sensor": [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
}

StrideTypes = Literal["level", "ascending", "descending", "slope_ascending", "slope_descending"]


def _participant_subfolder(base_dir: Path) -> Path:
    """Return the relative path to the participant subfolder."""
    return base_dir / "healthy"


def _calibration_folder(base_dir: Path) -> Path:
    """Return the relative path to the imu-calibration subfolder."""
    return base_dir / "calibrations"


def get_all_participants(*, base_dir: Path) -> List[str]:
    """Get the folder names of all participants."""
    # TODO: Rename healthy folder
    return [f.name for f in _participant_subfolder(base_dir).glob("subject_*")]


@lru_cache(maxsize=1)
def get_all_participants_and_tests(
    *, base_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Dict[str, Union[int, Literal["part_1", "part_2"]]]]]:
    """Get a dictionary containing all test information for all participants."""
    all_test_list = _participant_subfolder(base_dir).rglob("test_list.json")
    all_test_per_participant = {}

    for test_list in all_test_list:
        participant = test_list.parent.parent.name
        with open(test_list, "r", encoding="utf8") as f:
            test_data = json.load(f)
        test_data.pop("part_1_transition", None)
        part = test_list.parent.name
        # Avoid name clash of the "full_session" test
        test_data[f"full_session_{part}"] = test_data.pop("full_session")
        # Add the part the test belongs to the test_data
        for test_info in test_data.values():
            test_info["part"] = part

        tmp = all_test_per_participant.setdefault(participant, {})
        tmp.update(test_data)

    return all_test_per_participant


def get_participant_metadata(participant_folder_name: str, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get the metadata of a participant."""
    with open((_participant_subfolder(base_dir) / participant_folder_name / "metadata.json"), encoding="utf8") as f:
        metadata = json.load(f)
    return metadata


def get_all_data_for_participant(
    participant_folder_name: str,
    part: Literal["part_1", "part_2"],
    *,
    return_pressure_data: bool = True,
    return_baro_data: bool = True,
    return_hip_sensor: bool = True,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Get all the recorded data (imu + baro + pressure) for one of the two sessions of a participant."""
    data_dir = _participant_subfolder(base_dir) / participant_folder_name / part / "imu"
    session = SyncedSession.from_folder_path(data_dir, legacy_support="resolve")
    # Ignore the nilspodlib warnings about syncpackages and calibration dates
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=CalibrationWarning)
        warnings.filterwarnings("ignore", category=SynchronisationWarning)
        try:
            session = session.align_to_syncregion()
        except SynchronisationError:
            # This is a NilsPod bug that happens sometimes.
            # In this case the index of the last couple of values is broken.
            # Therefore, we simply remove them.
            session = session.cut(stop=-10)
            session = session.align_to_syncregion()
    # apply ferraris calibration on imu data
    session.calibrate_imu(
        session.find_closest_calibration(folder=_calibration_folder(base_dir), filter_cal_type="ferraris"), inplace=True
    )
    session_df = session.data_as_df(concat_df=True)

    # load_metadata to rename columns
    metadata = get_participant_metadata(participant_folder_name, base_dir=base_dir)
    sensor_mapping = {v: k for k, v in metadata["sensor_ids"].items()}
    # Rename all columns
    session_df = session_df.rename(columns=sensor_mapping)
    # Rotate the data to the correct coordinate system
    session_df = flip_dataset(
        session_df, {k: Rotation.from_matrix(v) for k, v in COORDINATE_SYSTEM_TRANSFORMATION.items()}
    )

    if return_pressure_data is True:
        # calibrate analog sensors
        all_calibrated_pressure_data = {}
        for s in ["left_sensor", "right_sensor"]:
            calibrated_data = calibrate_analog_data(
                session_df[s][["analog_0", "analog_1", "analog_2"]].to_numpy(),
                metadata["fsr_ids"][s],
                base_dir=base_dir,
            )
            calibrated_data = pd.DataFrame(
                calibrated_data, columns=[f"{p}_force" for p in metadata["fsr_ids"][s].keys()]
            ).assign(total_force=lambda df_: df_.sum(axis=1))
            all_calibrated_pressure_data[s] = calibrated_data

        all_calibrated_pressure_data = pd.concat(all_calibrated_pressure_data, axis=1)
        session_df = session_df.join(all_calibrated_pressure_data)

    session_df = session_df.drop(columns=["analog_0", "analog_1", "analog_2"], level=1)
    if return_baro_data is False:
        session_df = session_df.drop(columns=["baro", "baro"], level=1)
    if return_hip_sensor is False:
        session_df = session_df.drop(columns="hip_sensor", level=0)

    return session_df


def get_segmented_stride_list(
    participant_folder_name: str, part: Literal["part_1", "part_2"], *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the manual stride borders for a participant."""
    path = _participant_subfolder(base_dir) / participant_folder_name / part / "manual_annotations_z_level.csv"
    manual_annotation = pd.read_csv(path, delimiter=";", index_col=0, header=[0, 1])[["left_sensor", "right_sensor"]]
    # We concat the two feet along the other axis to get a unique stride id via the index
    manual_annotation = (
        manual_annotation.stack(level=0)
        .reset_index(level=0, drop=True)
        .sort_values("start")
        .reset_index()
        .set_index("index", append=True)
        .swaplevel()
    )
    manual_annotation.index.names = ["sensor", "s_id"]
    return {
        sensor: manual_annotation.loc[sensor].dropna()[["start", "end", "type", "z_level"]]
        for sensor in ["left_sensor", "right_sensor"]
    }


def get_pressure_insole_events(
    participant_folder_name: str, part: Literal["part_1", "part_2"], *, base_dir: Optional[Path] = None
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the pressure insole events for a participant."""
    path = _participant_subfolder(base_dir) / participant_folder_name / part / "pressure_events.csv"
    events = pd.read_csv(path, index_col=[0, 1], header=0)
    return {sensor: events.loc[sensor] for sensor in ["left_sensor", "right_sensor"]}
