import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from nilspodlib import SyncedSession
from scipy.spatial.transform import Rotation

from mad_datasets.stair_ambulation_healthy_2021.pressure_sensor_helper import calibrate_analog_data
from mad_datasets.utils.coordinate_transforms import rotate_dataset

COORDINATE_SYSTEM_TRANSFORMATION = {  # stair_ambulation_instep_nilspodv2
    # [[-y -> +x], [+x -> +y], [+z -> +z]]
    "left_sensor": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    # [[+y -> +x], [-x -> +y], [+z -> +z]]
    "right_sensor": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    # [[-z -> +x], [+x -> +y], [-y -> +z]]
    "hip_sensor": [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
}


def _participant_subfolder(base_dir: Path) -> Path:
    return base_dir / "healthy"


def _calibration_folder(base_dir: Path) -> Path:
    return base_dir / "calibrations"


def get_all_participants(*, base_dir: Optional[Path] = None) -> List[str]:
    # TODO: Rename healthy folder
    return [f.name for f in _participant_subfolder(base_dir).glob("subject_*")]


@lru_cache(maxsize=1)
def get_all_participants_and_tests(*, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    all_test_list = _participant_subfolder(base_dir).rglob("test_list.json")
    all_test_per_participant = {}

    for test_list in all_test_list:
        participant = test_list.parent.parent.name
        with open(test_list, "r") as f:
            test_data = json.load(f)
        test_data.pop("full_session", None)
        test_data.pop("part_1_transition", None)
        # Add the part the test belongs to the test_data
        part = int(test_list.parent.name.split("_")[-1])
        for test_info in test_data.values():
            test_info["part"] = int(part)

        tmp = all_test_per_participant.setdefault(participant, {})
        tmp.update(test_data)

    return all_test_per_participant


def get_participant_metadata(participant_folder_name: str, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    with open((_participant_subfolder(base_dir) / participant_folder_name / "metadata.json")) as f:
        metadata = json.load(f)
    return metadata


def get_all_data_for_participant(
    participant_folder_name: str,
    part: Literal[1, 2],
    return_pressure_data: bool = True,
    *,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:

    data_dir = _participant_subfolder(base_dir) / participant_folder_name / f"part_{part}" / "imu"
    session = SyncedSession.from_folder_path(data_dir, legacy_support="resolve")
    try:
        session = session.align_to_syncregion()
    except ValueError as e:
        print("Sync Warning: {e}" % e, file=sys.stderr)
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
    session_df = rotate_dataset(
        session_df, {k: Rotation.from_matrix(v) for k, v in COORDINATE_SYSTEM_TRANSFORMATION.items()}
    )

    if return_pressure_data is False:
        session_df = session_df.drop(columns=["analog_0", "analog_1", "analog_2"], level=1)
        return session_df
    # calibrate analog sensors
    all_calibrated_pressure_data = {}
    for s in ["left_sensor", "right_sensor"]:
        calibrated_data = calibrate_analog_data(
            session_df[s][["analog_0", "analog_1", "analog_2"]].to_numpy(), metadata["fsr_ids"][s], base_dir=base_dir
        )
        calibrated_data = pd.DataFrame(
            calibrated_data, columns=[f"{p}_force" for p in metadata["fsr_ids"][s].keys()]
        ).assign(total_force=lambda df_: df_.sum(axis=1))
        all_calibrated_pressure_data[s] = calibrated_data

    all_calibrated_pressure_data = pd.concat(all_calibrated_pressure_data, axis=1)
    session_df = session_df.join(all_calibrated_pressure_data)

    session_df = session_df.drop(columns=["analog_0", "analog_1", "analog_2"], level=1)
    return session_df
