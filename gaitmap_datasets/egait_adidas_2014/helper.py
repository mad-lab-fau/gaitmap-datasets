"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""

from itertools import product
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import pandas as pd

from gaitmap_datasets.utils.c3d import load_c3d_data


COORDINATE_SYSTEM_TRANSFORMATION_SH2R = {  # egait_lateral_shimmer2r
    # [[-y -> +x], [+z -> +y], [-x -> +z]]
    "left_sensor": [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    # [[+y -> +x], [-z -> +y], [-x -> +z]]
    "right_sensor": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
}

COORDINATE_SYSTEM_TRANSFORMATION_SH3 = (  # egait_lateral_shimmer3
    {
        # [[-x -> +x], [-z -> +y], [-y -> +z]]
        "left_sensor": [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        # [[+x -> +x], [+z -> +y], [-y -> +z]]
        "right_sensor": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    },
)

_SENSOR_SHORTHANDS = {"shimmer2r": "sh2", "shimmer3": "sh3"}

_MARKER_NAMES = ["".join(a) for a in product(["r_", "l_"], ["to_l", "to_m", "to_2", "cal_m", "cal_l", "hee"])]
# TODO: Add angle names
_ANGLE_NAMES = ["".join(a) + "footangles" for a in product(["l", "r"], ["rear", "fore"])]
_SENSOR_NAMES = Literal["shimmer2r", "shimmer3"]


# TODO: Change once decided on the final folder structure
def _raw_data_folder(base_dir: Path) -> Path:
    return base_dir / "data"


def get_all_participants(*, base_dir: Optional[Path] = None) -> List[str]:
    """Get the folder names of all participants."""
    return list(set(f.name.split("_")[0] for f in _raw_data_folder(base_dir).glob("*")))


class MetaDataRecord(TypedDict):
    participant: str
    test: str
    sensor: _SENSOR_NAMES


def get_all_participants_and_tests(*, base_dir: Optional[Path] = None) -> List[MetaDataRecord]:
    all_values = []
    for f in _raw_data_folder(base_dir).glob("*"):
        participant_id, test, sensor = f.name.split("_")
        all_values.append(MetaDataRecord(participant=participant_id, test=test, sensor=sensor))
    return all_values


def get_data_folder(participant: str, test: str, sensor: _SENSOR_NAMES, *, base_dir: Optional[Path] = None) -> Path:
    base_folder = _raw_data_folder(base_dir)
    return base_folder / f"{participant}_{test}_{_SENSOR_SHORTHANDS[sensor]}"


def get_mocap_data_for_participant_and_test(
    participant: str, test: str, sensor: _SENSOR_NAMES, *, base_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get all mocap data for a participant."""
    mocap_data = load_c3d_data(
        get_data_folder(participant, test, sensor, base_dir=base_dir) / f"{participant}_{test}.c3d"
    )
    marker_data = mocap_data[_MARKER_NAMES].copy()
    angle_data = mocap_data[_ANGLE_NAMES].copy()
    return marker_data, angle_data


def get_all_data_for_participant_and_test(
    participant: str, test: str, sensor: _SENSOR_NAMES, *, base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Get all data for a participant."""
    mocap_data = load_c3d_data(
        get_data_folder(participant, test, sensor, base_dir=base_dir) / f"{participant}-{test}.c3d"
    )
    return mocap_data
