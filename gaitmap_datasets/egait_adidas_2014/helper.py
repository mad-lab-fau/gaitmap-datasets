"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap_datasets.utils.c3d_loading import load_c3d_data
from gaitmap_datasets.utils.coordinate_transforms import flip_sensor
from gaitmap_datasets.utils.egait_loading_helper import find_extended_calib_files, load_shimmer2_data

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
_SENSOR_SHORTHANDS_REVERSE = {v: k for k, v in _SENSOR_SHORTHANDS.items()}

_MARKER_NAMES = ["".join(a) for a in product(["r_", "l_"], ["to_l", "to_m", "to_2", "cal_m", "cal_l", "hee"])]
_MARKER_RENAMES = {"hee": "heel", "to_2": "toe_2", "to_m": "toe_m", "to_l": "toe_l"}
# TODO: Add angle names
_ANGLE_NAMES = ["".join(a) + "footangles" for a in product(["l", "r"], ["rear", "fore"])]
_SENSOR_NAMES = Literal["shimmer2r", "shimmer3"]


# TODO: Change once decided on the final folder structure
def _raw_data_folder(base_dir: Path) -> Path:
    return base_dir / "data"


def _calibration_folder(base_dir: Path) -> Path:
    return base_dir / "calibrations"


class MetaDataRecord(TypedDict):
    participant: str
    stride_length: Literal["low", "normal", "high"]
    stride_velocity: Literal["low", "normal", "high"]
    repetition: Literal["1", "2", "3"]
    sensor: _SENSOR_NAMES


def get_all_participants_and_tests(*, base_dir: Optional[Path] = None) -> List[MetaDataRecord]:
    all_values = []
    for f in sorted(_raw_data_folder(base_dir).rglob("*.c3d")):
        participant_id, sensor, stride_length, stride_velocity, repetition = f.name.split("_")
        all_values.append(
            MetaDataRecord(
                participant=participant_id[4:],
                stride_velocity=stride_velocity,
                stride_length=stride_length,
                repetition=str(int(repetition.split(".")[0])),
                sensor=_SENSOR_SHORTHANDS_REVERSE[sensor],
            )
        )
    return all_values


def get_data_folder(
    participant: str,
    sensor: _SENSOR_NAMES,
    test_postfix: str,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    base_folder = _raw_data_folder(base_dir)
    return base_folder / f"subj{participant}" / _SENSOR_SHORTHANDS[sensor] / test_postfix


def get_test_postfix(stride_length: str, stride_velocity: str, repetition: str) -> str:
    return f"{stride_length}_{stride_velocity}_0{repetition}"


def _transform_marker_names(name: str, coord: str):
    foot, marker = name.split("_", 1)
    foot_side = "left" if foot == "l" else "right"
    marker = _MARKER_RENAMES.get(marker, marker)
    return foot_side + "_sensor", f"{marker}_{coord}"


def get_mocap_data_for_participant_and_test(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    *,
    base_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get all mocap data for a participant."""
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    mocap_data = load_c3d_data(
        get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)
        / f"subj{participant}_{_SENSOR_SHORTHANDS[sensor]}_{test_postfix}.c3d"
    )
    marker_data = mocap_data[_MARKER_NAMES].copy()
    marker_data.columns = pd.MultiIndex.from_tuples(_transform_marker_names(*name) for name in marker_data.columns)
    angle_data = mocap_data[_ANGLE_NAMES].copy()
    return marker_data, angle_data


def get_all_data_for_participant_and_test(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    *,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Get all data for a participant."""
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    data_folder = get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)
    all_data = {}
    for foot in ["left", "right"]:
        foot_sensor = foot + "_sensor"
        try:
            dat_file = next(data_folder.glob(f"*_{foot}_data.dat"))
        except StopIteration:
            # No data for this foot
            continue
        sensor_id = dat_file.name.split("_")[5]
        calibration_path = find_extended_calib_files(
            _calibration_folder(base_dir),
            sensor_id,
        )
        if sensor == "shimmer2r":
            sensor_data = load_shimmer2_data(dat_file, calibration_path)
            sensor_data = flip_sensor(
                sensor_data, Rotation.from_matrix(COORDINATE_SYSTEM_TRANSFORMATION_SH2R[foot_sensor])
            )
        else:
            raise NotImplementedError("Only shimmer2r is supported")

        all_data[foot_sensor] = sensor_data
    assert len(all_data) > 0, "No data found for this participant test combi. That should not happen"
    if len(all_data) == 2:
        assert len(all_data["left_sensor"]) == len(all_data["right_sensor"]), (
            "Unequal data lengths. This should not " "happen, as data is synced."
        )
    return pd.concat(
        all_data,
        axis=1,
    )


def get_synced_stride_list(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    *,
    system: Literal["mocap", "imu"] = "imu",
    base_dir: Optional[Path] = None,
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get all data for a participant."""
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    data_folder = get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)
    all_strides = {}
    file_postfix = "ShimmerStrides.txt" if system == "imu" else "viconStrides.txt"

    for foot in ["left", "right"]:
        foot_sensor = foot + "_sensor"
        try:
            stride_borders_file = next(data_folder.glob(f"*_{foot}_{file_postfix}"))
        except StopIteration:
            # No data for this foot
            continue
        strides = (
            pd.read_csv(stride_borders_file, sep=",", skiprows=8, header=0)
            .rename(columns=lambda x: x.lower())
            .rename_axis("s_id")
        )
        if system == "mocap":
            # The stride border for the mocap system are (for some reason) also provided at 204.8 Hz
            # To have them at the same frequency as the mocap data, we need transform them to 200 Hz
            strides *= 200 / 204.8
        all_strides[foot_sensor] = strides

    assert len(all_strides) > 0, "No data found for this participant test combi. That should not happen"
    return all_strides


@lru_cache(maxsize=1)
def get_mocap_offset(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    /,
    imu_sampling_rate: float,
    mocap_sampling_rate: float,
    *,
    base_dir: Optional[Path] = None,
) -> float:
    """Get all data for a participant."""
    mocap_strides = get_synced_stride_list(
        participant, sensor, stride_length, stride_velocity, repetition, system="mocap", base_dir=base_dir
    )
    imu_strides = get_synced_stride_list(
        participant, sensor, stride_length, stride_velocity, repetition, system="imu", base_dir=base_dir
    )
    assert len(mocap_strides) == len(imu_strides), (
        "Data found for different feet. This should not happen.\n"
        f"{participant, sensor, stride_length, stride_velocity, repetition}"
    )
    offsets = []
    for foot in ["left_sensor", "right_sensor"]:
        try:
            m_strides = mocap_strides[foot] / mocap_sampling_rate
            i_strides = imu_strides[foot] / imu_sampling_rate
        except KeyError:
            # No data for this foot
            continue
        assert len(m_strides) == len(i_strides), (
            "Unequal number of strides. This should not "
            f"happen. \n ({participant, sensor, stride_length, stride_velocity, repetition, foot})"
        )

        offset = i_strides["start"] - m_strides["start"]
        assert len(set(offset)) == 1
        offsets.append(offset.iloc[0])
    assert set(offsets) == {offsets[0]}
    return offsets[0]
