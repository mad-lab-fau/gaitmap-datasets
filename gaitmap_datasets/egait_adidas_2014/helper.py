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
from gaitmap_datasets.utils.egait_loading_helper import (
    find_extended_calib_files,
    load_shimmer2_data,
    load_shimmer3_data,
)

COORDINATE_SYSTEM_TRANSFORMATION_SH2R = {  # egait_lateral_shimmer2r
    # [[-y -> +x], [+z -> +y], [-x -> +z]]
    "left_sensor": [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    # [[+y -> +x], [-z -> +y], [-x -> +z]]
    "right_sensor": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
}

COORDINATE_SYSTEM_TRANSFORMATION_SH3 = {  # egait_lateral_shimmer3
    # [[-x -> +x], [-z -> +y], [-y -> +z]]
    "left_sensor": [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
    # [[+x -> +x], [+z -> +y], [-y -> +z]]
    "right_sensor": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
}

_PARAMETER_RENAMES = {
    "StrideLength": "stride_length",
    "StrideTime": "stride_time",
    "StanceTime": "stance_time",
    "SwingTime": "swing_time",
    "maxTC": "max_toe_clearance",
    "maxHC": "max_heel_clearance",
    "heelStrikeAngle": "ic_angle",
    "toeOffAngle": "tc_angle",
    "maxLatSwing": "max_lateral_excursion",
}

_SENSOR_SHORTHANDS = {"shimmer2r": "sh2", "shimmer3": "sh3"}
_SENSOR_SHORTHANDS_REVERSE = {v: k for k, v in _SENSOR_SHORTHANDS.items()}

_MARKER_NAMES = ["".join(a) for a in product(["r_", "l_"], ["to_l", "to_m", "to_2", "cal_m", "cal_l", "hee"])]
_MARKER_RENAMES = {"hee": "heel", "to_2": "toe_2", "to_m": "toe_m", "to_l": "toe_l"}
_EVENT_RENAMES = {"MS": "min_vel", "HS": "ic", "TO": "tc"}
# TODO: Add angle names
_ANGLE_NAMES = ["".join(a) + "footangles" for a in product(["l", "r"], ["rear", "fore"])]
_SENSOR_NAMES = Literal["shimmer2r", "shimmer3"]  # pylint: disable=invalid-name


# TODO: Change once decided on the final folder structure
def _raw_data_folder(base_dir: Path) -> Path:
    return base_dir / "data"


def _calibration_folder(base_dir: Path) -> Path:
    return base_dir / "calibrations"


class MetaDataRecord(TypedDict):
    """Metadata record for a single test."""

    participant: str
    stride_length: Literal["low", "normal", "high"]
    stride_velocity: Literal["low", "normal", "high"]
    repetition: Literal["1", "2", "3"]
    sensor: _SENSOR_NAMES


def get_all_participants_and_tests(*, base_dir: Optional[Path] = None) -> List[MetaDataRecord]:
    """Get all participant and sensor/test combinations."""
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
    """Get the data folder for a given participant and sensor/test combination."""
    base_folder = _raw_data_folder(base_dir)
    return base_folder / f"subj{participant}" / _SENSOR_SHORTHANDS[sensor] / test_postfix


def get_test_postfix(stride_length: str, stride_velocity: str, repetition: str) -> str:
    """Get the postifix for a specific test, used in many file names."""
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
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Get all mocap data for a participant.

    The dataset contains marker data and angle data.
    We load both, but only use the marker data for now.

    Marker data is returned as a dictionary, even-though the marker data is synchronized, to have it in the same
    structure as the IMU data.
    """
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    mocap_data = load_c3d_data(
        get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)
        / f"subj{participant}_{_SENSOR_SHORTHANDS[sensor]}_{test_postfix}.c3d"
    )
    marker_data = mocap_data[_MARKER_NAMES].copy()
    marker_data.columns = pd.MultiIndex.from_tuples(_transform_marker_names(*name) for name in marker_data.columns)
    marker_data = {k: marker_data[k] for k in marker_data.columns.get_level_values(0)}
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
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get all IMU data for participant - test combination.

    The data is not synchronized, hence we return a dictionary.
    Note, that in many cases data from only one foot is available.
    """
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
            sensor_data = load_shimmer3_data(dat_file, calibration_path)
            sensor_data = flip_sensor(
                sensor_data, Rotation.from_matrix(COORDINATE_SYSTEM_TRANSFORMATION_SH3[foot_sensor])
            )

        all_data[foot_sensor] = sensor_data
    assert len(all_data) > 0, "No data found for this participant test combi. That should not happen"
    return all_data


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
    """Get the stride lists for one of the two systems.

    The dataset provides separate files for the IMU and the mocap stride borders.
    However, they are identical, just the samples are converted between the two systems.
    We only use the IMU stride list in the end, but we need to be able to load the mocap stride list to calculate the
    offset between the two systems.
    """
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

        # We runtime check that the end of each stride is the start of the next stride
        assert ((strides["start"] - strides["end"].shift(1)).dropna() == 0).all(), (
            "Stride ends don't match stride " "starts."
        )
        assert (strides["end"] - strides["start"]).min() > 0, "Stride borders are not sorted"
        all_strides[foot_sensor] = strides

    assert len(all_strides) > 0, "No data found for this participant test combi. That should not happen"
    return all_strides


@lru_cache(maxsize=1)
def get_mocap_offset_s(
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
) -> Dict[Literal["left_sensor", "right_sensor"], float]:
    """Get the offset between the mocap and the imu recording.

    This offset might be different for the two feet, as the IMU sensors don't perfectly start their recording at the
    same time.
    Hence, we return a dictionary with the offsets for each foot.
    """
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
    offsets = {}
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
        offsets[foot] = offset.iloc[0]
    return offsets


def get_mocap_parameters(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the stride parameters from the mocap system."""
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    data_folder = get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)
    all_stride_paras = {}

    for foot in ["left", "right"]:
        foot_sensor = foot + "_sensor"
        try:
            stride_para_file = next(data_folder.glob(f"*_{foot}_data_strideParameters.txt"))
        except StopIteration:
            # No data for this foot
            continue
        stride_paras = (
            pd.read_csv(stride_para_file, sep=",", skiprows=8, header=0)
            .rename(columns=_PARAMETER_RENAMES)
            .rename_axis("s_id")
        )
        # All cm values need to be converted to m
        stride_paras[["stride_length", "max_toe_clearance", "max_heel_clearance"]] /= 100
        all_stride_paras[foot_sensor] = stride_paras

    assert len(all_stride_paras) > 0, "No data found for this participant test combi. That should not happen"
    return all_stride_paras


def get_mocap_events(
    participant: str,
    sensor: _SENSOR_NAMES,
    stride_length: str,
    stride_velocity: str,
    repetition: str,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    """Get the events from the mocap system."""
    stride_list = get_synced_stride_list(
        participant, sensor, stride_length, stride_velocity, repetition, system="mocap", base_dir=base_dir
    )
    test_postfix = get_test_postfix(stride_length, stride_velocity, repetition)
    data_folder = get_data_folder(participant, sensor, test_postfix, base_dir=base_dir)

    final_stride_list = {}

    for sensor_name, strides in stride_list.items():
        foot = sensor_name.split("_")[0]
        # All events are relative to the start of the stride.
        # BUT, in the original dataset they accidentally interpreted the stride start with a sampling rate of 200 Hz
        # when extracting the events from the raw marker trajectory.
        # Hence, we need to correct for this.
        #
        # Note, that this did not impact the stride parameters provided with the dataset, as the calculation
        # error was kept consistent throughout the calculations.
        # However, to properly align the reported events with the mocap trajectories again, we need to correct
        # this offset.
        # We calculate it here and add it later.
        offset = strides["start"] - strides["start"] * 200 / 204.8
        events = {}
        for event in ("MS", "TO", "HS"):
            try:
                event_file = next(data_folder.glob(f"*_{foot}_data_{event}.txt"))
            except StopIteration:
                # No data for this foot
                continue
            event_data = pd.read_csv(event_file, sep=",", skiprows=8, header=None)[0]
            # The event data is also provided at 204.8 Hz
            # Hence, we convert it to the mocap sampling rate
            event_data *= 200 / 204.8
            events[_EVENT_RENAMES[event]] = strides["start"] + event_data - offset
        # We make some dummy checks:
        assert all(strides["tc"] < strides["ic"])
        assert all(strides["ic"] < strides["min_vel"])
        assert all(strides["min_vel"] < strides["end"])
        final_stride_list[sensor_name] = strides.assign(**events).round(0).astype(int)
    return final_stride_list
