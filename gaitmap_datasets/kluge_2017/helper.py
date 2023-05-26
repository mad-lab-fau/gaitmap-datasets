"""A set of helpers to load the dataset."""
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Sequence, Tuple, Union

import pandas as pd
import scipy.io as sio
from scipy.spatial.transform import Rotation

from gaitmap_datasets.utils.array_handling import bool_array_to_start_end_array
from gaitmap_datasets.utils.consts import SF_ACC
from gaitmap_datasets.utils.coordinate_transforms import flip_sensor

COORDINATE_SYSTEM_TRANSFORMATION_SH3 = {  # egait_lateral_shimmer3
    # [[-x -> +x], [-z -> +y], [-y -> +z]]
    "left_sensor": [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
    # [[+x -> +x], [+z -> +y], [-y -> +z]]
    "right_sensor": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
}

MOCAP_SAMPLING_RATE_HZ = 100.0
IMU_SAMPLING_RATE_HZ = 102.4


SENSOR_RENAMING_DICT = {
    "LeftFoot": "left_sensor",
    "RightFoot": "right_sensor",
}

SENSOR_AXIS_RENAMING_DICT = {
    "Time": "time after start [s]",
    "AccX": "acc_x",
    "AccY": "acc_y",
    "AccZ": "acc_z",
    "GyrX": "gyr_x",
    "GyrY": "gyr_y",
    "GyrZ": "gyr_z",
}


def _interleave_foot_events(starts: pd.Series, foot_names: Tuple[str, str] = ("L", "R")) -> pd.Series:
    """Map the hs of the opposite foot to the stride of the other foot it occurred in."""
    opposite_starts = {}

    foot_0_starts = starts.loc[foot_names[0]]
    foot_1_starts = starts.loc[foot_names[1]]

    for foot_name, foot, opposite_foot in zip(
        foot_names, (foot_0_starts, foot_1_starts), (foot_1_starts, foot_0_starts)
    ):
        if not foot.is_monotonic_increasing:
            raise ValueError(f"The {foot_name} foot events are not sorted by value!")

        # We find the insertion positions of the opposite foot events
        # In case we find multiple events that would be sorted to the same position, we will remove both, invalidating
        # the stride.
        # If an event was sorted to position 0, this means that it happened before the first start.
        # We remove that as well.
        # Afterwards, the stride the event belongs to is the stride that starts one before the insertion position.
        insertion_index_opposite = (
            pd.Series(foot.searchsorted(opposite_foot, side="left"))
            .drop_duplicates(keep=False)
            .where(lambda s: s > 0)
            .dropna()
            .astype(int)
        ) - 1

        opposite_start = pd.Series(index=foot.index, dtype=float)
        opposite_start.iloc[insertion_index_opposite] = opposite_foot.iloc[insertion_index_opposite.index]
        opposite_starts[foot_name] = opposite_start
    return pd.concat(opposite_starts).reindex(starts.index)


def intersect_data(data: pd.DataFrame, interval_starts: Sequence[float], interval_ends: Sequence[float]):
    """Set all datapoints to nan if they are not covered by any interval."""
    interval_per_data = pd.cut(data.index, pd.IntervalIndex.from_arrays(interval_starts, interval_ends))
    data = data.copy()
    data.loc[interval_per_data.isna()] = pd.NA
    return data


def intersect_strides(strides: pd.DataFrame, interval_starts: Sequence[float], interval_ends: Sequence[float]):
    """Remove all rows from events where not all events are covered by the same interval.

    Note: we assume that the events only have a single level column index and the returned events might have changed
    order.
    """
    interval_per_event = (
        pd.cut(  # noqa: PD010
            strides[["start", "end"]].stack(), pd.IntervalIndex.from_arrays(interval_starts, interval_ends)
        )
        .unstack()
        .T
    )
    # Check if all columns are in the same interval
    return strides[((interval_per_event.nunique() == 1) & ~(interval_per_event.isna().any())).loc[strides.index]]


def _marker_axis_renaming(marker_axis: str) -> Union[str, Tuple[str, str, str]]:
    if marker_axis == "Time":
        return "time after start [s]"

    marker, rest = marker_axis.split(",")
    marker = marker.lower().replace(" ", "_")

    foot, metric = rest.strip().split(" ")

    if "(" in metric:
        metric = metric.replace("(", "_")[:-1]
        if metric.startswith("a"):
            metric = "acc" + metric.lower()[1:]
        if metric.startswith("v"):
            metric = "vel" + metric.lower()[1:]
    else:
        metric = "pos_" + metric.lower()
    metric = metric.lower()

    return foot, marker, metric


class AllData(NamedTuple):
    """Representing all data from a single participant and repetition."""

    imu_data: Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]
    marker_positions: pd.DataFrame
    reference_events: Dict[Literal["left", "right"], pd.DataFrame]
    tests_start_end: pd.DataFrame


@lru_cache(maxsize=1)
def get_data(base_dir: Path):
    """Load the single matlab file from the given path."""
    mat_data = sio.loadmat(str(base_dir / "raw_data_export.mat"), squeeze_me=True)["rawdata"]
    return {r[0]: r for r in mat_data}


def get_all_participants_and_tests(*, base_dir: Path) -> pd.DataFrame:
    """Get the folder names of all participants."""
    participants = (
        pd.read_csv(base_dir / "subjects.csv")
        .rename(columns={"patid": "participant", "retest": "repetition"})
        .astype({"participant": str, "repetition": int})
        .assign(patient=lambda df_: df_["patient"] == "Patient")
    )

    speed_df = pd.DataFrame(product(participants["ga"], ("slow", "normal", "fast")), columns=["ga", "speed"]).astype(
        {"speed": "category"}
    )
    participants = participants.merge(speed_df, on="ga").drop(columns=["ga"])

    return (
        participants[["patient", "participant", "repetition", "speed"]]
        .sort_values(["patient", "participant", "repetition", "speed"])
        .reset_index(drop=True)
    )


def get_meas_id_from_group(participant: str, repetition: int, *, base_dir: Path) -> str:
    """Create a measurement id from the given participant, repetition and speed."""
    mapping = (
        pd.read_csv(base_dir / "subjects.csv").astype({"patid": str, "retest": int}).set_index(["patid", "retest"])
    )
    return mapping.loc[(participant, repetition), "ga"]


def get_all_data_for_recording(
    participant: str,
    repetition: int,
    *,
    base_dir: Path,
    camera_visible_border_offset_s: float = 0.3,
    camera_visible_minimal_interval_s: float = 1.5,
) -> AllData:
    """Get the data for the given participant, repetition and speed."""
    meas_id = get_meas_id_from_group(participant, repetition, base_dir=base_dir)
    all_data_for_measurement = get_data(base_dir)[meas_id]
    all_data_for_measurement = {
        n: all_data_for_measurement[i] for i, n in enumerate(all_data_for_measurement.dtype.names)
    }

    tests_start_end = pd.DataFrame(all_data_for_measurement["camera_events"]["Tests"][()][()][0])
    tests_start_end = (
        tests_start_end[tests_start_end["Name"].str.startswith("4x10_")]
        .rename(columns=lambda s: s.lower())
        .assign(name=lambda df_: df_["name"].str.split("_").str[1])
        .set_index("name")
        .drop(columns=["duration"])
    )

    # Marker data
    marker_positions = (
        pd.DataFrame(
            all_data_for_measurement["camera_data"][()]["data"],
            columns=all_data_for_measurement["camera_data"][()]["columnNames"],
        )
        .drop(columns=["L_Foot_Ground_Angle", "R_Foot_Ground_Angle"])
        .rename(columns=_marker_axis_renaming)
        .iloc[1:-1]
        .set_index("time after start [s]")
    )
    marker_positions.columns = pd.MultiIndex.from_tuples(marker_positions.columns, names=["foot", "marker", "metric"])
    # We don't select the `foot` marker. According to Felix this is not important and was calculated differently.
    marker_positions = marker_positions.loc[:, pd.IndexSlice[:, ["ankle", "foot_tip"]]]

    # Tracking available
    # Tracking of the trajectories is only avalibale for some regions of the signal.
    # If no tracking information is available, all markers have a velocity of zero.
    # We can use this to detect the regions where tracking is available.
    camera_visible = marker_positions.groupby("foot", axis=1).apply(
        lambda df_: (df_.filter(like="vel_") != 0.0).any(axis=1)
    )

    camera_visible_intervals = {}
    for foot in camera_visible.columns:
        camera_visible_intervals[foot] = pd.DataFrame(
            bool_array_to_start_end_array(camera_visible[foot].to_numpy()), columns=["start", "end"]
        )
    camera_visible_intervals = pd.concat(camera_visible_intervals, names=["foot", "interval"]) / MOCAP_SAMPLING_RATE_HZ

    # We inset the border to avoid edge effects, as we don't trust the trajectories shortly
    # before they go out of view.
    camera_visible_intervals["start"] += camera_visible_border_offset_s
    camera_visible_intervals["end"] -= camera_visible_border_offset_s

    # We remove short intervals, as they are likely to be noise.
    camera_visible_intervals = camera_visible_intervals.loc[
        camera_visible_intervals.eval("end - start") >= camera_visible_minimal_interval_s
    ]

    marker_positions = marker_positions.groupby("foot", axis=1, group_keys=False).apply(
        lambda df_: intersect_data(
            df_, camera_visible_intervals.loc[df_.name, "start"], camera_visible_intervals.loc[df_.name, "end"]
        )
    )

    # IMU data
    imu_data = {}
    for foot_data_raw in all_data_for_measurement["sensor_data"][()]:
        foot_data = foot_data_raw[()]
        foot_sensor = SENSOR_RENAMING_DICT[foot_data["sensorPosition"]]
        # NOTE: The data file contains an "offset" field, but this offset is already applied to the data.
        sensor_data = (
            pd.DataFrame(foot_data["data"], columns=foot_data["columnNames"])
            .rename(columns=SENSOR_AXIS_RENAMING_DICT)
            .iloc[1:-1]
            .set_index("time after start [s]")
        )
        sensor_data = flip_sensor(sensor_data, Rotation.from_matrix(COORDINATE_SYSTEM_TRANSFORMATION_SH3[foot_sensor]))
        sensor_data.loc[:, SF_ACC] *= 9.81  # The data is in g, we convert to m/s^2
        imu_data[foot_sensor] = sensor_data

    # Reference Events
    reference_events = (
        pd.DataFrame(all_data_for_measurement["camera_events"]["Gait_Events"][()][()][0])  # noqa: PD010
        .dropna(subset=["Reference_HS"])
        .assign(foot=lambda df_: df_["Name"].str.split("_").str[0].replace({"L": "left", "R": "right"}))
        .assign(event=lambda df_: df_["Name"].str.split("_").str[1])
        .assign(s_id=lambda df_: df_["Reference_HS"].astype("category").cat.codes)
        .set_index(["foot", "s_id", "event"])["Start"]
        # At this stage it can happen that the same event is found twice for one stride.
        # This is an error and we will drop all instances of this.
        # Note, that we will drop both instances, basically producing an invalid stride that we drop later
        .loc[(lambda df_: ~df_.index.duplicated(keep=False))]
        .unstack("event")
        .sort_index()
    )

    # Now we have the events per stride (each row is a stride)
    # We need to define a start and an end for each stride
    # This will be HS->HS
    # Without knowledge about the validity of each stride we will first of all do this for all strides after grouping
    # by foot
    reference_events["start"] = reference_events["HS"]
    reference_events["end"] = reference_events["HS"].groupby("foot").shift(-1)

    reference_events = reference_events.astype(float)

    # For reference checking, we also add the start of the stride of the opposite foot that happened during the stance
    # phase of the current foot
    reference_events = reference_events.sort_values(["start"])
    reference_events["start_opposite"] = _interleave_foot_events(reference_events["start"], ("left", "right"))

    # New we do validity checks
    # Check 1: All events need to be present (no nans)
    reference_events = reference_events.dropna()
    # Check 2: All events should happend between start and end
    start_is_smallest = reference_events.min(axis=1) == reference_events["start"]
    end_is_largest = reference_events.max(axis=1) == reference_events["end"]
    reference_events = reference_events[start_is_smallest & end_is_largest]

    # Remove all events outside the regions tracked by the camera
    reference_events = reference_events.groupby("foot", group_keys=False).apply(
        lambda df_: intersect_strides(
            df_, camera_visible_intervals.loc[df_.name, "start"], camera_visible_intervals.loc[df_.name, "end"]
        )
    )
    # Some final mangling
    reference_events = (
        reference_events.drop(columns="start_opposite")
        .rename(columns={"HS": "ic", "TO": "to", "HO": "ho"})
        .assign(tc=lambda df_: df_[["to", "ho"]].max(axis=1))
        .astype(float)
        .mul(100)  # Convert from seconds to index since start mocap
        .round()
        .astype(int)
    )

    return AllData(
        imu_data=imu_data,
        marker_positions=marker_positions,
        # Note: For some reason, the following line can not be replaced by a simple `dict` call
        reference_events={k: v for k, v in reference_events.groupby("foot")},  # noqa: C416
        tests_start_end=tests_start_end,
    )
