"""A set of helpers to load the dataset."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import c3d
import numpy as np
import pandas as pd
from imucal.management import CalibrationWarning
from nilspodlib import SyncedSession
from nilspodlib.exceptions import CorruptedPackageWarning, LegacyWarning, SynchronisationWarning
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from gaitmap_datasets.utils.coordinate_transforms import flip_dataset, rotation_from_angle

COORDINATE_TRANSFORMATION_DICT = dict(
    qualisis_lateral_nilspodv1={
        # [[+y -> +x], [+z -> +y], [+x -> +z]]
        "left_sensor": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        # [[-y -> +x], [-z -> +y], [+x -> +z]]
        "right_sensor": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    },
    qualisis_medial_nilspodv1={
        # [[-y -> +x], [-z -> +y], [+x -> +z]]
        "left_sensor": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        # [[+y -> +x], [+z -> +y], [+x -> +z]]
        "right_sensor": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    },
    qualisis_instep_nilspodv1={
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "left_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "right_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    },
    qualisis_cavity_nilspodv1={
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "left_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        # [[-x -> +x], [-y -> +y], [+z -> +z]]
        "right_sensor": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    },
    qualisis_heel_nilspodv1={
        # [[-z -> +x], [+y -> +y], [+x -> +z]]
        "left_sensor": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        # [[-z -> +x], [+y -> +y], [+x -> +z]]
        "right_sensor": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    },
    qualisis_insole_nilspodv1={
        # [[+y -> +x], [-x -> +y], [+z -> +z]]
        "left_sensor": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        # [[-y -> +x], [+x -> +y], [+z -> +z]]
        "right_sensor": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    },
)


def get_data_folder(data_folder=None, data_subfolder=True):
    """Get the data folder or subfolder."""
    data_folder = Path(data_folder)
    if data_subfolder:
        return Path(data_folder) / "data"
    return Path(data_folder)


def get_all_participants(include_wrong_recording: bool = False, data_folder=None):
    """Iterate over all participant ids.

    If `include_wrong_recording` the first recording of 6dbe is included.
    This recording is missing one of the sensors and should therefore not be used in most cases.
    """
    data_folders = sorted(get_data_folder(data_folder).glob("[!.]*"), key=lambda x: x.name)
    if len(data_folders) == 0 or not all(len(f.name) in (4, 6) for f in data_folders):
        raise ValueError(
            "The selected folder does not seem to be correct! "
            "No data could be found. "
            f'The selected folder is: "{data_folder}"'
        )
    for participant in data_folders:
        if participant.name == "6dbe" and not include_wrong_recording:
            continue
        yield participant.name


def get_all_tests(participant_id: str, data_folder=None):
    """Iterate over all tests of a participant."""
    tests = get_metadata_participant(participant_id, data_folder=data_folder)["mocap_test_start"].keys()
    for k in tests:
        yield k


def get_participant_folder(participant_id: str, data_folder=None) -> Path:
    """Get the toplevel data folder of a participant."""
    return get_data_folder(data_folder) / participant_id


def get_participant_imu_folder(participant_id: str, data_folder=None) -> Path:
    """Get the IMU data folder of a participant."""
    return get_participant_folder(participant_id, data_folder=data_folder) / "imu"


def get_participant_mocap_folder(participant_id: str, data_folder=None) -> Path:
    """Get the mocap data folder of a participant."""
    return get_participant_folder(participant_id, data_folder=data_folder) / "mocap"


def get_mocap_events(participant_id, test, data_folder=None) -> pd.DataFrame:
    """Get all Mocap events extracted with the Zeni Algorithm.

    Note that the events are provided in mocap samples after the start of the test.
    """
    return pd.read_csv(get_participant_mocap_folder(participant_id, data_folder=data_folder) / (test + "_steps.csv"))


def get_session_df(participant_id: str, data_folder=None) -> pd.DataFrame:
    """Get and prepare the data of all sensors of a participant.

    This methods does multiple things:

    - Load all IMU files
    - Calibrate all IMU files
    - sync all IMU files correctly
    - Fix known issues as far as possible

    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
        )
        session = SyncedSession.from_folder_path(
            get_participant_imu_folder(participant_id, data_folder=data_folder), legacy_support="resolve"
        )
        session = session.align_to_syncregion()
        session = session.calibrate_imu(
            session.find_closest_calibration(
                ignore_file_not_found=False, folder=get_data_folder(data_folder, data_subfolder=False) / "calibrations"
            )
        )

    meta_data = get_metadata_participant(participant_id, data_folder=data_folder)
    sensor_map = {v.lower(): k for k, v in meta_data["sensors"].items()}

    df = session.data_as_df(concat_df=True, index="utc_datetime")
    df.columns = pd.MultiIndex.from_tuples([(sensor_map[s], a) for s, a in df.columns])
    trigger = df["sync"]["analog_2"]
    df = df.sort_index(axis=1).drop("sync", axis=1)

    # Some sensors were wrongly attached, this will be fixed here:
    if participant_id in ["8d60", "cb3d", "cdfc"]:
        rotation = rotation_from_angle(np.deg2rad(180), np.array([0, 0, 1]))
        df = flip_dataset(df, {"l_cavity": rotation})

    if participant_id in ["4d91", "5237", "80b8", "c9bb"]:
        rotation = rotation_from_angle(np.deg2rad(180), np.array([0, 0, 1]))
        df = flip_dataset(df, {"l_medial": rotation})

    df[("sync", "trigger")] = trigger
    return df


def get_imu_test(
    participant: str,
    test_name: str,
    session_df: Optional[pd.DataFrame] = None,
    padding_samples: int = 0,
    data_folder=None,
) -> pd.DataFrame:
    """Get the imu data from a single performed test.

    This will extract the IMU data from a single performed gait test.
    The start and the end of this test is synchronised with the start and end of the same test in the mocap data.

    Parameters
    ----------
    participant
        The participant id
    test_name
        The test name (must be one of the tests listed in the metadata file of the participant)
    session_df
        An optional df obtained by calling `get_session_df`. If not provided it will be loaded using the same function.
        Providing it, can improve performance if multiple tests from the same participant are required.
    padding_samples
        Additional padding for and after the test that is included in the output.
        This might be helpful to get a longer region of no movement before the test starts to perform gravity
        alignments.
        **Remember to remove the padding before comparing the output with the mocap data or the manual labeled stride
        borders**
    data_folder
        The data folder to use

    """
    if session_df is None:
        session_df = get_session_df(participant, data_folder=data_folder)
    meta_data = get_metadata_participant(participant, data_folder=data_folder)
    start_index = meta_data["imu_tests"][test_name]["start_idx"] - padding_samples
    stop_index = meta_data["imu_tests"][test_name]["stop_idx"] + padding_samples
    test = session_df.iloc[start_index : stop_index + 1]
    return test


def get_manual_labels(participant_id: str, data_folder=None) -> pd.DataFrame:
    """Get the manual stride border labels for a participant."""
    labels = get_participant_folder(participant_id, data_folder=data_folder) / "manual_stride_border.csv"
    return pd.read_csv(labels, header=0)


def get_manual_labels_for_test(participant_id: str, test_name: str, data_folder=None) -> pd.DataFrame:
    """Get all manual labels of a test.

    The label values are adapted to be the index since the start of the test.
    """
    meta_data = get_metadata_participant(participant_id, data_folder=data_folder)
    # Find the start and end index
    test_start = meta_data["imu_tests"][test_name]["start_idx"]
    test_stop = meta_data["imu_tests"][test_name]["stop_idx"]
    labels = get_manual_labels(participant_id, data_folder=data_folder)
    labels = labels[(labels["start"] >= test_start) & (labels["end"] <= test_stop)].copy()
    labels[["start", "end"]] -= test_start
    return labels.reset_index(drop=True)


def get_mocap_test(participant: str, test_name: str, data_folder=None) -> pd.DataFrame:
    """Get the marker trajectories for a single test.

    The start and end of this test are synchronised with the start and end of the IMU tests that can be obtained using
    `get_imu_test`.

    Remember, that the IMU and the Mocap system have different sampling rates that need to be adjusted before a
    comparison is possible.
    """
    folder = get_participant_folder(participant, data_folder=data_folder)
    try:
        return load_c3d_data(folder / f"mocap/{test_name}.c3d")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No Mocap data exists for participant {participant} and test {test_name}") from e


def get_metadata_participant(participant_id: str, data_folder=None) -> Dict[str, Any]:
    """Get the content of the meta data file."""
    folder = get_data_folder(data_folder) / participant_id
    return json.load((folder / "meta_data.json").open("r"))


def update_metadata_participant(participant_id: str, new_metadata: Dict[str, Any], data_folder=None):
    """Update the metadata for a participant."""
    folder = get_data_folder(data_folder) / participant_id
    json.dump(new_metadata, (folder / "meta_data.json").open("w"), indent=4, sort_keys=True)


def get_sensor_file(participant_id: str, sensor_name: str, data_folder=None) -> Path:
    """Get the path to the sensor file based on the simple position name.

    The sensor name should be of form {l/r}_{position}
    """
    folder = get_data_folder(data_folder) / participant_id
    meta_data = json.load((folder / "meta_data.json").open("r"))
    sensor_id = meta_data["sensors"][sensor_name]
    imu = folder / "imu"
    for f in imu.glob("*.bin"):
        if f.name.split("-")[1].startswith(sensor_id.upper()):
            return f
    raise FileNotFoundError(f"No sensor file found for {sensor_name} of {participant_id}")


def load_c3d_data(path: Union[Path, str], insert_nan: bool = True) -> pd.DataFrame:
    """Load a c3d file.

    Parameters
    ----------
    path
        Path to the file
    insert_nan
        If True missing values in the marker paths will be indicated with a np.nan.
        Otherwise, there are just 0 (?).

    """
    with open(path, "rb") as handle:
        reader = c3d.Reader(handle)
        frames = []

        for _, points, _ in reader.read_frames():
            frames.append(points[:, :3])

        labels = [label.strip().lower() for label in reader.point_labels]
        frames = np.stack(frames)
        frames = frames.reshape(frames.shape[0], -1)
    index = pd.MultiIndex.from_product([labels, list("xyz")])
    data = pd.DataFrame(frames, columns=index) / 1000  # To get the data in m
    if insert_nan is True:
        data[data == 0.000000] = np.nan
    return data


def get_foot_sensor(foot: Literal["left", "right"], include_insole: bool = True) -> List[str]:
    """Get the names of all sensors that are attached to a foot (left or right)."""
    sensors = ["{}_cavity", "{}_heel", "{}_lateral", "{}_medial", "{}_instep"]
    if include_insole is True:
        sensors.append("{}_insole")
    return [s.format(foot[0]) for s in sensors]


def get_foot_marker(foot: Literal["left", "right"]) -> List[str]:
    """Get the names of all markers that are attached ot a foot (left or right)."""
    sensors = ["{}_fcc", "{}_toe", "{}_fm5", "{}_fm1"]
    return [s.format(foot[0]) for s in sensors]


def align_coordinates(multi_sensor_data: pd.DataFrame):
    """Rotate all coordinate systems into the expected foot-sensor-frame."""
    feet = {"r": "right", "l": "left"}
    rotations = {}
    for s in multi_sensor_data.columns.unique(level=0):
        if "_" not in s:
            continue
        foot, pos = s.split("_")
        rot = COORDINATE_TRANSFORMATION_DICT.get(f"qualisis_{pos}_nilspodv1", None)
        if not rot:
            continue
        rotations[s] = Rotation.from_matrix(rot[f"{feet[foot]}_sensor"])
    ds = flip_dataset(multi_sensor_data.drop(columns="sync"), rotations)
    return ds
