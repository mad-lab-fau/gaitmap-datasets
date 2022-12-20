"""Helper to load the pyshoe data."""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.transform import Rotation

from gaitmap_datasets.utils.consts import SF_COLS, SF_GYR
from gaitmap_datasets.utils.coordinate_transforms import flip_sensor

COORDINATE_TRANSFORMATION_DICT = {
    # [[x -> x], [y -> -y], [z -> -z]]
    "right_sensor": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
}

VICON_SUBFOLDER = Path("data/vicon/processed")
HALLWAY_SUBFOLDER = Path("data/hallway/")


def _transform_imu_data(data: pd.DataFrame) -> pd.DataFrame:
    imu = flip_sensor(data, Rotation.from_matrix(COORDINATE_TRANSFORMATION_DICT["right_sensor"]))
    imu.loc[:, SF_GYR] = np.rad2deg(imu.loc[:, SF_GYR])
    imu.columns = pd.MultiIndex.from_tuples((("right_sensor", c) for c in imu.columns), names=["sensor", "axis"])
    return imu


@lru_cache(maxsize=1)
def get_data_vicon(trial: str, *, base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the data from the vicon portion of the trial."""
    data_path = Path(base_dir) / VICON_SUBFOLDER
    data = sio.loadmat(str(data_path / f"{trial}.mat"))
    ts = pd.Series(data["ts"][0], name="time [s]")
    imu = _transform_imu_data(pd.DataFrame(data["imu"], columns=SF_COLS, index=ts))
    gt = pd.DataFrame(data["gt"], columns=["x", "y", "z"], index=ts) * 1000
    gt.columns = pd.MultiIndex.from_tuples((("right_sensor", c) for c in gt.columns), names=["sensor", "direction"])
    return imu, gt


@lru_cache(maxsize=1)
def get_data_hallway(trial: Tuple[str, str, str], *, base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Get the data from the hallway portion of the trial."""
    data_path = Path(base_dir) / HALLWAY_SUBFOLDER / trial[1] / trial[0][1] / trial[2]

    data = sio.loadmat(str(data_path / "processed_data.mat"))
    ts = data["ts"][0]
    imu = _transform_imu_data(pd.DataFrame(data["imu"], columns=SF_COLS, index=pd.Series(ts, name="time [s]")))
    gt_idx = data["gt_idx"][0]
    gt = pd.DataFrame(data["gt"], columns=["x", "y", "z"], index=pd.Series(ts[gt_idx], name="time [s]")) * 1000
    gt.columns = pd.MultiIndex.from_tuples((("right_sensor", c) for c in gt.columns), names=["sensor", "direction"])
    return imu, gt, pd.Series(gt_idx, index=pd.Series(ts[gt_idx], name="time [s]"))


def get_all_vicon_trials(base_dir: Path):
    """Get all vicon trials."""
    data_path = Path(base_dir) / VICON_SUBFOLDER
    yield from tuple(f.stem for f in data_path.glob("*.mat"))


def get_all_hallway_trials(base_dir: Path):
    """Get all hallway trials."""
    data_path = Path(base_dir) / HALLWAY_SUBFOLDER
    for f in data_path.rglob("processed_data.mat"):
        # Participant, trial_type, trial_number
        yield f"p{f.parent.parent.name}", f.parent.parent.parent.name, f.parent.name
