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


@lru_cache(maxsize=1)
def get_data_vicon(trial: str, *, base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the data from the vicon portion of the trial."""
    data_path = Path(base_dir) / VICON_SUBFOLDER
    data = sio.loadmat(str(data_path / f"{trial}.mat"))
    ts = data["ts"][0]
    imu = pd.DataFrame(data["imu"], columns=SF_COLS, index=ts)
    imu = flip_sensor(imu, Rotation.from_matrix(COORDINATE_TRANSFORMATION_DICT["right_sensor"]))
    imu.loc[:, SF_GYR] = np.rad2deg(imu.loc[:, SF_GYR])
    imu.columns = pd.MultiIndex.from_tuples((("right_sensor", c) for c in imu.columns), names=["sensor", "axis"])
    gt = pd.DataFrame(data["gt"], columns=["x", "y", "z"], index=ts)
    gt.columns = pd.MultiIndex.from_tuples((("right_sensor", c) for c in gt.columns), names=["sensor", "direction"])
    return imu, gt


def get_all_vicon_trials(base_dir: Path) -> Tuple[str, ...]:
    """Get all vicon trials."""
    data_path = Path(base_dir) / VICON_SUBFOLDER
    return tuple(f.stem for f in data_path.glob("*.mat"))