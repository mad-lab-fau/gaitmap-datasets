from pathlib import Path
from typing import Union

import c3d
import numpy as np
import pandas as pd


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
