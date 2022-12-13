"""General Helpers to load binary data."""
from pathlib import Path
from typing import Dict, Type

import numpy as np
import pandas as pd


def load_bin_file(path: Path, dtype_dict: Dict[str, Type[np.dtype]]) -> pd.DataFrame:
    """Load a binary file containing sensor data.

    This function will load data from a binary file given the specified data type structure.

    Parameters
    ----------
    path:
        full path to binary file which shall be loaded

    dtype_dict:
        dictionary with pairs of descriptive strings and numpy.datatypes describing one sample

    Returns
    -------
    parsed data
        This will be a numpy array.....

    Examples
    --------
    >>> # pd.DataFrame containing one or multiple sensor data streams, each of containing all 6 IMU
    >>> path = "/path/to/your/file.bin"
    >>> data_type_dict = {'gyr_x': np.int16,
    ...                   'gyr_y': np.int16,
    ...                   'gyr_z': np.int16,
    ...                   'acc_x': np.int16,
    ...                   'acc_y': np.int16,
    ...                   'acc_z': np.int16,
    ...                   'counter': np.uint32}
    >>> data = load_bin_file(path, data_type_dict)

    """
    dtype_list = list(dtype_dict.items())
    data = np.fromfile(path, dtype=np.dtype(dtype_list))
    return pd.DataFrame(data)
