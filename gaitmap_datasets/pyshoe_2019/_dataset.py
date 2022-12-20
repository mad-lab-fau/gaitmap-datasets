from pathlib import Path
from typing import Union, Optional, List

import pandas as pd
from tpcp import Dataset

from gaitmap_datasets.pyshoe_2019.helper import get_data_vicon, get_all_vicon_trials


class PyShoe2019Vicon(Dataset):
    """Dataset helper for the Vicon portion for the PyShoe dataset.

    Parameters
    ----------
    data_folder
        The base folder where the dataset can be found.
        Note, this should be the folder that was created when downloading the PyShoe dataset and *not* just the
        "data" sub-folder.
    groupby_cols
        `tpcp` internal parameters.
    subset_index
        `tpcp` internal parameters.

    """

    data_folder: Union[str, Path]

    def __init__(
        self,
        data_folder: Union[str, Path],
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None
    ):
        self.data_folder = data_folder
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return 200.0

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        """Get the sampling rate of the motion capture system."""
        return 200.0

    @property
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        return Path(self.data_folder)

    @property
    def data(self) -> pd.DataFrame:
        """Get the imu data.

        The index is provided as seconds since the start of the trial.
        """
        self.assert_is_single(None, "data")
        trial = self.group
        return get_data_vicon(trial, base_dir=self._data_folder_path)[0]

    @property
    def marker_position_(self) -> pd.DataFrame:
        """Get the marker position data.

        The index is provided as seconds since the start of the trial and should line up with the imu data.
        """
        self.assert_is_single(None, "marker_position_")
        trial = self.group
        return get_data_vicon(trial, base_dir=self._data_folder_path)[1]

    def create_index(self) -> pd.DataFrame:
        """Create the index for the dataset."""
        return pd.DataFrame(get_all_vicon_trials(self._data_folder_path), columns=["trial"])
