"""The core tpcp dataset classes for the pyshoe dataset."""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.pyshoe_2019.helper import (
    get_all_hallway_trials,
    get_all_stairs_trials,
    get_all_vicon_trials,
    get_data_hallway,
    get_data_stairs,
    get_data_vicon,
)


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

    data_folder: Optional[Union[str, Path]]

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
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
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
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
        """Get the marker position in mm.

        The index is provided as seconds since the start of the trial and should line up with the imu data.
        """
        self.assert_is_single(None, "marker_position_")
        trial = self.group
        return get_data_vicon(trial, base_dir=self._data_folder_path)[1]

    def create_index(self) -> pd.DataFrame:
        """Create the index for the dataset."""
        return pd.DataFrame(get_all_vicon_trials(self._data_folder_path), columns=["trial"])


class PyShoe2019Hallway(Dataset):
    """Dataset helper for the hallway portion for the PyShoe dataset.

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

    data_folder: Optional[Union[str, Path]]

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
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
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
        return Path(self.data_folder)

    @property
    def data(self) -> pd.DataFrame:
        """Get the imu data.

        The index is provided as seconds since the start of the trial.
        """
        self.assert_is_single(None, "data")
        trial = self.group
        return get_data_hallway(trial, base_dir=self._data_folder_path)[0]

    @property
    def position_reference_(self) -> pd.DataFrame:
        """Get the position reference along the trial in mm from the starting position.

        The returning dataframe provides the expected position of the sensor during specific time points during the
        trials.
        The index is provided as seconds since the start of the trial and should line up with the imu data.

        If the sampling point of the reference is required as indices of the imu data, use the
        `position_reference_index_` property.
        """
        self.assert_is_single(None, "position_reference_")
        trial = self.group
        return get_data_hallway(trial, base_dir=self._data_folder_path)[1]

    @property
    def position_reference_index_(self) -> pd.Series:
        """Get the indices when the position reference was sampled."""
        self.assert_is_single(None, "position_reference_index_")
        trial = self.group
        return get_data_hallway(trial, base_dir=self._data_folder_path)[2]

    def create_index(self) -> pd.DataFrame:
        """Create the index for the dataset."""
        return pd.DataFrame(get_all_hallway_trials(self._data_folder_path), columns=["participant", "type", "trial"])


class PyShoe2019Stairs(Dataset):
    """Dataset helper for the staircase portion for the PyShoe dataset.

    Note, this only contains the data of the "test" subfolder, as only this part of the data contains the ground truth
    reference derived based on the stair geometries.

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

    data_folder: Optional[Union[str, Path]]

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
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
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
        return Path(self.data_folder)

    @property
    def data(self) -> pd.DataFrame:
        """Get the imu data.

        The index is provided as seconds since the start of the trial.
        """
        self.assert_is_single(None, "data")
        trial = self.group
        return get_data_stairs(trial, base_dir=self._data_folder_path)[0]

    @property
    def position_reference_(self) -> pd.DataFrame:
        """Get the position reference along the trial in mm from the starting position.

        The returning dataframe provides the expected position of the sensor during specific time points during the
        trials.
        The index is provided as seconds since the start of the trial and should line up with the imu data.

        If the sampling point of the reference is required as indices of the imu data, use the
        `position_reference_index_` property.

        Note, that for this dataset only reference for the z-level is provided as the ground truth was derived based on
        the stair geometries.
        """
        self.assert_is_single(None, "position_reference_")
        trial = self.group
        return get_data_stairs(trial, base_dir=self._data_folder_path)[1]

    @property
    def position_reference_index_(self) -> pd.Series:
        """Get the indices when the position reference was sampled."""
        self.assert_is_single(None, "position_reference_index_")
        trial = self.group
        return get_data_stairs(trial, base_dir=self._data_folder_path)[2]

    def create_index(self) -> pd.DataFrame:
        """Create the index for the dataset."""
        return pd.DataFrame(
            get_all_stairs_trials(self._data_folder_path), columns=["n_levels", "first_direction", "trial"]
        )
