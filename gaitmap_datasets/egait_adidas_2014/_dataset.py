"""The core tpcp Dataset class for the Egait Parameter Validation Dataset."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets import get_dataset_path
from gaitmap_datasets.egait_adidas_2014.helper import (
    get_all_data_for_participant_and_test,
    get_all_participants_and_tests,
    get_mocap_data_for_participant_and_test,
    get_mocap_offset,
    get_synced_stride_list,
)


class EgaitAdidas2014(Dataset):
    """Egait dataset with Mocap reference recorded in 2014."""

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None
    ):
        self.data_folder = data_folder
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def mocap_offset_s_(self) -> float:
        """Get the offset of the mocap data."""
        return get_mocap_offset(
            *self.group,
            imu_sampling_rate=self.sampling_rate_hz,
            mocap_sampling_rate=self.mocap_sampling_rate_hz_,
            base_dir=self._data_folder_path
        )

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        if self.group.sensor == "shimmer3":
            return 204.8
        return 102.4

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
        """Get the imu data."""
        self.assert_is_single(None, "data")
        data = self.memory.cache(get_all_data_for_participant_and_test)(*self.group, base_dir=self._data_folder_path)
        data.index /= self.sampling_rate_hz
        data.index.name = "time [s]"
        return data

    def _get_mocap_data(self):
        return self.memory.cache(get_mocap_data_for_participant_and_test)(*self.group, base_dir=self._data_folder_path)

    @property
    def segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list."""
        self.assert_is_single(None, "segmented_stride_list_")
        return get_synced_stride_list(*self.group, system="imu", base_dir=self._data_folder_path)

    @property
    def mocap_segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list."""
        self.assert_is_single(None, "mocap_segmented_stride_list_")
        return get_synced_stride_list(*self.group, system="mocap", base_dir=self._data_folder_path)

    @property
    def marker_position_(self) -> pd.DataFrame:
        """Get the marker position."""
        self.assert_is_single(None, "marker_position_")
        marker_position = self._get_mocap_data()[0]
        marker_position.index /= self.mocap_sampling_rate_hz_
        marker_position.index.name = "time [s]"
        return marker_position

    @property
    def gaitrite_parameters_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the gaitrite parameters."""
        self.assert_is_single(None, "gaitrite_parameters_")
        return get_gaitrite_parameters(self.group, base_dir=self._data_folder_path)

    def create_index(self) -> pd.DataFrame:
        """Create index."""
        return pd.DataFrame(get_all_participants_and_tests(base_dir=self._data_folder_path))[
            ["participant", "sensor", "stride_length", "stride_velocity", "repetition"]
        ]
