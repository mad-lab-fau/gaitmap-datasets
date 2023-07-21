"""Main dataset class to load the Kluge2017 dataset."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets import get_dataset_path
from gaitmap_datasets.kluge_2017.helper import (
    IMU_SAMPLING_RATE_HZ,
    MOCAP_SAMPLING_RATE_HZ,
    AllData,
    get_all_data_for_recording,
    get_all_participants_and_tests,
    get_meas_id_from_group,
    intersect_strides,
)


class Kluge2017(Dataset):
    """A dataset to validate spatial-temporal parameters in healthy and PD.

    Parameters
    ----------
    data_folder : Optional[Union[str, Path]], optional
        The base folder where the dataset can be found.
    memory : Memory, optional
        A memory object to optioanlly use disk caching to speed up loading.
    groupby_cols
        `tpcp` internal parameters.
    subset_index
        `tpcp` internal parameters.
    """

    data_folder: Optional[Union[str, Path]]
    memory: Memory

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None
    ):
        self.memory = memory
        self.data_folder = data_folder
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
        return Path(self.data_folder)

    def _get_all_data_for_recording_cached(self, participant, repetition) -> AllData:
        return get_all_data_for_recording(participant, repetition, base_dir=self._data_folder_path)

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return IMU_SAMPLING_RATE_HZ

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        """Get the sampling rate of the IMUs."""
        return MOCAP_SAMPLING_RATE_HZ

    @property
    def _meas_id(self):
        self.assert_is_single(None, "_meas_id")
        _, participant, repetition, _ = self.group
        return get_meas_id_from_group(participant, repetition, base_dir=self._data_folder_path)

    @property
    def data(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        self.assert_is_single(None, "data")
        _, participant, repetition, test = self.group
        all_data = self._get_all_data_for_recording_cached(participant, repetition)
        test_start, test_end = all_data.tests_start_end.loc[test]

        data_per_test = {}
        for sensor, data in all_data.imu_data.items():
            tmp = data.loc[data.index.to_series().between(test_start, test_end)]
            tmp.index = tmp.index - test_start
            data_per_test[sensor] = tmp
        return data_per_test

    @property
    def marker_position_(self):
        self.assert_is_single(None, "marker_position_")
        _, participant, repetition, test = self.group
        all_data = self._get_all_data_for_recording_cached(participant, repetition)
        test_start, test_end = all_data.tests_start_end.loc[test]

        data_per_test = all_data.marker_positions.loc[
            all_data.marker_positions.index.to_series().between(test_start, test_end)
        ]
        data_per_test.index = data_per_test.index - test_start
        return data_per_test

    @property
    def marker_position_per_stride_(self):
        self.assert_is_single(None, "marker_position_per_stride_")
        trajectory = self.marker_position_
        mocap_events = self.mocap_events_

        per_stride_trajectory = {}
        for foot, events in mocap_events.items():
            output_per_foot = {}
            data = trajectory[foot]
            for s_id, stride in events.iterrows():
                # This cuts out the n+1 samples for each stride.
                # The first sample is the value before the stride started.
                # This is the equivalent to the "initial" position/orientation
                output_per_foot[s_id] = data.iloc[int(stride["start"]) : int(stride["end"] + 1)].reset_index(drop=True)
            per_stride_trajectory[foot] = pd.concat(output_per_foot, names=["s_id", "sample"])
        return per_stride_trajectory

    @property
    def mocap_events_(self):
        self.assert_is_single(None, "stride_event_list_")
        _, participant, repetition, test = self.group
        all_data = self._get_all_data_for_recording_cached(participant, repetition)
        test_start, test_end = all_data.tests_start_end.loc[test] * self.mocap_sampling_rate_hz_

        return {
            k: (intersect_strides(v, [test_start], [test_end]) - test_start).astype(int).reset_index("foot", drop=True)
            for k, v in all_data.reference_events.items()
        }

    def create_index(self) -> pd.DataFrame:
        """Create the index for the dataset."""
        return self.memory.cache(get_all_participants_and_tests)(base_dir=self._data_folder_path)
