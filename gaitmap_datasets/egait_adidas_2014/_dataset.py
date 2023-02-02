"""The core tpcp Dataset class for the Egait Parameter Validation Dataset."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Callable

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.egait_adidas_2014.helper import (
    get_all_data_for_participant_and_test,
    get_all_participants_and_tests,
    get_mocap_data_for_participant_and_test,
    get_mocap_offset_s,
    get_mocap_parameters,
    get_synced_stride_list,
)
from gaitmap_datasets.utils.event_detection import convert_sampling_rates_event_list


def _apply_to_dict_dfs(
    dict_or_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], function: Callable[[str, pd.DataFrame], pd.DataFrame]
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if isinstance(dict_or_df, dict):
        return {k: function(k, v) for k, v in dict_or_df.items()}
    else:
        return function(dict_or_df)


class EgaitAdidas2014(Dataset):
    """Egait dataset with Mocap reference recorded in 2014."""

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_folder = data_folder
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def mocap_offset_s_(self) -> Dict[Literal["left_sensor", "right_sensor"], float]:
        """Get the offset of the mocap data."""
        return get_mocap_offset_s(
            *self.group,
            imu_sampling_rate=self.sampling_rate_hz,
            mocap_sampling_rate=self.mocap_sampling_rate_hz_,
            base_dir=self._data_folder_path,
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
    def data(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the imu data."""
        self.assert_is_single(None, "data")
        data = self.memory.cache(get_all_data_for_participant_and_test)(*self.group, base_dir=self._data_folder_path)
        mocap_offset = self.mocap_offset_s_
        final_data = {}
        for sensor, sensor_data in data.items():
            sensor_data.index /= self.sampling_rate_hz
            sensor_data.index -= mocap_offset[sensor]
            sensor_data.index.name = "time [s]"
            final_data[sensor] = sensor_data
        return final_data

    def _get_mocap_data(self):
        return self.memory.cache(get_mocap_data_for_participant_and_test)(*self.group, base_dir=self._data_folder_path)

    @property
    def segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list in samples relative to the IMU start."""
        self.assert_is_single(None, "segmented_stride_list_")
        return get_synced_stride_list(*self.group, system="imu", base_dir=self._data_folder_path)

    @property
    def marker_position_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the marker position."""
        self.assert_is_single(None, "marker_position_")
        marker_position = self._get_mocap_data()[0]
        final_marker_position = {}
        for k, v in marker_position.items():
            v.index /= self.mocap_sampling_rate_hz_
            v.index.name = "time [s]"
            final_marker_position[k] = v
        return final_marker_position

    @property
    def mocap_parameters_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Stride parameters extracted from the mocap system."""
        self.assert_is_single(None, "mocap_parameters_")
        return get_mocap_parameters(*self.group, base_dir=self._data_folder_path)

    def convert_events(
        self,
        events: Dict[str, pd.DataFrame],
        from_time_axis: Literal["mocap", "imu"],
        to_time_axis: Literal["mocap", "imu", "time"],
    ):
        if not isinstance(events, dict):
            raise TypeError("The events always need to be a dict of form `sensor_name: event_df`. "
                            "This is required, as we need to know, which foot the data belongs to, to do some "
                            "conversions properly.")
        if from_time_axis == to_time_axis:
            return events.copy()

        if from_time_axis == "mocap":
            if to_time_axis == "imu":
                return _apply_to_dict_dfs(
                    events,
                    lambda name, df: convert_sampling_rates_event_list(
                        df, self.mocap_sampling_rate_hz_, self.sampling_rate_hz
                    ) + int(round(self.mocap_offset_s_[name] / self.sampling_rate_hz)),
                )
            if to_time_axis == "time":
                return _apply_to_dict_dfs(events, lambda _, df: df / self.mocap_sampling_rate_hz_)

        if from_time_axis == "imu":
            if to_time_axis == "mocap":
                return _apply_to_dict_dfs(
                    events,
                    lambda name, df: convert_sampling_rates_event_list(
                        df - int(round(self.mocap_offset_s_[name] / self.sampling_rate_hz)), self.sampling_rate_hz, self.mocap_sampling_rate_hz_
                    ),
                )
            if to_time_axis == "time":
                return _apply_to_dict_dfs(
                    events, lambda name, df: df / self.sampling_rate_hz - self.mocap_offset_s_[name]
                )
        raise ValueError(f"Cannot convert from {from_time_axis} to {to_time_axis}.")

    def create_index(self) -> pd.DataFrame:
        """Create index."""
        return pd.DataFrame(get_all_participants_and_tests(base_dir=self._data_folder_path))[
            ["participant", "sensor", "stride_length", "stride_velocity", "repetition"]
        ]
