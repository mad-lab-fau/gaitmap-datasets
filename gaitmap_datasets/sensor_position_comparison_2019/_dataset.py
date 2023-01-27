"""The core tpcp Dataset class for the Stair Postion Comparison dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset
from typing_extensions import Literal

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.sensor_position_comparison_2019.helper import (
    align_coordinates,
    get_all_participants,
    get_all_tests,
    get_foot_sensor,
    get_imu_test,
    get_manual_labels,
    get_manual_labels_for_test,
    get_metadata_participant,
    get_mocap_events,
    get_mocap_test,
    get_session_df,
)
from gaitmap_datasets.utils.event_detection import convert_sampling_rates_event_list


def _get_session_and_align(participant, data_folder):
    session_df = get_session_df(participant, data_folder=data_folder)
    return align_coordinates(session_df)


class _SensorPostionDataset(Dataset):
    data_folder: Optional[Union[str, Path]]
    include_wrong_recording: bool
    memory: Memory
    align_data: bool

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        include_wrong_recording: bool = False,
        align_data: bool = True,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_folder = data_folder
        self.include_wrong_recording = include_wrong_recording
        self.memory = memory
        self.align_data = align_data
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
        return Path(self.data_folder)

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return 204.8

    @property
    def segmented_stride_list_(self) -> Dict[str, pd.DataFrame]:
        """Get the manual segmented stride list per foot."""
        self.assert_is_single(None, "segmented_stride_list_")
        sl = self._get_segmented_stride_list(self.index)
        sl.index = sl.index.astype(int)
        sl = {k: v.drop("foot", axis=1) for k, v in sl.groupby("foot")}
        return sl

    def _get_segmented_stride_list(self, index) -> pd.DataFrame:
        raise NotImplementedError()

    def _get_base_df(self):
        self.assert_is_single(None, "data")
        if self.align_data is True:
            session_df = self.memory.cache(_get_session_and_align)(
                self.index["participant"].iloc[0], data_folder=self._data_folder_path
            )
        else:
            session_df = self.memory.cache(get_session_df)(
                self.index["participant"].iloc[0], data_folder=self._data_folder_path
            )
        return session_df

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata for a participant."""
        self.assert_is_single(["participant"], "metadata")
        particpant_id = self.group
        if isinstance(particpant_id, tuple):
            particpant_id = particpant_id.participant
        return get_metadata_participant(particpant_id, data_folder=self._data_folder_path)

    @property
    def segmented_stride_list_per_sensor_(self) -> Dict[str, pd.DataFrame]:
        """Get the segmented stride list per sensor.

        Instead of providing the stride list per foot, this ouput has all the sensors as keys and the correct
        stridelist (either left or right foot) as value.
        This can be helpful, if you want to iterate over all sensors and get the correct stride list.
        """
        stride_list = self.segmented_stride_list_
        final_stride_list = {}
        for foot in ["left", "right"]:
            foot_stride_list = stride_list[foot][["start", "end"]]
            for s in get_foot_sensor(foot):
                final_stride_list[s] = foot_stride_list
        return final_stride_list


class SensorPositionComparison2019Segmentation(_SensorPostionDataset):
    """A dataset for stride segmentation benchmarking.

    Data is only loaded once the respective attributes are accessed.
    This means filtering the dataset should be fast, but accessing attributes like `.data` can be slow.
    By default, we do not perform any caching of these values.
    This means, if you need to use the value multiple times, the best way is to assign it to a variable.
    Alternatively, you can use the `memory` parameter to create a disk based cache for the data loading.

    Parameters
    ----------
    data_folder
        The base folder where the dataset can be found.
    include_wrong_recording
        If True the first trail of 6dbe is included, which has one missing sensor
    align_data
        If True the coordinate systems of all sensors are roughly aligned based on their known mounting orientation
    memory
        Optional joblib memory object to cache the data loading. Note that this can lead to large hard disk usage!
    groupby_cols
        `tpcp` internal parameters.
    subset_index
        `tpcp` internal parameters.

    """

    @property
    def data(self) -> pd.DataFrame:
        """Get the IMU data of the as pandas Dataframe."""
        df = self._get_base_df()
        df = df.reset_index(drop=True)
        df.index /= self.sampling_rate_hz
        return df

    def _get_segmented_stride_list(self, index) -> pd.DataFrame:
        stride_list = get_manual_labels(index["participant"].iloc[0], self._data_folder_path)
        stride_list = stride_list.set_index("s_id")
        return stride_list

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"participant": get_all_participants(self.include_wrong_recording, data_folder=self._data_folder_path)}
        )


class SensorPositionComparison2019Mocap(_SensorPostionDataset):
    """A dataset for trajectory benchmarking.

    Data is only loaded once the respective attributes are accessed.
    This means filtering the dataset should be fast, but accessing attributes like `.data` can be slow.
    By default, we do not perform any caching of these values.
    This means, if you need to use the value multiple times, the best way is to assign it to a variable.
    Alternatively, you can use the `memory` parameter to create a disk based cache for the data loading.

    Parameters
    ----------
    data_folder
        The base folder where the dataset can be found.
    include_wrong_recording
        If True the first trail of 6dbe is included, which has one missing sensor
    align_data
        If True the coordinate systems of all sensors are roughly aligned based on their known mounting orientation
    data_padding_s
        A number of seconds that are added to the start and the end of each IMU recording.
        This can be used to get a longer static period before each gait test to perform e.g. gravity based alignments.
        For samples before the start of the gait test, the second index of the pd.DataFrame is set to negative values.
        This should make it easy to remove the padded values if required.

        .. warning:: The same padding is not applied to the mocap samples (as we do not have any mocap samples
                     outside the gait tests)!
                     However, the time value provided in the index of the pandas Dataframe are still aligned,
                     as we add negative time values to the IMU time index.
    memory
        Optional joblib memory object to cache the data loading. Note that this can lead to large hard disk usage!
    groupby_cols
        `tpcp` internal parameters.
    subset_index
        `tpcp` internal parameters.

    """

    data_padding_s: float

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        include_wrong_recording: bool = False,
        align_data: bool = True,
        data_padding_s: float = 0,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_padding_s = data_padding_s
        super().__init__(
            data_folder,
            include_wrong_recording=include_wrong_recording,
            align_data=align_data,
            memory=memory,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    @property
    def data(self) -> pd.DataFrame:
        """Get the IMU data per gait test.

        Get the data per gait test.
        If `self.data_padding_s` is set, the extracted data region extends by that amount of second beyond the actual
        gait test.
        Keep that in mind, when aligning data to mocap.
        The time axis is provided in seconds and the 0 will be at the actual start of the gait test.
        """
        session_df = self._get_base_df()
        df = get_imu_test(
            *self.group,
            session_df=session_df,
            data_folder=self._data_folder_path,
            padding_samples=self.data_padding_imu_samples,
        )
        df = df.reset_index(drop=True)
        # We first subtract the offset in samples and then divide by the sampling rate
        # This ensures, that the 0 is still at the same position independent of the padding
        df.index = (df.index - self.data_padding_imu_samples) / self.sampling_rate_hz
        df.index.name = "time after start [s]"

        return df

    @property
    def data_padding_imu_samples(self) -> int:
        """Get the actual padding in samples based on `data_padding_s`."""
        return int(round(self.data_padding_s * self.sampling_rate_hz))

    def _get_segmented_stride_list(self, index) -> pd.DataFrame:
        stride_list = get_manual_labels_for_test(
            index["participant"].iloc[0], index["test"].iloc[0], data_folder=self._data_folder_path
        )
        stride_list = stride_list.set_index("s_id")
        stride_list[["start", "end"]] += self.data_padding_imu_samples
        return stride_list

    @property
    def mocap_events_(self) -> Dict[str, pd.DataFrame]:
        """Get mocap events calculated the Zeni Algorithm.

        Note that the events are provided in mocap samples after the start of the test.
        This means `self.data_padding_s` is ignored here.
        Use `self.convert_with_padding` to convert the events to IMU samples/seconds while respecting the padding.
        """
        self.assert_is_single(None, "mocap_events_")
        mocap_events = get_mocap_events(*self.group, data_folder=self._data_folder_path)
        mocap_events = {k: v.drop("foot", axis=1).set_index("s_id") for k, v in mocap_events.groupby("foot")}
        return mocap_events

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        """Get the sampling rate of the motion capture system."""
        return 100.0

    @property
    def marker_position_(self) -> pd.DataFrame:
        """Get the marker trajectories of a test.

        Note, the index is provided in seconds after the start of the test and `self.data_padding_s` is ignored!
        However, as long as the time domain index is used, the two data streams are aligned.

        All values are provided in mm in the global coordinate system of the motion capture system.

        NaN values are provided, if one of the marker was not visible in the mocap system and its trajectory could
        not be restored.

        """
        self.assert_is_single(None, "marker_position_")
        df = self.memory.cache(get_mocap_test)(*self.group, data_folder=self._data_folder_path)
        df = df.reset_index(drop=True)
        df.index /= self.mocap_sampling_rate_hz_
        df.index.name = "time after start [s]"
        return df

    def create_index(self) -> pd.DataFrame:
        tests = (
            (p, t)
            for p in get_all_participants(self.include_wrong_recording, data_folder=self._data_folder_path)
            for t in get_all_tests(p, self._data_folder_path)
        )
        return pd.DataFrame(tests, columns=["participant", "test"])

    def convert_with_padding(
        self,
        events: pd.DataFrame,
        from_time_axis: Literal["mocap", "imu"],
        to_time_axis: Literal["mocap", "imu", "time"],
    ):
        """Convert the time/sample values of mocap and IMU events into other time domains.

        This method will use the respective sampling rates and the padding of the IMU data to convert the time/sample.

        ... warning::
            This method will only work, if the provided samples follow the padding conventions used in this class!
            This means, if the input are events in IMU samples (`from_time_axis="imu"`), they must respect the
            padding of the IMU data.
            I.e. the first sample of the IMU data is sample 0 and test start is sample `self.data_padding_imu_samples`.
            If the input are events in mocap samples (`from_time_axis="mocap"`), they must not include the padding.
            I.e. the first sample of the mocap data is sample 0 and test start is sample 0.

        ... note::
            If the input are events in IMU samples (`from_time_axis="imu"`) and padding is used, it can happen that the
            resulting mocap samples have negative values (as the events occure before the start of the test).

        """
        if from_time_axis == to_time_axis:
            return events.copy()
        if from_time_axis == "mocap":
            if to_time_axis == "imu":
                return (
                    convert_sampling_rates_event_list(
                        events, old_sampling_rate=self.mocap_sampling_rate_hz_, new_sampling_rate=self.sampling_rate_hz
                    )
                    + self.data_padding_imu_samples
                )
            if to_time_axis == "time":
                # time 0 == mocap sample 0 -> This is independent of the padding
                return events / self.mocap_sampling_rate_hz_
        if from_time_axis == "imu":
            if to_time_axis == "mocap":
                return convert_sampling_rates_event_list(
                    events - self.data_padding_imu_samples,
                    old_sampling_rate=self.sampling_rate_hz,
                    new_sampling_rate=self.mocap_sampling_rate_hz_,
                )
            if to_time_axis == "time":
                return (events - self.data_padding_imu_samples) / self.sampling_rate_hz
        raise ValueError(f"Cannot convert from {from_time_axis} to {to_time_axis}.")
