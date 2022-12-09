"""The core tpcp Dataset class for the Stair Psotion Comparison dataset.

We provide 2 versions of the dataset:

SensorPositionDatasetSegmentation: In this dataset no Mocap ground truth is provided and the IMu data is not cut to
    the individdual gait test, but just a single recording for all participants exists with all tests (including
    failed once) and movement between the tests.
    This can be used for stride segmentation tasks, as we hand-labeled all stride-start-end events in these recordings
SensorPositionDatasetMocap: In this dataset the data is cut into the individual tests.
    This means 7 data segments exist per participants.
    For each of these segments full synchronised motion capture reference is provided.

For more information about the dataset, see the dataset [documentation](https://zenodo.org/record/5747173)
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from imucal.management import CalibrationWarning
from joblib import Memory
from nilspodlib.exceptions import CorruptedPackageWarning, LegacyWarning, SynchronisationWarning
from tpcp import Dataset

from mad_datasets.sensor_position_comparison_2019.helper import (
    align_coordinates,
    get_all_participants,
    get_all_tests,
    get_foot_sensor,
    get_imu_test,
    get_manual_labels,
    get_manual_labels_for_test,
    get_mocap_events,
    get_mocap_test,
    get_session_df,
)


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
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning)
            )
            if self.align_data is True:
                session_df = self.memory.cache(_get_session_and_align)(
                    self.index["participant"].iloc[0], data_folder=self.data_folder
                )
            else:
                session_df = self.memory.cache(get_session_df)(
                    self.index["participant"].iloc[0], data_folder=self.data_folder
                )
        return session_df

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


class SensorPositionDatasetSegmentation(_SensorPostionDataset):
    """A dataset for stride segmentation benchmarking.

    Data is only loaded once the respective attributes are accessed.
    This means filtering the dataset should be fast, but accessing attributes like `.data` can be slow.
    By default we do not perform any caching of these values.
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
        stride_list = get_manual_labels(index["participant"].iloc[0], self.data_folder)
        stride_list = stride_list.set_index("s_id")
        return stride_list

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"participant": get_all_participants(self.include_wrong_recording, data_folder=self.data_folder)}
        )


class SensorPositionDatasetMocap(_SensorPostionDataset):
    """A dataset for trajectory benchmarking.

    Data is only loaded once the respective attributes are accessed.
    This means filtering the dataset should be fast, but accessing attributes like `.data` can be slow.
    By default we do not perform any caching of these values.
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
                     outside the gait tests!
                     However, the time value provided in the index of the pandas Dataframe are still aligned!
    memory
        Optional joblib memory object to cache the data loading. Note that this can lead to large hard disk usage!
    groupby_cols
        `tpcp` internal parameters.
    subset_index
        `tpcp` internal parameters.

    """

    data_padding_s: int

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        include_wrong_recording: bool = False,
        align_data: bool = True,
        data_padding_s: int = 0,
        memory: Optional[Memory] = None,
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
            self.index["participant"].iloc[0],
            self.index["test"].iloc[0],
            session_df=session_df,
            data_folder=self.data_folder,
            padding_s=self.data_padding_s,
        )
        df = df.reset_index(drop=True)
        df.index /= self.sampling_rate_hz
        df.index -= self.data_padding_s
        df.index.name = "time after start [s]"

        return df

    @property
    def data_padding_imu_samples(self):
        """Get the actual padding in samples based on `data_padding_s`."""
        return int(round(self.data_padding_s * self.sampling_rate_hz))

    def _get_segmented_stride_list(self, index) -> pd.DataFrame:
        stride_list = get_manual_labels_for_test(
            index["participant"].iloc[0], index["test"].iloc[0], data_folder=self.data_folder
        )
        stride_list = stride_list.set_index("s_id")
        stride_list[["start", "end"]] += self.data_padding_imu_samples
        return stride_list

    @property
    def mocap_events_(self) -> Dict[str, pd.DataFrame]:
        """Get mocap events calculated the Zeni Algorithm.

        Note that the events are provided in mocap samples after the start of the test.
        `self.data_padding_s` is also ignored.
        """
        self.assert_is_single(None, "mocap_events_")
        mocap_events = get_mocap_events(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.data_folder
        )
        mocap_events = {k: v.drop("foot", axis=1) for k, v in mocap_events.groupby("foot")}
        return mocap_events

    @property
    def mocap_sampling_rate_hz_(self) -> float:
        """Get the sampling rate of the motion capture system."""
        return 100.0

    @property
    def marker_position_(self) -> pd.DataFrame:
        """Get the marker trajectories of a test.

        Note, the index is provided in seconds after the start of the test and `self.data_padding_s` is ignored!
        """
        self.assert_is_single(None, "marker_position_")
        df = self.memory.cache(get_mocap_test)(
            self.index["participant"].iloc[0], self.index["test"].iloc[0], data_folder=self.data_folder
        )
        df = df.reset_index(drop=True)
        df.index /= self.mocap_sampling_rate_hz_
        df.index.name = "time after start [s]"
        return df

    def create_index(self) -> pd.DataFrame:
        tests = (
            (p, t)
            for p in get_all_participants(self.include_wrong_recording, data_folder=self.data_folder)
            for t in get_all_tests(p, self.data_folder)
        )
        return pd.DataFrame(tests, columns=["participant", "test"])
