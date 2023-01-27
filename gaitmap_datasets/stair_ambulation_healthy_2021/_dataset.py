"""The core tpcp Dataset class for the Stair Ambulation dataset."""
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.stair_ambulation_healthy_2021.helper import (
    StrideTypes,
    get_all_data_for_participant,
    get_all_participants,
    get_all_participants_and_tests,
    get_participant_metadata,
    get_pressure_insole_events,
    get_segmented_stride_list,
)
from gaitmap_datasets.utils.consts import SF_COLS


class _StairAmbulationHealthy2021(Dataset):
    _PRESSURE_COLUMNS = ("toe_force", "mth_force", "heel_force", "total_force")

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        memory: Memory = Memory(None),
        include_pressure_data: bool = False,
        include_hip_sensor: bool = False,
        include_baro_data: bool = False,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.memory = memory
        self.data_folder = data_folder
        self.include_pressure_data = include_pressure_data
        self.include_hip_sensor: bool = include_hip_sensor
        self.include_baro_data: bool = include_baro_data
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return 204.8

    def _get_participant_and_part(self, error_name: str) -> Tuple[str, Literal["part_1", "part_2"]]:
        """Get the participant and part of the dataset."""
        raise NotImplementedError

    def _cut_data_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cut the data to the region of interest."""
        raise NotImplementedError

    def _cut_events_to_region(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Cut the events to the region of interest."""
        raise NotImplementedError

    def _get_base_df(self, participant: str, part: Literal["part_1", "part_2"]) -> pd.DataFrame:
        """Get all the data (including baro and pressure) of a participant."""
        return self.memory.cache(get_all_data_for_participant)(
            participant,
            part,
            return_pressure_data=self.include_pressure_data,
            return_baro_data=self.include_baro_data,
            return_hip_sensor=self.include_hip_sensor,
            base_dir=self._data_folder_path,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata for a participant."""
        self.assert_is_single(["participant"], "metadata")
        return get_participant_metadata(self.group.participant, base_dir=self._data_folder_path)

    @property
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        if self.data_folder is None:
            return get_dataset_path(Path(__file__).parent.name)
        return Path(self.data_folder)

    @property
    def data(self) -> pd.DataFrame:
        """Get the IMU data of the dataset."""
        participant, part = self._get_participant_and_part("data")
        df = self._get_base_df(participant, part).drop(
            columns=["baro", *self._PRESSURE_COLUMNS], level=1, errors="ignore"
        )

        df = self._cut_data_to_region(df)
        df.index /= self.sampling_rate_hz
        df.index.name = "time [s]"
        return df

    @property
    def baro_data(self) -> pd.DataFrame:
        """Get the barometer data of the dataset."""
        if self.include_baro_data is False:
            raise ValueError("The barometer data is not loaded. Please set `include_baro_data` to True.")
        participant, part = self._get_participant_and_part("baro_data")
        df = self._get_base_df(participant, part).drop(
            columns=[*SF_COLS, *self._PRESSURE_COLUMNS], level=1, errors="ignore"
        )
        df = self._cut_data_to_region(df)
        df.index /= self.sampling_rate_hz
        df.index.name = "time [s]"
        return df

    @property
    def pressure_data(self) -> pd.DataFrame:
        """Get the pressure data of the dataset."""
        if self.include_pressure_data is False:
            raise ValueError("The pressure data is not loaded. Please set `include_pressure_data` to True.")
        participant, part = self._get_participant_and_part("pressure_data")
        df = self._get_base_df(participant, part).drop(columns=["baro", *SF_COLS], level=1, errors="ignore")
        df = self._cut_data_to_region(df)
        df.index /= self.sampling_rate_hz
        df.index.name = "time [s]"
        return df

    def get_segmented_stride_list_with_type(
        self,
        stride_type: Optional[List[StrideTypes]] = None,
        return_z_level: bool = True,
    ) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the manual stride borders of the dataset filtered by stride type.

        Parameters
        ----------
        stride_type : List[Literal["level", "ascending", "descending"]], optional
            A list of stride types to be included in the output.
            If None all strides are included.
        return_z_level
            If True, the z-level (i.e. the height change of the stride based on the stair geometry) and the stride
            type are included as additional columns in the output.

        """
        participant, part = self._get_participant_and_part("segmented_stride_list")
        stride_borders = get_segmented_stride_list(participant, part, base_dir=self._data_folder_path)
        final_stride_borders = {}
        for k, v in stride_borders.items():
            per_sensor = self._cut_events_to_region(v, numeric_cols=["start", "end"])
            if stride_type is not None:
                per_sensor = per_sensor.loc[per_sensor["type"].isin(stride_type)]
            if return_z_level is False:
                per_sensor = per_sensor.drop(columns=["type", "z_level"])
            final_stride_borders[k] = per_sensor
        return final_stride_borders

    @property
    def segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the manual stride borders of the dataset.

        This is equivalent to calling `get_segmented_stride_list_with_type(None, return_z_level=False)`.
        If you need more control, use `get_segmented_stride_list_with_type` directly.
        """
        return self.get_segmented_stride_list_with_type(return_z_level=False)

    @property
    def pressure_insole_event_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the event list based on the pressure insole (and the IMU).

        This returns all events from the pressure-insoles as a min-vel event list.
        This means that each stride starts and ends in a midstance.
        This midstance (min_vel) was detected based on the gyro energy and not the pressure insole.
        The pressure insole was used to find the IC and TC in each stride.

        If a stride has a `pre_ic` of NaN, it indicates that this is the first strid eof a gait sequence, i.e. there is
        no other stride directly before this one.

        The s_id of these strides are consistent with the `segmented_stride_list_`.
        The s_id is derived based on which min_vel - stride contains the `start` event of the segmented stride.
        """
        participant, part = self._get_participant_and_part("min_vel_event_list")
        events = get_pressure_insole_events(participant, part, base_dir=self._data_folder_path)
        final_events = {}
        for k, v in events.items():
            per_sensor = self._cut_events_to_region(v, numeric_cols=["start", "end", "ic", "tc", "min_vel", "pre_ic"])
            final_events[k] = per_sensor
        return final_events


class StairAmbulationHealthy2021PerTest(_StairAmbulationHealthy2021):
    """Dataset class representing the Stair Ambulation dataset.

    This version of the dataset contains the data split into individual tests.
    For more information about the dataset, see the `README.md` file of the dataset that is included in the dataset
    download.

    Parameters
    ----------
    data_folder
        The path to the data folder.
    memory
        The joblib memory object to cache the data loading.
    include_pressure_data
        Whether to load the raw pressure data recorded by the insole sensors.
        This will increase the load time and RAM requirements.
        Usually this is not needed unless you want to calculate your own gait events based on the pressure data.
        The precalculated gait events will still be available independent of this setting.
    include_hip_sensor
        Whether to load the raw data recorded by the hip sensor.
    include_baro_data
        Whether to load the raw data recorded by the barometer.

    """

    def _get_participant_and_part(self, error_name: str) -> Tuple[str, Literal["part_1", "part_2"]]:
        self.assert_is_single(None, error_name)
        participant, test = self.group
        part = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][test]["part"]
        return participant, part

    def _cut_data_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        # We assume that the df we get is from the correct participant and part
        participant, test = self.group
        test = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][test]
        df = df.iloc[test["start"] : test["end"]]
        return df.reset_index(drop=True)

    def _cut_events_to_region(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        # We assume that the df we get is from the correct participant and part
        participant, test = self.group
        test = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][test]
        df = df.loc[(df["start"] >= test["start"]) & (df["end"] <= test["end"])].copy()
        df.loc[:, numeric_cols] -= test["start"]
        return df

    def create_index(self) -> pd.DataFrame:
        base_dir = Path(self._data_folder_path)
        all_tests = get_all_participants_and_tests(base_dir=self._data_folder_path)
        if len(all_tests) == 0:
            raise ValueError(
                "No data found in the data folder! Please check that you selected the correct folder.\n"
                f"Currently selected folder: {base_dir.resolve()}"
            )
        all_test_names = {p: list(t.keys()) for p, t in all_tests.items()}
        index = (
            pd.DataFrame(all_test_names)
            .stack()
            .reset_index(level=0, drop=True)
            .reset_index()
            .rename(columns={"index": "participant", 0: "test"})
            .sort_values(["participant", "test"])
            .reset_index(drop=True)
        )
        return index.loc[~index["test"].isin(["full_session_part_1", "full_session_part_2"])]


class StairAmbulationHealthy2021Full(_StairAmbulationHealthy2021):
    """Dataclass representing the full sessions of Stair Ambulation dataset not split into individual tests.

    Compared to the PerTest dataset, this dataset does not have a separate test for each participant, but the `data`
    attribute will return the entire recording.
    Note, that the entire recording is still split into two parts.
    These are represented in the index of the dataset.

    Within each part of the recording, a number of tests were performed.
    You can extract the test list using `self.test_list`.
    Note, that between the tests participants were instructed to jump up and down three times to mark the start and
    end of each test.

    Parameters
    ----------
    data_folder
        The path to the data folder.
    memory
        The joblib memory object to cache the data loading.
    ignore_manual_session_markers
        Some datasets had some issues either at the start or end of the recording.
        Therefore, we decided to cut these regions by default.
        Anyway, this will only affect the recordings of three participants.
        If you want to keep these regions, set this to False.
    include_pressure_data
        Whether to load the raw pressure data recorded by the insole sensors.
        This will increase the load time and RAM requirements.
        Usually this is not needed unless you want to calculate your own gait events based on the pressure data.
        The precalculated gait events will still be available independent of this setting.
    include_hip_sensor
        Whether to load the raw data recorded by the hip sensor.
    include_baro_data
        Whether to load the raw data recorded by the barometer.

    """

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        memory: Memory = Memory(None),
        include_pressure_data: bool = False,
        include_hip_sensor: bool = False,
        include_baro_data: bool = False,
        ignore_manual_session_markers: bool = False,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.ignore_manual_session_markers = ignore_manual_session_markers
        super().__init__(
            data_folder,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            memory=memory,
            include_pressure_data=include_pressure_data,
            include_hip_sensor=include_hip_sensor,
            include_baro_data=include_baro_data,
        )

    def _get_full_session_start_end(self, participant: str, part: Literal["part_1", "part_2"]) -> Tuple[int, int]:
        session = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][f"full_session_{part}"]
        return int(session["start"]), int(session["end"])

    def _get_participant_and_part(self, error_name: str) -> Tuple[str, Literal["part_1", "part_2"]]:
        self.assert_is_single(None, error_name)
        participant, part = self.group
        return participant, part

    def _cut_data_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        participant, part = self._get_participant_and_part("_cut_to_region")
        if self.ignore_manual_session_markers is False:
            df = df.iloc[slice(*self._get_full_session_start_end(participant, part))]
        return df.reset_index(drop=True)

    def _cut_events_to_region(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        participant, part = self._get_participant_and_part("_cut_events_to_region")
        if self.ignore_manual_session_markers is False:
            session_start, session_end = self._get_full_session_start_end(participant, part)
            df = df.loc[(df["start"] >= session_start) & (df["end"] <= session_end)].copy()
            df.loc[:, numeric_cols] -= session_start
        return df

    @property
    def test_list(self) -> pd.DataFrame:
        """Get the list of all tests contained in the recording."""
        participant, part = self._get_participant_and_part("test_list")
        tests = pd.DataFrame(get_all_participants_and_tests(base_dir=self._data_folder_path)[participant]).T
        tests = tests[tests["part"] == part]
        tests = tests.drop(f"full_session_{part}").drop(columns=["part"])
        tests.index.name = "roi_id"
        tests = self._cut_events_to_region(tests, ["start", "end"])
        return tests

    def create_index(self) -> pd.DataFrame:
        # There are two parts per participant. We use parts as the second index column.
        all_participants = get_all_participants(base_dir=self._data_folder_path)
        if len(all_participants) == 0:
            raise ValueError(
                "No data found in the data folder! Please check that you selected the correct folder.\n"
                f"Currently selected folder: {self._data_folder_path.resolve()}"
            )
        return (
            pd.DataFrame(list(product(all_participants, ("part_1", "part_2"))), columns=["participant", "part"])
            .sort_values(["participant", "part"])
            .reset_index(drop=True)
        )
