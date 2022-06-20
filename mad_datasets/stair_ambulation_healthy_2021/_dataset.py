"""The core tpcp Dataset class for the Stair Ambulation dataset."""
from itertools import product
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from mad_datasets.stair_ambulation_healthy_2021.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_all_participants_and_tests,
)
from mad_datasets.utils.consts import SF_COLS


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

    def _cut_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cut the data to the region of interest."""
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
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        return Path(self.data_folder)

    @property
    def data(self) -> pd.DataFrame:
        """Get the IMU data of the dataset."""
        participant, part = self._get_participant_and_part("data")
        df = self._get_base_df(participant, part).drop(
            columns=["baro", *self._PRESSURE_COLUMNS], level=1, errors="ignore"
        )

        return self._cut_to_region(df)

    @property
    def baro_data(self) -> pd.DataFrame:
        """Get the barometer data of the dataset."""
        if self.include_baro_data is False:
            raise ValueError("The barometer data is not loaded. Please set `include_baro_data` to True.")
        participant, part = self._get_participant_and_part("baro_data")
        df = self._get_base_df(participant, part).drop(
            columns=[*SF_COLS, *self._PRESSURE_COLUMNS], level=1, errors="ignore"
        )
        return self._cut_to_region(df)

    @property
    def pressure_data(self) -> pd.DataFrame:
        """Get the pressure data of the dataset."""
        if self.include_pressure_data is False:
            raise ValueError("The pressure data is not loaded. Please set `include_pressure_data` to True.")
        participant, part = self._get_participant_and_part("pressure_data")
        df = self._get_base_df(participant, part).drop(columns=["baro", *SF_COLS], level=1, errors="ignore")
        return self._cut_to_region(df)


class StairAmbulationHealthy2021PerTest(_StairAmbulationHealthy2021):
    """Dataset class representing the Stair Ambulation dataset."""

    def _get_participant_and_part(self, error_name: str) -> Tuple[str, Literal["part_1", "part_2"]]:
        self.assert_is_single(None, error_name)
        participant, test = self.index.iloc[0]
        part = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][test]["part"]
        return participant, part

    def _cut_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        # We assume that the df we get is from the correct participant and part
        participant, test = self.index.iloc[0]
        test = get_all_participants_and_tests(base_dir=self._data_folder_path)[participant][test]
        df = df.iloc[test["start"] : test["end"]]
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

    Parameters
    ----------
    data_folder
        The path to the data folder.
    memory
        The joblib memory object to cache the data loading.
    ignore_manual_session_markers
        Some datasets had some issues either at the start or end of the recording.
        Therefore, we decided to cut these regions by default.
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
        participant, part = self.index.iloc[0]
        return participant, part

    def _cut_to_region(self, df: pd.DataFrame) -> pd.DataFrame:
        participant, part = self._get_participant_and_part("_cut_to_region")
        if self.ignore_manual_session_markers is False:
            df = df.iloc[slice(*self._get_full_session_start_end(participant, part))]
        return df

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
