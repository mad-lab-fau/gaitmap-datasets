from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tpcp import Dataset

from mad_datasets.stair_ambulation_healthy_2021.helper import get_all_participants_and_tests

# TODO: Split it into test and full data


class StairAmbulationHealthy2021(Dataset):
    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_folder = data_folder
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """The sampling rate of the IMUs."""
        return 204.8

    # def _get_base_df(self, participant: str, test: str) -> pd.DataFrame:

    def create_index(self) -> pd.DataFrame:
        base_dir = Path(self.data_folder)
        all_tests = get_all_participants_and_tests(base_dir)
        if len(all_tests) == 0:
            raise ValueError(
                "No data found in the data folder! Please check that you selected the correct folder.\n"
                f"Currently selected folder: {base_dir.resolve()}"
            )
        all_test_names = {p: list(t.keys()) for p, t in all_tests.items()}
        return (
            pd.DataFrame(all_test_names)
            .stack()
            .reset_index(level=0, drop=True)
            .reset_index()
            .rename(columns={"index": "participant", 0: "test"})
            .sort_values(["participant", "test"])
            .reset_index(drop=True)
        )
