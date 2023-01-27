"""The core tpcp Dataset class for the Egait Stride Segementation Validation Dataset."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.egait_segmentation_validation_2014.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_segmented_stride_list,
)


class EgaitSegmentationValidation2014(Dataset):
    """Egait stride segmentation validation 2014 dataset."""

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        exclude_incomplete_participants: bool = False,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None
    ):
        self.exclude_incomplete_participants = exclude_incomplete_participants
        self.data_folder = data_folder
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return 102.4

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
        cohort, test, participant = self.group
        data = self.memory.cache(get_all_data_for_participant)(
            participant, cohort, test, base_dir=self._data_folder_path
        )
        final_data = {}
        for k, v in data.items():
            if v is None:
                # For one participant the data from one sensor is missing.
                # For the output, we ignore it and not include it in the output.
                continue
            v.index /= self.sampling_rate_hz
            v.index.name = "time [s]"
            final_data[k] = v
        return final_data

    @property
    def segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list."""
        self.assert_is_single(None, "segmented_stride_list_")
        (
            cohort,
            test,
            participant,
        ) = self.group
        stride_list = get_segmented_stride_list(participant, cohort, test, base_dir=self._data_folder_path)
        # For one participant the data from one sensor is missing.
        # For the output, we ignore it and not include it in the output.
        return {k: v for k, v in stride_list.items() if v is not None}

    def create_index(self) -> pd.DataFrame:
        """Create index."""
        index = pd.DataFrame(
            get_all_participants(base_dir=self._data_folder_path), columns=["cohort", "test", "participant"]
        )
        if self.exclude_incomplete_participants:
            index = index[index.participant != "GA214026"].copy()
        return index
