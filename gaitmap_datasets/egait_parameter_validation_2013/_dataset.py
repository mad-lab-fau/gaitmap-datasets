"""The core tpcp Dataset class for the Egait Parameter Validation Dataset."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets.egait_parameter_validation_2013.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_gaitrite_parameters,
    get_segmented_stride_list,
)


class EgaitParameterValidation2013(Dataset):
    """Egait parameter validation 2013 dataset."""

    def __init__(
        self,
        data_folder: Optional[Union[str, Path]] = None,
        *,
        use_alternative_calibrations: bool = True,
        memory: Memory = Memory(None),
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None
    ):
        self.data_folder = data_folder
        self.use_alternative_calibrations = use_alternative_calibrations
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
        data = self.memory.cache(get_all_data_for_participant)(
            self.group, use_alternative_calibrations=self.use_alternative_calibrations, base_dir=self._data_folder_path
        )
        final_data = {}
        for k, v in data.items():
            v.index /= self.sampling_rate_hz
            v.index.name = "time [s]"
            final_data[k] = v
        return final_data

    @property
    def segmented_stride_list_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list."""
        self.assert_is_single(None, "segmented_stride_list_")
        return get_segmented_stride_list(self.group, base_dir=self._data_folder_path)

    @property
    def gaitrite_parameters_(self) -> Dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the gaitrite parameters."""
        self.assert_is_single(None, "gaitrite_parameters_")
        return get_gaitrite_parameters(self.group, base_dir=self._data_folder_path)

    def create_index(self) -> pd.DataFrame:
        """Create index."""
        return pd.DataFrame(get_all_participants(base_dir=self._data_folder_path), columns=["participant"])
