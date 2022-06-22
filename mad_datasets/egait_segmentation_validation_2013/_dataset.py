"""The core tpcp Dataset class for the Egait Stride Segementation Validation Dataset."""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from joblib import Memory
from tpcp import Dataset


class EgaitSegmentationValidation2013(Dataset):
    """Egait stride segmentation validation 2013 dataset."""

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
    def sampling_rate_hz(self) -> float:
        """Get the sampling rate of the IMUs."""
        return 102.4

    @property
    def _data_folder_path(self) -> Path:
        """Get the path to the data folder as Path object."""
        return Path(self.data_folder)
