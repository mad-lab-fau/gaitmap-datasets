"""The core tpcp Dataset class for the Egait Stride Segementation Validation Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets._download import raise_manual_download_error
from gaitmap_datasets.egait_segmentation_validation_2014.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_original_segmented_stride_list,
    get_segmented_stride_list,
)


class EgaitSegmentationValidation2014(Dataset):
    """Egait stride segmentation validation 2014 dataset."""

    @staticmethod
    def download(_base_path: str | Path) -> Path:
        """Raise an error explaining how to obtain the dataset manually."""
        raise_manual_download_error(
            "EgaitSegmentationValidation2014",
            "This dataset is not publicly hosted for direct download.\n"
            "Please contact the data owner via https://www.mad.tf.fau.de/research/datasets/#collapse_18 and "
            "download the dataset manually.\n"
            "After unpacking it, pass the dataset root folder to `EgaitSegmentationValidation2014(...)`.",
        )

    def __init__(
        self,
        data_folder: str | Path | None = None,
        *,
        exclude_incomplete_participants: bool = False,
        memory: Memory = Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
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
    def data(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
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
    def segmented_stride_list_(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list with all strides labeled (straight, stairs, turns).

        This stride list differes from the stride list originally distributed with the dataset.
        It was generated later by relabling all strides, and not just the straight strides.
        """
        self.assert_is_single(None, "segmented_stride_list_")
        (
            cohort,
            test,
            participant,
        ) = self.group
        return get_segmented_stride_list(participant, cohort, test, base_dir=self._data_folder_path)

    @property
    def segmented_stride_list_original_(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the original segmented stride list provided with the dataset."""
        self.assert_is_single(None, "segmented_stride_list_original_")
        (
            cohort,
            test,
            participant,
        ) = self.group
        stride_list = get_original_segmented_stride_list(participant, cohort, test, base_dir=self._data_folder_path)
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
