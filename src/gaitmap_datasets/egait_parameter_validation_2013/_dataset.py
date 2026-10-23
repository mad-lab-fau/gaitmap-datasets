"""The core tpcp Dataset class for the Egait Parameter Validation Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from joblib import Memory
from tpcp import Dataset

from gaitmap_datasets._config import get_dataset_path
from gaitmap_datasets._download import raise_manual_download_error
from gaitmap_datasets.egait_parameter_validation_2013.helper import (
    get_all_data_for_participant,
    get_all_participants,
    get_gaitrite_parameters,
    get_segmented_stride_list,
)

_EXPECTED_ROOT_CONTENTS = (
    "ValidationRawData",
    "GoldStandard_StrideBorders",
    "GoldStandard_GaitRite",
)


def _validate_dataset_path(path: Path) -> Path:
    missing = [entry for entry in _EXPECTED_ROOT_CONTENTS if not (path / entry).exists()]
    if not missing:
        return path

    raise ValueError(
        "EgaitParameterValidation2013 could not find the expected dataset structure.\n"
        f"Selected folder: {path.resolve()}\n"
        "Expected the dataset root folder to contain the `ValidationRawData`, `GoldStandard_StrideBorders`, "
        "and `GoldStandard_GaitRite` folders.\n"
        f"Missing from this folder: {', '.join(f'`{entry}`' for entry in missing)}.\n"
        "Pass the dataset root folder itself, not one of the validation or gold-standard subfolders."
    )


class EgaitParameterValidation2013(Dataset):
    """Egait parameter validation 2013 dataset.

    Parameters
    ----------
    data_folder
        The path to the data folder. If None, the path from the config is used.
    include_bad_data
        A couple of participants appear to have bad data.
        If True, these participants are included (as in the original publication).
        If False, these participants are excluded (recommended).
    use_alternative_calibrations
        The original calibration files showed a big acc offset.
        Therefore, we recommend to use the alternative calibration files.
        If True, the alternative calibration files are used.
        If False, the original calibration files are used (as in the original publication).

    """

    @staticmethod
    def download(_base_path: str | Path) -> Path:
        """Raise an error explaining how to obtain the dataset manually."""
        raise_manual_download_error(
            "EgaitParameterValidation2013",
            "This dataset is not publicly hosted for direct download.\n"
            "Please contact the data owner via https://www.mad.tf.fau.de/research/datasets/#collapse_18 and "
            "download the dataset manually.\n"
            "After unpacking it, pass the dataset root folder to `EgaitParameterValidation2013(...)`.",
        )

    def __init__(
        self,
        data_folder: str | Path | None = None,
        *,
        include_bad_data: bool = False,
        use_alternative_calibrations: bool = True,
        memory: Memory = Memory(None),
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
    ):
        self.data_folder = data_folder
        self.include_bad_data = include_bad_data
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
            return _validate_dataset_path(get_dataset_path(Path(__file__).parent.name))
        return _validate_dataset_path(Path(self.data_folder))

    @property
    def data(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the imu data."""
        self.assert_is_single(None, "data")
        data = self.memory.cache(get_all_data_for_participant)(
            self.group_label.participant,
            use_alternative_calibrations=self.use_alternative_calibrations,
            base_dir=self._data_folder_path,
        )
        final_data = {}
        for k, v in data.items():
            v.index /= self.sampling_rate_hz
            v.index.name = "time [s]"
            final_data[k] = v
        return final_data

    @property
    def segmented_stride_list_(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the segmented stride list."""
        self.assert_is_single(None, "segmented_stride_list_")
        return get_segmented_stride_list(self.group_label.participant, base_dir=self._data_folder_path)

    @property
    def gaitrite_parameters_(self) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Get the gaitrite parameters."""
        self.assert_is_single(None, "gaitrite_parameters_")
        return get_gaitrite_parameters(self.group_label.participant, base_dir=self._data_folder_path)

    def create_index(self) -> pd.DataFrame:
        """Create index."""
        return pd.DataFrame(
            get_all_participants(include_bad_data=self.include_bad_data, base_dir=self._data_folder_path),
            columns=["participant"],
        )
