"""Dataset-specific path validation helpers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetPathCheck:
    """Definition of the expected root structure of a dataset."""

    dataset_name: str
    expected_contents: Sequence[Sequence[str]]
    expected_root_description: str
    common_fix: Optional[str] = None


def _format_expected_contents(expected_contents: Iterable[Sequence[str]]) -> str:
    formatted = []
    for group in expected_contents:
        if len(group) == 1:
            formatted.append(f"`{group[0]}`")
        else:
            formatted.append("one of " + ", ".join(f"`{entry}`" for entry in group))
    return ", ".join(formatted)


def validate_dataset_path(path: Path, check: DatasetPathCheck) -> Path:
    """Validate that a path looks like the root folder of a dataset."""
    path = Path(path)

    missing = [
        group for group in check.expected_contents if not any((path / candidate).exists() for candidate in group)
    ]
    if not missing:
        return path

    message = [
        f"{check.dataset_name} could not find the expected dataset structure.",
        f"Selected folder: {path.resolve()}",
        f"Expected the dataset root folder to contain {check.expected_root_description}.",
        f"Missing from this folder: {_format_expected_contents(missing)}.",
    ]
    if check.common_fix:
        message.append(check.common_fix)
    raise ValueError("\n".join(message))


EGAIT_PARAMETER_VALIDATION_2013_CHECK = DatasetPathCheck(
    dataset_name="EgaitParameterValidation2013",
    expected_contents=(
        ("ValidationRawData",),
        ("GoldStandard_StrideBorders",),
        ("GoldStandard_GaitRite",),
    ),
    expected_root_description=(
        "the `ValidationRawData`, `GoldStandard_StrideBorders`, and `GoldStandard_GaitRite` folders"
    ),
    common_fix="Pass the dataset root folder itself, not one of the validation or gold-standard subfolders.",
)

EGAIT_SEGMENTATION_VALIDATION_2014_CHECK = DatasetPathCheck(
    dataset_name="EgaitSegmentationValidation2014",
    expected_contents=(
        ("Controls_GoldStandard_StrideBorders",),
        ("PDs_GoldStandard_StrideBorders",),
        ("Geriatrics_GoldStandard_StrideBorders",),
        ("calibrationFiles",),
    ),
    expected_root_description=(
        "the cohort-specific `*_GoldStandard_StrideBorders` folders and the `calibrationFiles` folder"
    ),
    common_fix="Pass the dataset root folder itself, not one of the cohort-specific raw-data or annotation folders.",
)

SENSOR_POSITION_COMPARISON_2019_CHECK = DatasetPathCheck(
    dataset_name="SensorPositionComparison2019",
    expected_contents=(
        ("data",),
        ("calibrations",),
    ),
    expected_root_description="the `data` and `calibrations` folders",
    common_fix="Pass the dataset root folder itself, not the nested `data` folder.",
)

EGAIT_ADIDAS_2014_CHECK = DatasetPathCheck(
    dataset_name="EgaitAdidas2014",
    expected_contents=(
        ("data",),
        ("calibrations",),
    ),
    expected_root_description="the `data` and `calibrations` folders",
    common_fix="Pass the dataset root folder itself, not the nested `data` folder.",
)

PYSHOE_2019_CHECK = DatasetPathCheck(
    dataset_name="PyShoe2019",
    expected_contents=(
        ("data",),
        ("data/vicon/processed",),
        ("data/hallway",),
        ("data/stairs/test",),
    ),
    expected_root_description=("the `data` folder with the `vicon/processed`, `hallway`, and `stairs/test` subfolders"),
    common_fix="Pass the PyShoe dataset root folder, not the nested `data`, `hallway`, `vicon`, or `stairs` folder.",
)

STAIR_AMBULATION_HEALTHY_2021_CHECK = DatasetPathCheck(
    dataset_name="StairAmbulationHealthy2021",
    expected_contents=(
        ("healthy",),
        ("calibrations",),
    ),
    expected_root_description="the `healthy` and `calibrations` folders",
    common_fix="Pass the dataset root folder itself, not the nested `healthy` folder.",
)

KLUGE_2017_CHECK = DatasetPathCheck(
    dataset_name="Kluge2017",
    expected_contents=(
        ("raw_data_export.mat",),
        ("subjects.csv",),
        ("stride_parameters.csv",),
    ),
    expected_root_description="the files `raw_data_export.mat`, `subjects.csv`, and `stride_parameters.csv`",
    common_fix="Pass the folder containing the exported MATLAB data files, not a parent or unrelated download folder.",
)
