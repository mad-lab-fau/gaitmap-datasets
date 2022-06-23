"""General helper to load and manage the dataset.

These are the logic behind the dataset implementation, but can also be used independently.
"""
from pathlib import Path
from typing import Literal, Optional, List, Tuple

Cohorts = Literal["control", "pd", "geriatric"]
Tests = Literal["4x10m", "free_walk"]

_test_rename_dict = {"4x10m": "", "free_walk": "_4MW"}
_cohort_rename_dict = {"control": "Control", "pd": "PD", "geriatric": "Geriatric"}


def _raw_data_folder(base_dir: Path, cohort: Cohorts, test: Tests) -> Path:
    """Return the relative path to the participant subfolder."""
    return base_dir / f"{_cohort_rename_dict[cohort]}s_RawDataValidation{_test_rename_dict[test]}"


def _reference_stride_borders_folder(base_dir: Path, cohort: Cohorts, test: Tests) -> Path:
    """Return the relative path to the reference stride borders subfolder."""
    return base_dir / f"{_cohort_rename_dict[cohort]}s_GoldStandard_StrideBorders{_test_rename_dict[test]}"


def _extract_participant_id(file_name: str, test: Tests) -> str:
    """Extract the participant id from the file name."""
    if test == "4x10m":
        return file_name.split("_")[0].upper()
    elif test == "free_walk":
        for foot in ["left", "right"]:
            if f"{foot}foot" in file_name.lower():
                return file_name.lower().split(f"{foot}foot")[0].upper()
    raise ValueError("Invalid file format")


def get_all_participants(*, base_dir: Optional[Path] = None) -> List[Tuple[str, str, str]]:
    """Get the folder names of all participants."""
    all_participants = []
    for cohort in ["control", "pd", "geriatric"]:
        for test in ["4x10m", "free_walk"]:
            stride_border_folder = _reference_stride_borders_folder(base_dir, cohort, test)
            for file_name in stride_border_folder.glob("*.txt"):
                if file_name.is_file():
                    participant_id = _extract_participant_id(file_name.name, test)
                    all_participants.append((cohort, test, participant_id))
    return list(set(all_participants))
