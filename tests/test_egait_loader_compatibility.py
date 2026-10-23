"""Exercise loader properties without the restricted full datasets."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from gaitmap_datasets import EgaitParameterValidation2013


@pytest.fixture
def egait_recording(tmp_path):
    """A real sample IMU recording with small reference tables."""
    samples = Path(__file__).parent / "egait_validation_2013_test_data"
    for name in ["ValidationRawData", "GoldStandard_StrideBorders", "GoldStandard_GaitRite"]:
        (tmp_path / name).mkdir()
    for foot, calibration in [("left", "A917.csv"), ("right", "A6DF.csv")]:
        shutil.copyfile(samples / f"P115_E4_{foot}.dat", tmp_path / "ValidationRawData" / f"P115_E4_{foot}.dat")
        shutil.copyfile(samples / calibration, tmp_path / calibration)
        header = "fixture header\n" * 8
        (tmp_path / "GoldStandard_StrideBorders" / f"P115_E4_{foot}.txt").write_text(
            header + "Start,End\n0,100\n100,200\n"
        )
        (tmp_path / "GoldStandard_GaitRite" / f"P115_E4_{foot}.txt").write_text(header + "StrideLength\n120\n130\n")
    return EgaitParameterValidation2013(tmp_path, use_alternative_calibrations=False)[0]


@pytest.mark.parametrize("attribute", ["data", "segmented_stride_list_", "gaitrite_parameters_"])
def test_egait_2013_properties_load_the_selected_participant(egait_recording, attribute):
    result = getattr(egait_recording, attribute)
    assert set(result) == {"left_sensor", "right_sensor"}
    if attribute == "gaitrite_parameters_":
        assert list(result["left_sensor"]["stride_length"]) == pytest.approx([1.2, 1.3])
    elif attribute == "segmented_stride_list_":
        assert list(result["left_sensor"]["end"]) == [100, 200]
    else:
        assert len(result["left_sensor"]) > 0
        assert result["left_sensor"].index[1] == pytest.approx(1 / 102.4)
