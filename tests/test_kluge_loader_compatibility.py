"""Exercise Kluge loader preprocessing at the MATLAB file boundary."""

import numpy as np
import pandas as pd
import pytest

from gaitmap_datasets import Kluge2017
from gaitmap_datasets.kluge_2017 import helper as kluge_helper


def _mat_cell(value):
    cell = np.empty((), dtype=object)
    cell[()] = value
    return cell


@pytest.fixture
def kluge_recording(tmp_path, monkeypatch):
    """Mock the MATLAB file boundary while exercising all loader preprocessing."""
    times = np.arange(602) / 100
    columns = ["Time", "L_Foot_Ground_Angle", "R_Foot_Ground_Angle"]
    values = [times, np.zeros(len(times)), np.zeros(len(times))]
    for marker in ["Ankle", "Foot Tip"]:
        for foot, visible in [("left", (times >= 1) & (times < 5)), ("right", (times >= 2) & (times < 6))]:
            for metric, data in [("X", times), ("Y", times * 0), ("Z", times * 0), ("v(X)", visible.astype(float))]:
                columns.append(f"{marker}, {foot} {metric}")
                values.append(data)
    camera = _mat_cell({"data": np.column_stack(values), "columnNames": columns})
    tests = [{"Name": "4x10_normal", "Start": 1.5, "End": 5.5, "Duration": 4.0}]
    events = []
    for foot, starts in [("L", [1.5, 2.5, 3.5, 4.5, 5.5]), ("R", [2.0, 3.0, 4.0, 5.0, 6.0])]:
        for stride, start in enumerate(starts):
            for event, offset in [("HS", 0), ("TO", 0.6), ("HO", 0.5)]:
                events.append({"Name": f"{foot}_{event}", "Start": start + offset, "Reference_HS": f"{foot}{stride}"})
    camera_events = {"Tests": _mat_cell(_mat_cell([tests])), "Gait_Events": _mat_cell(_mat_cell([events]))}
    sensor_data = _mat_cell(
        [
            _mat_cell(
                {
                    "sensorPosition": foot,
                    "columnNames": ["Time", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
                    "data": np.column_stack([times, *[np.full(len(times), value) for value in [1, 2, 3, 4, 5, 6]]]),
                }
            )
            for foot in ["LeftFoot", "RightFoot"]
        ]
    )
    measurements = np.empty(
        2, dtype=[(field, object) for field in ["id", "camera_data", "camera_events", "sensor_data"]]
    )
    for i in range(2):
        measurements[i] = (f"GA{i}", camera, camera_events, sensor_data)
    monkeypatch.setattr(kluge_helper.sio, "loadmat", lambda *_args, **_kwargs: {"rawdata": measurements})
    kluge_helper.get_data.cache_clear()
    (tmp_path / "raw_data_export.mat").touch()
    (tmp_path / "stride_parameters.csv").touch()
    pd.DataFrame({"patid": ["P01"], "retest": [1], "ga": ["GA0"], "patient": ["Healthy"]}).to_csv(
        tmp_path / "subjects.csv", index=False
    )
    yield Kluge2017(
        tmp_path,
        subset_index=pd.DataFrame({"patient": [False], "participant": ["P01"], "repetition": [1], "speed": ["normal"]}),
    )
    kluge_helper.get_data.cache_clear()


def test_kluge_loader_keeps_per_foot_visibility_and_coordinates(kluge_recording):
    marker = kluge_recording.marker_position_
    expected_columns = pd.MultiIndex.from_tuples(
        [
            (foot, name, metric)
            for name in ["ankle", "foot_tip"]
            for foot in ["left", "right"]
            for metric in ["pos_x", "pos_y", "pos_z", "vel_x"]
        ],
        names=["foot", "marker", "metric"],
    )
    pd.testing.assert_index_equal(marker.columns, expected_columns)
    assert marker.loc[0.5, ("left", "ankle", "pos_x")] == pytest.approx(2.0)
    assert pd.isna(marker.loc[0.5, ("right", "ankle", "pos_x")])
    assert marker.loc[1.5, ("right", "ankle", "pos_x")] == pytest.approx(3.0)
    assert pd.isna(marker.loc[3.5, ("left", "ankle", "pos_x")])
    assert marker.loc[3.5, ("right", "ankle", "pos_x")] == pytest.approx(5.0)
    assert list(kluge_recording.mocap_events_["left"]["start"]) == [100, 200]
    assert list(kluge_recording.mocap_events_["right"]["start"]) == [150, 250]
    assert kluge_recording.data["left_sensor"]["acc_x"].iloc[0] == pytest.approx(-9.81)
