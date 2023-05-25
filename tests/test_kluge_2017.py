import pandas as pd

from gaitmap_datasets.kluge_2017 import Kluge2017
from gaitmap_datasets.kluge_2017.helper import _interleave_foot_events


def test_number_valid_strides():
    dataset = Kluge2017()
    per_dp = [sum(len(v) for v in dp.mocap_events_.values()) for dp in dataset]

    assert sum(per_dp) == 1166


class TestInterleaveFootEvents:
    def test_simple_case(self):
        events = pd.DataFrame(range(10), index=range(10))
        events["foot"] = ["left", "right"] * 5
        events = events.set_index("foot", append=True).reorder_levels(["foot", None])[0]

        expected = events.copy().shift(-1)

        result = _interleave_foot_events(events, ("left", "right"))

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_missing_hs(self):
        events = pd.DataFrame(range(10), index=range(10))
        events["foot"] = ["left", "right"] * 5
        events = events.set_index("foot", append=True).reorder_levels(["foot", None])[0]

        # This should be the second right HS
        events = events.drop(("right", 3))

        expected = events.copy().shift(-1)
        expected.iloc[[1, 2]] = pd.NA

        result = _interleave_foot_events(events, ("left", "right"))

        pd.testing.assert_series_equal(result, expected, check_names=False)
