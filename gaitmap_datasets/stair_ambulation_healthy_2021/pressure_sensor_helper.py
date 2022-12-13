"""Helpers to load and calibrate the pressure insoles."""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from gaitmap_datasets.utils.consts import BF_GYR
from gaitmap_datasets.utils.event_detection import detect_min_vel


def _fsr_calibration_path(base_dir: Path) -> Path:
    """Get the relative path to the fsr-calibration data."""
    return base_dir / "calibrations" / "fsr_calibrations"


def load_fsr_calibration_data_from_path(path: Path):
    """Load fsr-calibration data from path and sort / average it in 0.5kg bins."""
    calib_data_df = pd.read_csv(path, header=0)
    data = []

    # TODO: This can likely be made much more efficient
    # bin data in 0.5kg steps and average measurements within those bins (bins form 0.5kg to 20kg)
    for weight in np.arange(40) + 1:
        weight = weight * 500
        res = (
            calib_data_df[(calib_data_df["weight"] >= weight - 250) & (calib_data_df["weight"] <= weight + 250)]
            .mean()
            .resistance
        )
        data.append([weight, res])

    data = np.array(data)

    # create new dataframe to store all converted data
    tmp_df = pd.DataFrame(data, columns=["weight_g", "res_ohm"])
    tmp_df["weight_kg"] = tmp_df["weight_g"] / 1000
    tmp_df = tmp_df.set_index("weight_kg")
    return tmp_df


def load_fsr_calibration_data_from_id(fsr_id: int, base_dir: Path):
    """Load fsr-calibration data from fsr-id and sort / average it in 0.5kg bins."""
    return load_fsr_calibration_data_from_path(_fsr_calibration_path(base_dir).joinpath(f"fsr_{fsr_id}.csv"))


def non_inverting_opamp(r_fsr: float, r_ref: float, v_ref: float):
    """Simulate the characteristic of an ideal non-inverting opamp amplifier."""
    return v_ref * (1 + (r_ref / r_fsr))


def generate_conversion_function(fsr_calib_df: pd.DataFrame, r_ref: int, v_ref: float):
    """Generate a calibration function by fitting a 6-order polynom to the calibration measurement.

    .. warning::
        r_ref and v_ref must correspond to the actual configuration of the FSR extension board during data
        recording!

    """
    # simulate opamp circuit
    data = [
        non_inverting_opamp(float(r_fsr), float(r_ref), float(v_ref)) for r_fsr in fsr_calib_df["res_ohm"].to_numpy()
    ]
    # create polynomial fit for calibration function
    return np.poly1d(np.polyfit(data, fsr_calib_df.index.to_numpy(), 6))


def factory_calibrate_analog_data(analog_data: np.ndarray):
    """Apply ADC to Voltage conversion for NilsPod analog channels (valid for versions >= 0.18.0)."""
    return np.asarray(analog_data) * (2.4 / 16384)


def calibrate_analog_data(analog_data: np.ndarray, fsr_id_dict, v_ref: float = 0.1, base_dir=None):
    """Apply individual calibration functions for fsr sensors."""
    # convert raw data into volts

    data = factory_calibrate_analog_data(analog_data)

    for i, position in enumerate(["toe", "mth", "heel"]):
        fsr_id = fsr_id_dict[position]["id"]
        fsr_calib_df = load_fsr_calibration_data_from_id(fsr_id, base_dir).dropna()

        conv_func_toe = generate_conversion_function(fsr_calib_df, fsr_id_dict[position]["r_ref"], v_ref)
        data[:, i] = conv_func_toe(data[:, i])

    return data


def _stride_list_to_min_vel_list(dataset_bf, segmented_stride_list, min_vel_search_window_s, sampling_rate_hz):
    """Detect at least one zupt update per stride (similar to min_vel event detection from gaitmap)."""
    window_size = int(np.round(min_vel_search_window_s * sampling_rate_hz))

    # find all min_vel events for the segmented strides
    min_vel_list = []
    for stride_start_end in segmented_stride_list[["start", "end"]].dropna().to_numpy():
        gyr_bf_stride = (
            dataset_bf[BF_GYR].reset_index(drop=True).iloc[int(stride_start_end[0]) : int(stride_start_end[1])]
        )

        # try to only look for minimal energy windows after the swing peak (we know that ground contact (= min velocity/
        # midstance) has to be after the swing phase)
        gyr_ml_max_idx = np.argmax(gyr_bf_stride["gyr_ml"])

        # look at least from mid of stride to end
        gyr_ml_max_idx = np.min([len(gyr_bf_stride) // 2, gyr_ml_max_idx])

        try:
            min_vel_list.append(
                detect_min_vel(gyr_bf_stride.to_numpy()[gyr_ml_max_idx:], window_size)
                + gyr_ml_max_idx
                + stride_start_end[0]
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to detect min-vel for {stride_start_end} ({e})")

    # now we need to add "pre_min_vel" events for those strides who have no previous stride (aka strides which do not
    # share their start index with the end index of another stride)

    stride_ids_with_no_prev_stride = (
        np.argwhere(
            (segmented_stride_list["end"] - segmented_stride_list["start"][1:].reset_index(drop=True))
            .dropna()
            .to_numpy()
            != 0
        )
        + 1
    ).flatten()

    stride_ids_with_no_prev_stride = np.insert(
        stride_ids_with_no_prev_stride, 0, 0
    )  # the first stride always needs a pre min_vel event!

    pre_min_vel_list = []

    for stride_id in stride_ids_with_no_prev_stride:
        stride = segmented_stride_list.iloc[stride_id]

        start = np.max([0, int(stride.start - (stride.end - stride.start))])
        end = int(stride.start)

        pre_min_vel_list.append(detect_min_vel(dataset_bf[BF_GYR].iloc[start:end], window_size) + start)

    stride_ids_with_no_prev_stride = np.append(stride_ids_with_no_prev_stride, len(segmented_stride_list))

    # now put everything together
    min_vel_stride_list = pd.DataFrame()

    for i in np.arange(len(stride_ids_with_no_prev_stride) - 1):
        sl = [pre_min_vel_list[i]] + min_vel_list[
            stride_ids_with_no_prev_stride[i] : stride_ids_with_no_prev_stride[i + 1]
        ]

        min_vel_stride_list = (
            pd.concat([min_vel_stride_list, pd.DataFrame(np.column_stack([sl[:-1], sl[1:]]), columns=["start", "end"])])
            .astype(int)
            .reset_index(drop=True)
        )

    return min_vel_stride_list


def convert_segmented_stride_list_to_min_vel_list(
    dataset_bf,
    segmented_stride_list,
    sampling_rate_hz,
    min_vel_search_window_sec=0.2,
):
    """Convert segmented stride list to min_vel list by detecting mid-stance events based on the Gyro data."""
    min_vel_list = {}
    for sensor in ["left_sensor", "right_sensor"]:

        min_vel_list_single_sensor = _stride_list_to_min_vel_list(
            dataset_bf[sensor], segmented_stride_list[sensor], min_vel_search_window_sec, sampling_rate_hz
        )
        min_vel_list_single_sensor.index.name = "s_id"
        # add all columns required for gaitmap stride lists
        min_vel_list_single_sensor["ic"] = np.nan
        min_vel_list_single_sensor["tc"] = np.nan
        min_vel_list_single_sensor["min_vel"] = min_vel_list_single_sensor["start"]
        min_vel_list_single_sensor["pre_ic"] = np.nan

        min_vel_list[sensor] = min_vel_list_single_sensor

    return min_vel_list


class PressureEventDetection:
    """Find gait events from FSR sensor pressure data."""

    max_threshold_kg: float
    threshold_increment_kg: float
    min_pressure_search_window_s: float

    min_vel_event_list_: Dict[str, pd.DataFrame]
    sampling_rate_hz: float
    stride_list: pd.DataFrame

    def __init__(
        self,
        max_threshold_kg: float = 10,
        threshold_increment_kg: float = 0.25,
    ):
        self.max_threshold_kg = max_threshold_kg
        self.threshold_increment_kg = threshold_increment_kg

    def detect(self, dataset: pd.DataFrame, stride_list: Dict[str, pd.DataFrame], threshold_kg: float):
        """Detect gait events from FSR sensor pressure data."""
        self.min_vel_event_list_ = {}
        for sensor in ["left_sensor", "right_sensor"]:
            self.min_vel_event_list_[sensor] = self._detect_single_dataset(
                dataset[sensor],
                stride_list[sensor][["start", "end"]].dropna(),
                threshold_kg,
            )
        return self

    def _detect_single_dataset(
        self,
        dataset: pd.DataFrame,
        stride_list: pd.DataFrame,
        threshold_kg: float,
    ) -> pd.DataFrame:
        """Detect IC and TC events from pressure sensor data using some simple thresholding.

        This function heavily relies on a correct min_vel_stride_list!
        """
        ped_stride_list = []

        if self.max_threshold_kg > 15:
            raise ValueError(f"Max threshold_kg must be <= 15kg but was {self.max_threshold_kg}.")

        if threshold_kg >= self.max_threshold_kg:
            raise ValueError(f"threshold_kg must be < {self.max_threshold_kg} but was {threshold_kg}.")

        for _, stride in stride_list.astype(int).iterrows():
            total_force = dataset["total_force"].iloc[stride.start : stride.end].to_numpy()

            # remove baseline offset
            total_force = total_force - np.min(total_force)

            th = threshold_kg
            zero_crossings = np.array([])

            # search for threshold crossings until we found exactly two crossings, gradually increase threshold in
            # "threshold_increment_kg" steps on the way
            # if we don´t find two crossings until the threshold reaches max_threshold_kg the pressure event detection
            # failed!
            while len(zero_crossings) != 2:
                zero_crossings = np.where(np.diff(np.signbit(total_force - th)))[0]
                th = th + self.threshold_increment_kg
                if th > self.max_threshold_kg:
                    zero_crossings = np.array([None, None])
                    break

            tc = zero_crossings[0]
            ic = zero_crossings[1]

            # If the event detection fails, we remove the stride
            if tc is None or ic is None:
                continue

            tc = int(stride.start + tc)
            ic = int(stride.start + ic)

            ped_stride_list.append(
                {"start": stride.start, "end": stride.end, "ic": ic, "tc": tc, "min_vel": stride.start}
            )

        stride_list = pd.DataFrame.from_records(ped_stride_list).rename_axis("s_id")
        stride_list["pre_ic"] = _build_pre_ic_list(stride_list)  # pylint: disable=unsupported-assignment-operation

        return stride_list


def _build_pre_ic_list(min_vel_stride_list) -> np.ndarray:
    """Check where a pre_ic exists, if not pre-ic exists fill list with nan."""
    stride_ids_with_no_prev_stride = (
        np.argwhere(
            (min_vel_stride_list["end"] - min_vel_stride_list["start"][1:].reset_index(drop=True)).dropna().to_numpy()
            != 0
        )
        + 1
    ).flatten()
    stride_ids_with_no_prev_stride = np.insert(
        stride_ids_with_no_prev_stride, 0, 0
    )  # the first stride always needs a pre min_vel event!
    stride_ids_with_no_prev_stride = np.append(stride_ids_with_no_prev_stride, len(min_vel_stride_list))

    pre_ic_list = np.array([])

    for i in np.arange(len(stride_ids_with_no_prev_stride) - 1):
        sl = min_vel_stride_list[stride_ids_with_no_prev_stride[i] : stride_ids_with_no_prev_stride[i + 1]]["ic"][:-1]
        sl = np.append(np.nan, sl)

        pre_ic_list = np.append(pre_ic_list, sl)

    return pre_ic_list
