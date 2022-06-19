from pathlib import Path

import numpy as np
import pandas as pd


def _fsr_calibration_path(base_dir: Path) -> Path:
    return base_dir / "fsr_calibrations"


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
    return load_fsr_calibration_data_from_path(_fsr_calibration_path(base_dir).joinpath("fsr_%d.csv" % fsr_id))


def non_inverting_opamp(r_fsr: float, r_ref: float, v_ref: float):
    """This methods simulates the characteristic of an ideal non-inverting opamp amplifier."""
    return v_ref * (1 + (r_ref / r_fsr))


def generate_conversion_function(fsr_calib_df: pd.DataFrame, r_ref: int, v_ref: float):
    """Generate a calibration function by fitting a 6-order polynom to the calibration measurement.

    IMPORTANT:
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
    """Apply ADC to Voltage conversion for NilsPod analog channels (valid for versions >= 0.18.0"""
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
