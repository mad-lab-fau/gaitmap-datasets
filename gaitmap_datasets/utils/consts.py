"""Some common constants."""

SF_GYR = ["gyr_x", "gyr_y", "gyr_z"]
#: The default names of the Accelerometer columns in the sensor frame
SF_ACC = ["acc_x", "acc_y", "acc_z"]
#: The default names of all columns in the sensor frame
SF_COLS = [*SF_ACC, *SF_GYR]

#: The default names of the Gyroscope columns in the body frame
BF_GYR = ["gyr_pa", "gyr_ml", "gyr_si"]
#: The default names of the Accelerometer columns in the body frame
BF_ACC = ["acc_pa", "acc_ml", "acc_si"]
#: The default names of all columns in the body frame
BF_COLS = [*BF_ACC, *BF_GYR]

#: Sensor to body frame conversion for the left foot
FSF_FBF_CONVERSION_LEFT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (-1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (-1, "gyr_si"),
}

#: Sensor to body frame conversion for the right foot
FSF_FBF_CONVERSION_RIGHT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (-1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (1, "gyr_si"),
}
