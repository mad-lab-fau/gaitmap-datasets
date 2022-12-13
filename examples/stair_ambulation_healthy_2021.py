r"""
StairAmbulationHealthy2021 - A Stride Segmentation and Event Detection dataset with focus on stairs
===================================================================================================

The dataset can be downloaded from here:

.. note:: The dataset only contains the healthy participants of the full dataset presented in the paper!

We provide two `tpcp.Dataset` classes to access the data:

1. :class:`gaitmap_datasets.stair_ambulation_healthy_2021.StairAmbulationHealthy2021PerTest`: This class allows to access
   all data and events for each of the performed gait tests individually.
2. :class:`gaitmap_datasets.stair_ambulation_healthy_2021.StairAmbulationHealthy2021Full`: This class allows to access the
   entire recordings for each participant (two recordings per participant) independently of the performed gait tests.

In the following we will show the usage of both classes and the data that is contained within.
"""
# %%
# .. warning:: For this example to work, you need to modify the dataset path in the following line to point to the
#              location of the data on your machine.
from pathlib import Path

dataset_path = Path("/home/arne/Documents/repos/work/datasets/stair_ambulation_dataset/")

# %%
# StairAmbulationHealthy2021PerTest
# =================================
# First we can simple create an instance of the dataset class and directly see the contained data points.
# Note, that we will enable the loading of all available data (pressure, baro, and hip sensor).
# You might want to disable that, to reduce the RAM usage and speed up the data loading.
from joblib import Memory

from gaitmap_datasets import StairAmbulationHealthy2021PerTest

dataset = StairAmbulationHealthy2021PerTest(
    data_folder=dataset_path,
    include_pressure_data=True,
    include_baro_data=True,
    include_hip_sensor=True,
    memory=Memory("../.cache"),
)
dataset

# %%
# We can see that we have 20 participants and each of them has performed 26 gaittests on different level walking and
# stair configurations.
# For more information about the individual tests see the documentation of the dataset itself.
#
# Using the dataset class, we can select any subset of tests and participants.
subset = dataset.get_subset(
    test=["stair_long_down_normal", "stair_long_up_normal"], participant=["subject_01", "subject_02"]
)
subset

# %%
# Once we have the selection of data we want to work with, we can iterate the dataset object to access the data of
# individual datapoints or just index it as below.
datapoint = subset[0]
datapoint

# %%
# On this datapoint, we can now access the data.
# We will start with the metadata.
# It contains all the general information about the participant and the sensors.
datapoint.metadata

# %%
# We can also access the imu data, the pressure data and the barometer data.
# All of them have an index that marks the seconds from the start of the individual test we selected.
imu_data = datapoint.data
imu_data.head()

# %%
pressure_data = datapoint.pressure_data
pressure_data.head()

# %%
baro_data = datapoint.baro_data
baro_data.head()

# %%
# In addition we provide ground truth information for the event detection.
# All event data is provided in samples from the start of the test.
#
# Note that we use a trailing `_` to indicate that this is data calculated based on the ground truth and not just the
# IMU data.
#
# First, manually labeled stride borders.
segmented_stride_list = datapoint.segmented_stride_list_
segmented_stride_list["left_sensor"]

# %%
# Second, the events extracted using the pressure-insole.
# Note, that the `min_vel` event is actually calculated based on the IMU data.
# For more information see the docstring of this property.
insole_events = datapoint.pressure_insole_event_list_
insole_events["left_sensor"]

# %%
# As further groundtruth we provide a label for each segmented stride that contains information about the height
# change during the stride.
# This information is derived by measuring the heights of the individual stair steps and labeling each stride based
# on video, to mark all strides that were performed on a specicic stair configuration.
datapoint.get_segmented_stride_list_with_type()

# %%
# The same method used to access this information can also be used to filter the stride list (i.e. only level strides).
datapoint.get_segmented_stride_list_with_type(stride_type=["level"])

# %%
# Below we plot all the relevant data for a single gait test to make it easier to understand.
#
# For the selected test, we can see that the participant basically started walking right away.
# While it can not be easily seen from the raw data IMU itself, the participant walked down a stair in two bouts.
# This can be more clearly seen in the baro data, which shows a slowly increasing pressure value, indicating a
# reduction in altitude.
import matplotlib.pyplot as plt

foot = "right_sensor"
_, axs = plt.subplots(nrows=3, figsize=(10, 10), sharex=True)
imu_data[foot].filter(like="gyr").plot(ax=axs[0])
imu_data[foot].filter(like="acc").plot(ax=axs[1])
baro_data[foot].plot(ax=axs[2])

axs[0].set_ylabel("Rate of rotation [deg/s]")
axs[1].set_ylabel("Acceleration [m/s^2]")
axs[2].set_ylabel("Air Pressure [mbar]")
axs[2].set_xlabel("Time [s]")

plt.show()

# %%
# When zooming in we can see the individual events withing the strides.
# The min_vel event is in the resting period between strides, and the IC and TC events at the falling and rising
# edges of pressure signal, respectively.
# The start and endpoints of the segmented strides (dashed lines) are at the maximum of the `gyr_y` signal.

foot = "right_sensor"
fig, axs = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
imu_data[foot].filter(like="gyr").plot(ax=axs[0])
pressure_data[foot]["total_force"].plot(ax=axs[1])
events = insole_events[foot].drop(columns=["start", "end"])
events /= datapoint.sampling_rate_hz
styles = ["ro", "gs", "b^", "m*"]
for style, (i, e) in zip(styles, events.T.iterrows()):
    e = e.dropna()
    axs[0].plot(e, imu_data[foot]["gyr_y"].loc[e.to_numpy()].to_numpy(), style, label=i, markersize=8)
    axs[1].plot(e, pressure_data[foot]["total_force"].loc[e.to_numpy()].to_numpy(), style, markersize=8)
for (i, s) in segmented_stride_list[foot].iterrows():
    s /= datapoint.sampling_rate_hz
    axs[0].axvline(s["start"], color="k", linestyle="--")
    axs[0].axvline(s["end"], color="k", linestyle="--")
    axs[1].axvline(s["start"], color="k", linestyle="--")
    axs[1].axvline(s["end"], color="k", linestyle="--")

axs[0].legend()
axs[0].set_xlim(12, 15)
axs[0].set_ylim(-500, 600)

axs[0].set_ylabel("Rate of rotation [deg/s]")
axs[1].set_ylabel("Pressure equivalent weight [kg]")
axs[1].set_xlabel("Time [s]")

plt.show()

# %%
# StairAmbulationHealthy2021Full
# ==============================
# The StairAmbulationHealthy2021Full dataset is contains the complete recordings of all 20 participants, not cut into
# individual tests.
# Note, that there are still two recordings per participant.
# This is because data was collected at two different locations and hence, the data is split into two sections.
#
# The StairAmbulationHealthy2021Full dataclass can be used equivalently to the StairAmbulationHealthyPerTest dataset.
# The only difference is that instead of the individual tests, we can see the two parts in the index for the dataset.
from gaitmap_datasets import StairAmbulationHealthy2021Full

dataset = StairAmbulationHealthy2021Full(
    data_folder=dataset_path,
    include_pressure_data=True,
    include_baro_data=True,
    include_hip_sensor=True,
    memory=Memory("../.cache"),
)
dataset

# %%
subset = dataset.get_subset(participant="subject_01", part="part_2")
subset

# %%
# As most parameters and attributes are identical, we will not repeat them.
#
# One interesting addition is the `test_list` attribute.
# If it is required to understand which tests where performed in the respective sessions, we can access them as a
# region-of-interest list.
subset.test_list

# %%
# When plotting the data in the entire `part_2` recording, we can see that it spans multiple tests including multiple
# walks up and down various stairs.
#
# If you would zoom in, you can see that between each test, the participants were instructed to jump up and down 3
# times.
# These jump events were used as marker to cut the individual tests.
imu_data = subset.data
baro_data = subset.baro_data

foot = "right_sensor"
_, axs = plt.subplots(nrows=3, figsize=(10, 10), sharex=True)
imu_data[foot].filter(like="gyr").plot(ax=axs[0])
imu_data[foot].filter(like="acc").plot(ax=axs[1])
baro_data[foot].plot(ax=axs[2])
for (i, s) in subset.test_list.iterrows():
    s /= subset.sampling_rate_hz
    axs[0].axvspan(s["start"], s["end"], color="k", alpha=0.2)
    axs[1].axvspan(s["start"], s["end"], color="k", alpha=0.2)
    axs[2].axvspan(s["start"], s["end"], color="k", alpha=0.2)

axs[0].set_ylabel("Rate of rotation [deg/s]")
axs[1].set_ylabel("Acceleration [m/s^2]")
axs[2].set_ylabel("Air Pressure [mbar]")
axs[2].set_xlabel("Time [s]")

plt.xlim(550, 650)
plt.show()

# %%
# A note on caching
# =================
# To make it possible to interact with the entire dataset, without filling your RAM immediately, all data is only
# loaded once you access the respective data attribute (e.g. `data` or `pressure_data`).
# However, this means, if you access the same piece of data multiple times (or multiple pieces of related data),
# data needs to be loaded again from disk and preprocessed.
# This is slow.
# Therefore, we allow to use `joblib.Memory` to cache the data in a fast disk cache.
# You can configure the cache directory using the `memory` parameter of the dataset class.
# Keep in mind, that the cache directory can become quite large.
# We recommend clearing the cache from time to time, to free up space.
