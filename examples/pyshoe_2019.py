r"""
PyShoe2019 - Indoor Navigation with foot-mounted inertial sensors
=================================================================

The PyShoe dataset [1]_ contains multiple trials with spatial reference for longer trials.
This makes it perfect to benchmark trajectory reconstruction algorithms.
However, it does not contain any temporal marker or stride reference.

General information
-------------------
The dataset was recorded with a LORD MicroStrain 3DM-GX5-25 IMU sensor.
A single IMU node was attached on the top of the right shoe.

On loading, we transform the data to the coordinate system of the gaitmap coordinate system as shown below

.. figure:: /images/coordinate_systems/coordinate_transform_microstrain_instep_pyshoe.svg
    :alt: coordinate system definition
    :figclass: align-center

The data is split into three parts:

- Vicon: Individual trials recorded with a Vicon motion capture system as reference
- Hallway: Multiple longer walking/running trials recorded with specific landmarks as positional reference along the
  trial
- Stairs: Multiple trials from a single participant recorded with a stair climbing task

For each part of the data we provide a separate dataset class.

.. [1] Wagstaff, Brandon, Valentin Peretroukhin, and Jonathan Kelly.
       “Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation.” IEEE Sensors Journal 20,
       no. 2 (January 15, 2020): 957–67. https://doi.org/10.1109/JSEN.2019.2944412.

"""
# %%
# .. warning:: For this example to work, you need to modify the dataset path in the following line to point to the
#             location of the data on your machine.
#
# Vicon Dataset
# -------------
# First we will create a simple instance of the dataset class.
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Vicon

dataset = PyShoe2019Vicon()
dataset

# %%
# Based on the index you can select individual trials.
# Note, that some of the trials contain running/shuffling or backward walking.
# However, the dataset authors do not provide a label for that (https://github.com/utiasSTARS/pyshoe/issues/11).
# If it is important to you to only consider trials containing movements of a specific type, you need to manually
# check the raw data and guess based on that.
trial = dataset.get_subset(trial="2017-11-22-11-22-03")

# %%
# When we have selected a single trial, we can access the data.
# The data is stored in a pandas DataFrame, where each row corresponds to a single time step.
# Note, that the dataset only contains data from a single IMU attached to the right shoe.
# We still provide the data as a "nested" dataframe, where the outermost level corresponds to the foot.
imu_data = trial.data
imu_data

# %%
# The mocap marker data is also stored in a pandas DataFrame and the marker is placed directly on the sensor.
mocap_data = trial.marker_position_
mocap_data

# %%
# Both data types are recorded at 200 Hz and are synchronized.
# This means we can plot them in an aligned way without any modifications.
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, sharex=True)
imu_data["right_sensor"].filter(like="acc").plot(ax=axs[0])
imu_data["right_sensor"].filter(like="gyr").plot(ax=axs[1])
mocap_data["right_sensor"].plot(ax=axs[2])
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Position [m]")
axs[1].set_ylabel("Gyro [deg/s]")
axs[0].set_ylabel("Acc [m/s^2]")

fig.tight_layout()
fig.show()

# %%
# Hallway Dataset
# ---------------
# First we will create a simple instance of the dataset class.
# We can see that the dataset contains trials from multiple participants with the three trial types (walking, running,
# combined).
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Hallway

dataset = PyShoe2019Hallway()
dataset

# %%
# We can select arbitrary subsets of the data.
# For example, we can select all combined (running+walking) trials of a specific participant.
subset = dataset.get_subset(participant="p1", type="comb")
subset

# %%
# When we have selected a single trial, we can access the data.
trial = subset[0]
trial

# %%
# We can access the IMU data as before.
imu_data = trial.data
imu_data

# %%
# The reference position is only provided for individual points along the trial.
reference = trial.position_reference_
reference

# %%
# The index of the reference corresponds to timestamps in the IMU data.
# Hence, we can easily get the IMU data for the time points of the reference (or the position, once we have calculated
# it based on the IMU data).
imu_data.loc[reference.index]

# %%
# Below we plot all the data together.

fig, axs = plt.subplots(3, 1, sharex=True)
imu_data["right_sensor"].filter(like="acc").plot(ax=axs[0])
imu_data["right_sensor"].filter(like="gyr").plot(ax=axs[1])
reference["right_sensor"].plot(ax=axs[2], style="o")
axs[2].set_ylabel("Position [m]")
axs[1].set_ylabel("Gyro [deg/s]")
axs[0].set_ylabel("Acc [m/s^2]")

fig.tight_layout()
fig.show()

# %%
# Stairs Dataset
# --------------
# The dataset contains trails of a participant walking different number of levels of stairs in starcase.
# For each number of stairs one trail exists that starts with the participant walking down and then back up (
# `first_direction="down"`) and one trial that starts with the participant walking up and then back down (
# `first_direction="up"`).
#
# First we will create a simple instance of the dataset class.
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Stairs

dataset = PyShoe2019Stairs()
dataset

# %%
# We can simply select either the trials starting with going down the up or down trials.
#
subset = dataset.get_subset(first_direction="down")
subset

# %%
# When we have selected a single trial, we can access the data.
trial = subset.get_subset(n_levels="6")
trial

# %%
# We can access the IMU data as before.
imu_data = trial.data
imu_data

# %%
# As with the hallway dataset the reference position is only provided for individual points along the trial.
# Further, as the reference is derived from the stair geometry, reference only exist for the z-axis.
reference = trial.position_reference_
reference

# %%
# Below we plot all the data together.

fig, axs = plt.subplots(3, 1, sharex=True)
imu_data["right_sensor"].filter(like="acc").plot(ax=axs[0])
imu_data["right_sensor"].filter(like="gyr").plot(ax=axs[1])
reference["right_sensor"].plot(ax=axs[2], style="o")
axs[2].set_ylabel("Position [m]")
axs[1].set_ylabel("Gyro [deg/s]")
axs[0].set_ylabel("Acc [m/s^2]")

fig.tight_layout()
fig.show()
