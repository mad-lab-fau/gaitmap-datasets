r"""
EgaitAdidas2014 - Healthy Participants with MoCap reference
===========================================================


General Information
-------------------
The EgaitAdidas2014 dataset contains data healthy participants walking through a vicon motion capture system with one
IMU attached to each foot.

For many participants data for SHIMMER3 and SHIMMER2 is available. The SHIMMER3 data is sampled at 204.8 Hz and the
SHIMMER2R data at 102.4 Hz.
This also allows for a comparison of the two sensors.

For both IMUs we unify the coordinate system on loading as shown below:

.. figure:: /images/coordinate_systems/coordinate_transform_shimmer2R_lateral_eGait.svg
    :alt: coordinate system definition
    :figclass: align-center

.. figure:: /images/coordinate_systems/coordinate_transform_shimmer3_lateral_eGait.svg
    :alt: coordinate system definition
    :figclass: align-center


Participants where instructed to walk with a specific stride length and velocity to create more variation in the
data.
For each trial only a couple strides were recorded withing the motion capture system.
The IMU data contains the entire recording.
This additional data can contain just some additional strides or entire different movements depending on the trial.
We recommend inspecting the specific trial in case of issues.

The Vicon motion capture system was sampled at 200 Hz.
The IMUs and the mocap system are synchronized using a wireless trigger allowing for proper comparison of the calculated
trajectories.

Reference (expert labeled based on IMU data) stride borders are provided for all strides that are recorded by both
systems.

In the following we will show how to interact with the dataset and how to make sense of the reference information.
"""

# %%
# .. warning:: For this example to work, you need to have a global config set containing the path to the dataset.
#              Check the `README.md` for more information.
#
# First we create a simple instance of the dataset class.
from gaitmap_datasets import EgaitAdidas2014

dataset = EgaitAdidas2014()
dataset

# %%
# We can see that we have 5 levels in the metadata.
#
# - participant
# - sensortype (shimmer2, shimmer3)
# - stride_length (low, medium, high)
# - stride_velocity (low, medium, high)
# - repetition (1, 2, 3)
#
# The `stride_length` and `stride_velocity` are the instructions given to the participants.
# For each combination of these two parameters, 3 repetitions were recorded.
#
# However, for many participants data for at least some trials are missing for various technical issues.
#
# For now we are selecting the data for one participant.
subset = dataset.get_subset(participant="008")
subset

# %%
# For this participant we will have a look at the "normal" stride length and velocity trial of the shimmer2r sensor.
trial = subset.get_subset(stride_length="normal", stride_velocity="normal", sensor="shimmer2r", repetition="1")
trial

# %%
# The IMU data is stored in the `data` attribute, which is a dictionary of pandas dataframes.
sensor = "left_sensor"
imu_data = trial.data[sensor]
imu_data

# %%
# The mocap data is stored in the `marker_position_` attribute, which is a dictionary of pandas dataframes, too.
# Note, that sometimes there are NaN values at the start and the end of the data.
# In these regions the mocap system was recording, but none of the markers were in frame.
mocap_data = trial.marker_position_[sensor]
mocap_data

# %%
# Both data sources have the time as index, so that we can easily plot them together.
# We converted the time axis so that the start of the **Mocap** data is the global 0.
# This means that the IMU data will have **negative** time values for the datapoints before the MoCap start.
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
imu_data.filter(like="gyr").plot(ax=ax1, legend=False)
imu_data.filter(like="acc").plot(ax=ax2, legend=True)
mocap_data[["heel_z"]].plot(ax=ax3)

fig.show()

# %%
# For the strides that are withing the mocap volume manual stride labels based on the IMU data are available.
# They are provided in samples relative to the start of the IMU data stream.
segmented_strides = trial.segmented_stride_list_
segmented_strides[sensor]

# %%
# To get the events relative to the mocap data (i.e. in mocap samples relative to the start of the mocap data you can
# use the `convert_events` method.
trial.convert_events(segmented_strides, from_time_axis="imu", to_time_axis="mocap")[sensor]

# %%
# Similarly, you can convert the events to the same time axis as the data
trial.convert_events(segmented_strides, from_time_axis="imu", to_time_axis="time")[sensor]

# %%
# Below we plot the eventlist time converted event list into the plot from above
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
imu_data.filter(like="gyr").plot(ax=ax1, legend=False)
imu_data.filter(like="acc").plot(ax=ax2, legend=True)
mocap_data[["heel_z"]].plot(ax=ax3)

for ax in (ax1, ax2, ax3):
    for (i, s) in trial.convert_events(segmented_strides, from_time_axis="imu", to_time_axis="time")[sensor].iterrows():
        ax.axvspan(s["start"], s["end"], alpha=0.2, color="C1")

fig.show()


# %%
# As you can see, in this example, three strides are properly detected by both systems.
# These strides are defined based on the signal maximum in the `gyr_y` (i.e. `gyr_ml` axis).
#
# This definition is good for segmentation.
# However, for calculation of spatial parameters, the authors of the dataset defined strides from midstance (i.e. the
# `min_vel` point) to midstance of two consecutive strides.
# In result, when looking at the parameters, there will be one stride less than the number of strides in the segmented
# stride list.
# TODO: Add a section about ground truth events
trial.mocap_parameters_[sensor]



# %%
# Further Notes
# -------------
# In many cases clear drift in the IMU data is observed.
# The authors of the dataset corrected that drift before calculating the reference parameters using a linear drift
# model.
# For further information see the two papers using the dataset [1]_ and [2]_.
#
# .. [1] Kanzler et al., “Inertial Sensor Based and Shoe Size Independent Gait Analysis Including Heel and Toe
#    Clearance Estimation.”
# .. [2] Hannink et al., “Benchmarking Foot Trajectory Estimation Methods for Mobile Gait Analysis.”
