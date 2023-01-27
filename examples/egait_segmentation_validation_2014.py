r"""
EgaitSegmentationValidation2014 - A Stride Segmentation validation dataset
==========================================================================

The EgaitSegmentationValidation2014 dataset allows access to the stride segmentation validation dataset recorded for
the EGait system.
It contains multiple 4x10 m walks and simulated "free-living" walks recorded by two foot worn IMU sensors.
The individual strides were labeled manually by multiple gait experts.

General information
-------------------
The dataset was recorded with Shimmer 2R sensors.
In these IMU nodes, the coordinate systems of the accelerometer and the gyroscope are different.

In the version provided in this dataset, we fix this by transforming the gyroscope data to the accelerometer coordinate
system and then transform the combined data to the coordinate system of the gaitmap coordinate system.

.. figure:: /images/coordinate_systems/coordinate_transform_shimmer2R_lateral_eGait.svg
    :alt: coordinate system definition
    :figclass: align-center

.. warning:: The calibration files distributed with the dataset are likely of low quality.
             We recommend to only use this dataset for validation of stride segmentation algorithms.
             Algorithms for spatial parameters that depend on the exact values of the IMU, might not provide good
             results with this dataset.

"""

# %%
# .. warning:: For this example to work, you need to modify the dataset path in the following line to point to the
#             location of the data on your machine.
from pathlib import Path

import pandas as pd

from gaitmap_datasets import EgaitSegmentationValidation2014

# %%
# First we will create a simple instance of the dataset class.

dataset = EgaitSegmentationValidation2014()
dataset

# %%
# Based on the index you can select either a specific cohort or test, or a specific participant.
only_free_walk = dataset.get_subset(test="free_walk")
only_free_walk

# %%
# We will investigate the data for a single participant in the following for both types of tests
free_walk = only_free_walk.get_subset(participant="GA214030")
free_walk

# %%
# Free-walk
# ---------
# We will plot the data together with the manually labeled strides.
# We can see that multiple strides were labeled over the 4-min period of the measurement.
# The only exception is a small signal region in the center
import matplotlib.pyplot as plt

imu_data = free_walk.data
segmented_stride_list = free_walk.segmented_stride_list_


def plot_strides(imu_data, segmented_stride_list):
    fig, axs = plt.subplots(2, 1, sharex=True)
    foot = "right_sensor"
    imu_data[foot].filter(like="acc").plot(ax=axs[0])
    imu_data[foot].filter(like="gyr").plot(ax=axs[1])

    for (i, s) in segmented_stride_list[foot].iterrows():
        s /= free_walk.sampling_rate_hz
        for ax in axs:
            ax.axvline(s["start"], color="k", linestyle="--")
            ax.axvline(s["end"], color="k", linestyle="--")
    return fig


fig = plot_strides(imu_data, segmented_stride_list)
fig.show()

# %%
# If we zoom into this region, we can see that the signal looks "gait-like".
# However, this corresponds to stair walking, which was explicitly **not** labeled as a stride by the authors of the
# dataset, as they wanted to show that the algorithms they developed could differentiate between stair walking and
# level walking.
# Further, all turning strides were **not** labeled as a stride in the dataset.
#
# This is important to keep in mind, when evaluating the performance of the algorithms on this dataset!
fig = plot_strides(imu_data, segmented_stride_list)
fig.axes[0].set_xlim(145, 165)
fig.show()

# %%
# 4x10 m walk
# ------------
only_gait_test = dataset.get_subset(test="4x10m")
gait_test = only_gait_test.get_subset(participant="GA112030E3")
gait_test

# %%
# We will plot the data together with the manually labeled strides.
# We can clearly see the 4 straight walks during the test.
# In between the walks we can see turning strides.
# Again, these were explicitly **not** labeled as strides by the authors of the dataset.
imu_data = gait_test.data
segmented_stride_list = gait_test.segmented_stride_list_

fig = plot_strides(imu_data, segmented_stride_list)
fig.show()
