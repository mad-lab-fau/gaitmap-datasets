import matplotlib.pyplot as plt

from gaitmap_datasets import EgaitAdidas2014

dataset = EgaitAdidas2014()

print(dataset)

subset = dataset.get_subset(sensor="shimmer2r")[4]
foot = "left"
sensor = f"{foot}_sensor"

imu = subset.data[sensor]
strides = subset.segmented_stride_list_[sensor]
mocap_strides = subset.mocap_segmented_stride_list_[sensor]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
imu.filter(like="gyr").plot(ax=ax1)
for (i, s) in strides.iterrows():
    s /= subset.sampling_rate_hz
    ax1.axvline(s["start"], color="k", linestyle="--")
    ax1.axvline(s["end"], color="k", linestyle="--")

marker_position_ = subset.marker_position_
marker_position_.index += subset.mocap_offset_s_
marker_position_[sensor][["heel_z"]].plot(ax=ax2)

for (i, s) in mocap_strides.iterrows():
    s /= subset.mocap_sampling_rate_hz_
    s += subset.mocap_offset_s_
    ax2.axvline(s["start"], color="k", linestyle="--")
    ax2.axvline(s["end"], color="k", linestyle="--")

plt.show()
