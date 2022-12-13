"""Create the pressure-insole based events from the raw data.

This works in 3 steps:
1. We use the manual stride borders (which start/end in a minimum of the gyr_ml) data axis as regions of interest
   for all events
2. We find mid-stance regions (min_vel event) using the Gyro energy within the stride regions.
   Based on these we redefine the stride regions to start and end in a mid-stance.
   For strides at the start of a gait sequence (which will not have another manual segmented stride directly before
   them), we search backwards to find the closest resting period we can find.
   This way we don't loos any strides.
3. With the min_vel -> min_vel strides, we find the pressure events using the total force over all pressure sensors
   in the insole.
   We use a relatively simple peak detection and check if we can find two zero-crossings with an adaptive threshold.
   These crossings mark the IC and the TC.

Running this script will overwrite the previous event data in your downloaded dataset.
Usually there is no need to rerun this script unless the data or the event detection method have been changed.

Run the script as follows:

```shell
python -m gaitmap_datasets.stair_ambulation_healthy_2021.scripts.create_pressure_events --data_path="..."
```
"""
import argparse
from pathlib import Path

import pandas as pd
from joblib import Memory

from gaitmap_datasets.stair_ambulation_healthy_2021 import StairAmbulationHealthy2021Full
from gaitmap_datasets.stair_ambulation_healthy_2021.pressure_sensor_helper import (
    PressureEventDetection,
    convert_segmented_stride_list_to_min_vel_list,
)
from gaitmap_datasets.utils.coordinate_transforms import convert_to_fbf

if __name__ == "__main__":

    # Get path from commandline argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    args = parser.parse_args()
    data_path = Path(args.data_path)

    dataset = StairAmbulationHealthy2021Full(
        data_folder=data_path, ignore_manual_session_markers=True, include_pressure_data=True, memory=Memory(".cache")
    )

    for subset in dataset:
        print(subset.groups)
        imu_data = subset.data
        imu_data_bf = convert_to_fbf(imu_data)

        min_vel_stride_list = convert_segmented_stride_list_to_min_vel_list(
            imu_data_bf, subset.segmented_stride_list_, subset.sampling_rate_hz
        )

        # fsr threshold 7.5% of body weight
        fsr_threshold = subset.metadata["weight"] * 0.075
        ped = PressureEventDetection().detect(
            pd.concat([imu_data, subset.pressure_data], axis=1), min_vel_stride_list, threshold_kg=fsr_threshold
        )
        ped_events = ped.min_vel_event_list_
        ped_events = (
            pd.concat(ped_events, axis=0, names=["sensor", "s_id"])
            .reset_index(level="s_id", drop=True)
            .sort_values("start")
            .reset_index()
            .set_index("sensor", append=True)
            .swaplevel()
        )
        ped_events.index.names = ["sensor", "s_id"]

        # Save the events to the dataset
        participant, part = subset._get_participant_and_part("")
        ped_events.to_csv(data_path / "healthy" / participant / part / "pressure_events.csv")
