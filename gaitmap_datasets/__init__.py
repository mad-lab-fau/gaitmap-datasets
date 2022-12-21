# -*- coding: utf-8 -*-
"""Methods to access the open source gait datasets of the MaD-Lab."""
from gaitmap_datasets.egait_parameter_validation_2013 import EgaitParameterValidation2013
from gaitmap_datasets.egait_segmentation_validation_2014 import EgaitSegmentationValidation2014
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Stairs, PyShoe2019Hallway, PyShoe2019Vicon
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Segmentation,
    SensorPositionComparison2019Mocap,
)
from gaitmap_datasets.stair_ambulation_healthy_2021 import (
    StairAmbulationHealthy2021Full,
    StairAmbulationHealthy2021PerTest,
)

__all__ = [
    "EgaitSegmentationValidation2014",
    "EgaitParameterValidation2013",
    "StairAmbulationHealthy2021PerTest",
    "StairAmbulationHealthy2021Full",
    "SensorPositionComparison2019Segmentation",
    "SensorPositionComparison2019Mocap",
    "PyShoe2019Vicon",
    "PyShoe2019Hallway",
    "PyShoe2019Stairs",
]
__version__ = "0.5.0"
