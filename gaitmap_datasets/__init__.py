# -*- coding: utf-8 -*-
"""Methods to access the open source gait datasets of the MaD-Lab."""
from gaitmap_datasets._config import (
    DatasetsConfig,
    config,
    create_config_template,
    get_dataset_path,
    reset_config,
    set_config,
)
from gaitmap_datasets.egait_parameter_validation_2013 import EgaitParameterValidation2013
from gaitmap_datasets.egait_segmentation_validation_2014 import EgaitSegmentationValidation2014
from gaitmap_datasets.pyshoe_2019 import PyShoe2019Hallway, PyShoe2019Stairs, PyShoe2019Vicon
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Mocap,
    SensorPositionComparison2019Segmentation,
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
    "DatasetsConfig",
    "set_config",
    "reset_config",
    "config",
    "create_config_template",
    "get_dataset_path",
]
__version__ = "0.8.0"
