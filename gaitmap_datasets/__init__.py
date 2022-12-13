# -*- coding: utf-8 -*-
"""Methods to access the open source gait datasets of the MaD-Lab."""
from gaitmap_datasets.egait_parameter_validation_2013 import EgaitParameterValidation2013
from gaitmap_datasets.egait_segmentation_validation_2014 import EgaitSegmentationValidation2014
from gaitmap_datasets.stair_ambulation_healthy_2021 import (
    StairAmbulationHealthy2021Full,
    StairAmbulationHealthy2021PerTest,
)

__all__ = [
    "EgaitSegmentationValidation2014",
    "EgaitParameterValidation2013",
    "StairAmbulationHealthy2021PerTest",
    "StairAmbulationHealthy2021Full",
]
__version__ = "0.5.0"
