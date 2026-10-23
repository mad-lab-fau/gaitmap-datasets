"""Methods to access the open source gait datasets of the MaD-Lab."""

from gaitmap_datasets._config import (
    DatasetsConfig,
    config,
    create_config_template,
    get_dataset_path,
    reset_config,
    set_config,
)
from gaitmap_datasets.egait_adidas_2014 import EgaitAdidas2014
from gaitmap_datasets.egait_parameter_validation_2013 import EgaitParameterValidation2013
from gaitmap_datasets.egait_segmentation_validation_2014 import EgaitSegmentationValidation2014
from gaitmap_datasets.kluge_2017 import Kluge2017
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
    "DatasetsConfig",
    "EgaitAdidas2014",
    "EgaitParameterValidation2013",
    "EgaitSegmentationValidation2014",
    "Kluge2017",
    "PyShoe2019Hallway",
    "PyShoe2019Stairs",
    "PyShoe2019Vicon",
    "SensorPositionComparison2019Mocap",
    "SensorPositionComparison2019Segmentation",
    "StairAmbulationHealthy2021Full",
    "StairAmbulationHealthy2021PerTest",
    "config",
    "create_config_template",
    "get_dataset_path",
    "reset_config",
    "set_config",
]
__version__ = "0.15.1"
