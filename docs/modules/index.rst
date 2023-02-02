.. _api_ref:

=============
API Reference
=============

When working with gaitmap-datasets, you should interact with the highlevel Dataset classes listed below.
To better understand which class belongs to which dataset and how to use the respective classes,
checkout the README (Getting Started) and the Examples.

If you don't want to use the High Level API, you can also import the `helper` module of respective dataset.
However, be carefull, when doing that.
These low level function might return data formats that might not be in line with the return values of the high-level
interface.

.. currentmodule:: gaitmap_datasets

.. autosummary::
   :toctree: generated
   :template: class_no_inherited.rst

    EgaitSegmentationValidation2014
    EgaitParameterValidation2013
    StairAmbulationHealthy2021PerTest
    StairAmbulationHealthy2021Full
    SensorPositionComparison2019Segmentation
    SensorPositionComparison2019Mocap
    PyShoe2019Vicon
    PyShoe2019Hallway
    PyShoe2019Stairs

General Utility Functions
-------------------------
.. currentmodule:: gaitmap_datasets.utils

.. autosummary::
   :toctree: generated
   :template: function.rst

    convert_segmented_stride_list

Config API
----------
.. currentmodule:: gaitmap_datasets

.. autosummary::
   :toctree: generated
   :template: function.rst

    config
    set_config
    reset_config
    create_config_template
    get_dataset_path

.. autosummary::
   :toctree: generated
   :template: class_no_inherited.rst

    DatasetsConfig