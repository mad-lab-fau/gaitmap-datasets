[![PyPI](https://img.shields.io/pypi/v/gaitmap-datasets)](https://pypi.org/project/gaitmap-datasets/)
[![Documentation status](https://img.shields.io/badge/docs-online-green)](https://mad-lab-fau.github.io/gaitmap-datasets)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gaitmap-datasets)

# gaitmap-datasets

Helper to access to open-source gait datasets compatible with the MaD-Lab gaitanalysis library gaitmap.

The aim of this package is to ensure that all datasets can be loaded in a similar fashion and all data (and annotations)
are in the same format (i.e. the same sensor orientations, units, etc.).
This should allow to easily run the same algorithm across multiple datasets.

> :warning: While this makes it easier to work with the datasets, the coordinate system and other data information
> provided with the dataset might not match the format you get when using this library!


All datasets APIs are built using the 
[`tpcp.Dataset`](https://tpcp.readthedocs.io/en/latest/modules/generated/dataset/tpcp.Dataset.html#tpcp.Dataset)
interface.
For available datasets see the table below.

## Usage

Install the package from Pip

```
pip install gaitmap-datasets
```

Then download/obtain the dataset that you are planning to use (see below).
The best way to get started is to then check the example for the respective dataset on the 
[documentation page](https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/index.html).

## Datasets

| Dataset                         | Info Link                                                       | Download                            |
|---------------------------------|-----------------------------------------------------------------|-------------------------------------|
| EgaitSegmentationValidation2014 | https://www.mad.tf.fau.de/research/activitynet/digital-biobank/ | Email to data owner (see info link) |
| EgaitParameterValidation2013    | https://www.mad.tf.fau.de/research/activitynet/digital-biobank/ | Email to data owner (see info link) |
| StairAmbulationHealthy2021      | https://osf.io/sgbw7/                                           | https://osf.io/download/5ueq6/      |
| SensorPositionDataset2019       | https://zenodo.org/record/5747173                               | https://zenodo.org/record/5747173   |

## Testing

The `/tests` directory contains a set of tests to check the functionality of the library.
However, most tests rely on the existence of the respective datasets in certain folders outside the library.
Therefore, the tests can only be run locally and not on the CI server.

To run them locally, make sure datasets are downloaded into the correct folders and then run `poe test`.

## Documentation (build instructions)

Like the tests, the documentation requires the datasets to be downloaded into the correct folders to execute the 
examples.
Therefore, we can not build the docs automatically on RTD.
Instead we host the docs via github pages.
The HTML source can be found in the `gh-pages` branch of this repo.

To make the deplowment as easy as possible, we "mounted" the `gh-pages` branch as a submodule in the `docs/_build/html`
folder.
Hence, before you attempt to build the docs, you need to initialize the submodule.

```
git submodule update --init --recursive
```

After that you can run `poe docs` to build the docs and then `poe upload_docs` to push the changes to the gh-pages
branch.
We will always just update a single commit on the gh-pages branch to keep the effective file size small.

**WARNING: ** Don't delete the `docs/_build` folder manually or by running the sphinx make file!
This will delete the submodule and might cause issues.
The `poe` task is configured to clean all relevant files in the `docs/_build` folder before each run.

After an update of the documentation, you will see that you also need to make a commit in the main repo, as the commit 
hash of the docs submodule has changed.

To make sure you don't forget to update the docs, the `poe prepare_release` task will also build and upload the docs 
automatically.