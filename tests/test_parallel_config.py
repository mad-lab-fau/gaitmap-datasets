"""Tests for config propagation into multiprocessing workers."""

import pytest
from joblib import Parallel

from gaitmap_datasets import DatasetsConfig, config, reset_config, set_config

tpcp_parallel = pytest.importorskip("tpcp.parallel")


def _read_dataset_path_from_global_config():
    dataset_path = config().egait_parameter_validation_2013
    if dataset_path is None:
        return None
    return str(dataset_path)


@pytest.fixture(autouse=True)
def config_clean():
    """Reset the global config around each test."""
    reset_config()
    yield
    reset_config()


def test_config_is_available_in_joblib_workers_when_using_tpcp_delayed(tmp_path):
    """The global config is restored in loky workers when using `tpcp.parallel.delayed`."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    set_config(DatasetsConfig(egait_parameter_validation_2013=dataset_path))

    results = Parallel(n_jobs=2, backend="loky")(
        tpcp_parallel.delayed(_read_dataset_path_from_global_config)() for _ in range(2)
    )

    assert results == [str(dataset_path), str(dataset_path)]
