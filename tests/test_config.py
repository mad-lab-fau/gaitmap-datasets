"""Tests for the dataset config helpers."""

import json
import tempfile
from pathlib import Path

import pytest

from gaitmap_datasets import DatasetsConfig, config, create_config_template, reset_config, set_config


@pytest.fixture(autouse=True)
def config_file_clean():
    """Clean up the config file."""
    reset_config()

    yield

    reset_config()


def test_config_load_from_file():
    """Config files can be loaded into the global config."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        json_content = {"datasets": {"egait_parameter_validation_2013": tmpdirname}}

        with tempfile.NamedTemporaryFile(mode="w") as f:
            json.dump(json_content, f)
            f.seek(0)
            config_path = Path(f.name)
            set_config(config_path)

            c = config()
            assert c.egait_parameter_validation_2013 == Path(tmpdirname)


def test_create_config_template():
    """Template files contain all dataset keys and can be loaded again."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "config.json"
        create_config_template(config_path)
        assert config_path.exists()
        with config_path.open(encoding="utf8") as f:
            content = json.load(f)
        assert "datasets" in content
        assert len(content["datasets"]) == len(DatasetsConfig.__dataclass_fields__)

        # We just check that we don't get an error when loading the config.
        set_config(config_path)
        config()


def test_config_load_expands_env_vars_in_dataset_paths(monkeypatch):
    """Environment variables inside configured dataset paths are expanded."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        monkeypatch.setenv("GAITMAP_DATASET_PATH", tmpdirname)
        config_path = Path(tmpdirname) / "config.json"
        config_path.write_text(
            json.dumps({"datasets": {"egait_parameter_validation_2013": "$GAITMAP_DATASET_PATH"}}), encoding="utf8"
        )

        set_config(config_path)

        assert config().egait_parameter_validation_2013 == Path(tmpdirname)


def test_config_load_expands_env_vars_in_config_file_path(monkeypatch, tmp_path):
    """Environment variables inside the config file path are expanded."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"datasets": {"egait_parameter_validation_2013": str(dataset_path)}}), encoding="utf8"
    )
    monkeypatch.setenv("GAITMAP_CONFIG_FILE", str(config_path))

    set_config("$GAITMAP_CONFIG_FILE")

    assert config().egait_parameter_validation_2013 == dataset_path
