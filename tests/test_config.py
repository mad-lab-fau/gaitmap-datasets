import json
import tempfile
from pathlib import Path

import pytest

from gaitmap_datasets import config, create_config_template, reset_config, set_config


@pytest.fixture(scope="function")
def config_file_clean():
    """Clean up the config file."""
    reset_config()

    yield

    reset_config()


def test_config_load_from_file(config_file_clean):
    with tempfile.TemporaryDirectory() as tmpdirname:
        json_content = {"datasets": {"egait_parameter_validation_2013": tmpdirname}}

        with tempfile.NamedTemporaryFile(mode="w") as f:
            json.dump(json_content, f)
            f.seek(0)
            config_path = Path(f.name)
            set_config(config_path)

            c = config()
            assert c.egait_parameter_validation_2013 == Path(tmpdirname)


def test_create_config_template(config_file_clean):
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = Path(tmpdirname) / "config.json"
        create_config_template(config_path)
        assert config_path.exists()
        content = json.load(open(config_path))
        assert "datasets" in content
        assert len(content["datasets"]) == 5

        # We just check that we don't get an error when loading the config.
        set_config(config_path)
        config()
