"""Utils to manage the configuration of the datasets path."""
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

from pydantic import BaseSettings, DirectoryPath
from pydantic.env_settings import SettingsSourceCallable


def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """Get config data from json file."""
    config_file = getattr(settings.__config__, "config_file", None)
    using_default = False
    if config_file is None:
        config_file = getattr(settings.__config__, "default_config_file", None)
        using_default = True
    try:
        with open(config_file, encoding="utf8") as f:
            return json.load(f)["datasets"]
    except FileNotFoundError as e:
        if using_default:
            return {}
        raise ValueError(f"Config file {config_file} not found.") from e


class DatasetsConfig(BaseSettings):
    """Configuration class for the dataset paths."""

    egait_parameter_validation_2013: Optional[DirectoryPath]
    sensor_position_comparison_2019: Optional[DirectoryPath]
    egait_segmentation_validation_2014: Optional[DirectoryPath]
    pyshoe_2019: Optional[DirectoryPath]
    stair_ambulation_healthy_2021: Optional[DirectoryPath]

    class Config:
        """The config."""

        config_file: ClassVar[str]
        default_config_file: ClassVar[str] = (Path(__file__).parent.parent / ".datasets.dev.json").resolve()
        validate_assignment = True

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,  # pylint: disable=unused-argument
            file_secret_settings: SettingsSourceCallable,  # pylint: disable=unused-argument
        ) -> Tuple[SettingsSourceCallable, ...]:
            """Customize the sources."""
            return init_settings, json_config_settings_source


_GLOBAL_CONFIG: Optional[DatasetsConfig] = None


def set_config(config_obj_or_path: Union[str, Path, DatasetsConfig] = DatasetsConfig()):
    """Set the global config object containing configured paths for the datasets.

    This allows you to set a global configuration that is automatically used by the dataset classes.
    It allows you to load the datasets without explicitly passing the path to the dataset.

    Note, that you have to call this function before you initialize any dataset class.
    If you want to change the config during runtime, you can use the `reset_config` function and then call
    `set_config` again, or get the config (`config()`) and change the attributes of the config object.

    Parameters
    ----------
    config_obj_or_path
        You can either pass a valid config object or a path to a config file.
        To create a config file, you can use the `create_config_file` function.

    """
    global _GLOBAL_CONFIG  # pylint: disable=global-statement
    if _GLOBAL_CONFIG is not None:
        raise ValueError("Config is already set!")
    if isinstance(config_obj_or_path, (str, Path)):
        DatasetsConfig.Config.config_file = Path(config_obj_or_path)
        config_obj = DatasetsConfig()
    elif isinstance(config_obj_or_path, DatasetsConfig):
        config_obj = config_obj_or_path
    else:
        raise ValueError("Unknown config type.")
    _GLOBAL_CONFIG = config_obj


def reset_config():
    """Reset the global config to None.

    Afterwards you can use `set_config` to set a new config (e.g. to change the config file during runtime).
    """
    global _GLOBAL_CONFIG  # pylint: disable=global-statement
    _GLOBAL_CONFIG = None


def config() -> DatasetsConfig:
    """Get the global config object containing configured paths for the datasets.

    Returns
    -------
    DatasetsConfig
        The global config object.
        For each dataset it contains the configured path.
        I.e. `config().egait_parameter_validation_2013` is the path to the egait_parameter_validation_2013 dataset.

    """
    if _GLOBAL_CONFIG is None:
        set_config()
    return _GLOBAL_CONFIG


def create_config_template(path: Union[str, Path]):
    """Create a template json file that can be used to configure the datasets paths.

    Use that method once to create your local config file.
    Open it afterwards and fill in the paths to the datasets that you need.

    Then you can use `set_config(path_to_config)` to set the global config to your local config file.

    Parameters
    ----------
    path : Union[str, Path]
        The path to the file where the config should be created.

    """
    path = Path(path)
    if path.exists():
        raise ValueError(f"Config file {path} already exists.")

    with open(path, "w", encoding="utf8") as f:
        json.dump({"datasets": {k: None for k in DatasetsConfig.__fields__}}, f, indent=4, sort_keys=True)

    print(f"Created config template at {path.resolve()}.")


def get_dataset_path(dataset_name: str) -> Path:
    """Get the path to a dataset be reading the global config.

    If no path is configured for the dataset, a ValueError is raised.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
        The name is usually the module folder name of the dataset.

    """
    if (path := getattr(config(), dataset_name, None)) is not None:
        return Path(path)

    raise ValueError(
        f"We tried to load the dataset path for {dataset_name}, but it was not found in the config.\n"
        "There are a couple of options how to fix this:\n"
        "1. Explicitly provide a path to the dataset class. This will skip the config lookup entirely\n"
        "2. Modify the global config object to include the path to the dataset. "
        f"This can be done by calling `gaitmap_datasets.config().{dataset_name} = path_to_dataset`\n"
        "3. Create a json config file containing the required paths (see `create_config_template`) and set it as the "
        "global config: `gaitmap_datasets.set_config(path_to_config)`\n"
        "4. If you are currently developing the package, you can create a `.datasets.dev.json` file in the "
        "root of the package (`poe create_dev_config`)."
        "This file will be automatically loaded if no other config is set.\n\n"
        "If you already have a done one of the above steps, double check the spelling of the dataset name in your "
        "config."
    )
