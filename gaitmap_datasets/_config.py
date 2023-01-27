import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

from pydantic import BaseSettings, DirectoryPath
from pydantic.env_settings import SettingsSourceCallable


def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """Getting config data from yaml file."""
    config_file = getattr(settings.__config__, "config_file", None)
    using_default = False
    if config_file is None:
        config_file = getattr(settings.__config__, "default_config_file", None)
        using_default = True
    try:
        with open(config_file) as f:
            return json.load(f)["datasets"]
    except FileNotFoundError:
        if using_default:
            return {}
        raise ValueError(f"Config file {config_file} not found.")


class DatasetsConfig(BaseSettings):
    """Configuration class for the dataset paths."""

    egait_parameter_validation_2013: Optional[DirectoryPath]
    sensor_position_comparison_2019: Optional[DirectoryPath]
    egait_segmentation_validation_2014: Optional[DirectoryPath]
    pyshoe_2019: Optional[DirectoryPath]
    stair_ambulation_healthy_2021: Optional[DirectoryPath]

    class Config:
        config_file: ClassVar[str]
        default_config_file: ClassVar[str] = (Path(__file__).parent.parent / ".datasets.dev.json").resolve()
        validate_assignment = True

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            return init_settings, json_config_settings_source


_GLOBAL_CONFIG: Optional[DatasetsConfig] = None


def set_config(config_obj_or_path: Union[str, Path, DatasetsConfig] = DatasetsConfig()):
    global _GLOBAL_CONFIG
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
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None


def config() -> DatasetsConfig:
    if _GLOBAL_CONFIG is None:
        set_config()
    return _GLOBAL_CONFIG


def create_config_template(path: Union[str, Path]):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Config file {path} already exists.")

    with open(path, "w") as f:
        json.dump({"datasets": {k: None for k in DatasetsConfig.__fields__}}, f, indent=4, sort_keys=True)

    print(f"Created config template at {path.resolve()}.")


def get_dataset_path(dataset_name: str) -> Path:
    if (path := getattr(config(), dataset_name, None)) is not None:
        return Path(path)

    raise ValueError(
        f"We tried to load the dataset path for {dataset_name}, but it was not found in the config.\n"
        "There are a couple of options how to fix this:\n"
        "1. Explicitly provide a path to the dataset class. This will skip the config lookup entirely\n"
        "2. Modify the global config object to include the path to the dataset. "
        f"This can be done by calling `gaitmap_datasets.config().{dataset_name} = path_to_dataset`\n"
        "3. Create a json config file containing the required paths and set it as the global config: "
        "`gaitmap_datasets.set_config(path_to_config)`\n"
        "4. If you are currently developing the package, you can create a `.datasets.dev.json` file in the "
        "root of the package. This file will be automatically loaded if no other config is set."
    )
