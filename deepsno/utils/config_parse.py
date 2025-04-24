import importlib
from typing import Any

import h5py
from jsonargparse import Namespace
import jsonargparse


def get_class(class_path: str) -> type:
    if "." in class_path:
        module_path, class_str = class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(__name__)

    return getattr(module, class_str)


def check_instantiate_keys(cfg_obj: Namespace | dict, object_name: str):
    if "class_path" not in cfg_obj:
        raise KeyError(f"'class_path' not found in {object_name} config object")


def write_config_to_h5(h5_group: h5py.Group, config_obj: dict):
    for key, item in config_obj.items():
        if isinstance(item, dict):
            sub_group = h5_group.create_group(key)
            write_config_to_h5(sub_group, item)
        elif isinstance(item, jsonargparse.Path):
            item = str(item)
        elif item is not None:
            h5_group.attrs[key] = item


def instantiate(cfg_obj: Any):
    if hasattr(cfg_obj, "keys"):
        if "class_path" in cfg_obj:
            class_path = cfg_obj["class_path"]
            max_len = 2 if "init_args" in cfg_obj else 1

            if len(cfg_obj) > max_len:
                raise KeyError("Found 'class_path' key in config object but also invalid key(s) other than 'init_args'")

            class_type = get_class(class_path)
            if "init_args" in cfg_obj:
                cfg_obj["init_args"] = instantiate(cfg_obj["init_args"])

                return class_type(**cfg_obj["init_args"])

            return class_type()

        for key, item in cfg_obj.items():
            cfg_obj[key] = instantiate(item)

    return cfg_obj
