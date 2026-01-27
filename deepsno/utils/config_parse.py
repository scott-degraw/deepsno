import importlib
from typing import Any

from jsonargparse import Namespace


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


def instantiate(cfg_obj: Any):
    if isinstance(cfg_obj, str):
        return cfg_obj
    try:
        class_path = cfg_obj["class_path"]
    except KeyError:
        for key, item in cfg_obj.items():
            cfg_obj[key] = instantiate(item)
        return cfg_obj
    except TypeError:
        try:
            cfg_obj = [instantiate(item) for item in cfg_obj]
        except TypeError:
            pass
        return cfg_obj

    class_type = get_class(class_path)
    init_args = cfg_obj.get("init_args", {})

    for key, item in init_args.items():
        if key == "class_path":
            init_args[key] = get_class(item)
        else:
            init_args[key] = instantiate(item)

    return class_type(**init_args)
