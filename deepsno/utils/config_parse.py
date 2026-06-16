from __future__ import annotations

import importlib
from collections.abc import Mapping

from jsonargparse import Namespace


def check_instantiate_keys(cfg_obj: Namespace | dict, object_name: str):
    if "class_path" not in cfg_obj:
        raise KeyError(f"'class_path' not found in {object_name} config object")


"""Recursively instantiate objects from a nested config structure.

The config is built from plain scalars, strings, iterables (lists/tuples)
and mappings (dicts). A mapping that contains the key ``"class_path"`` is
treated as an instantiation directive: the dotted path is imported and the
resulting class is called with the arguments found under ``"init_args"``.

This mirrors the ``class_path`` / ``init_args`` convention used by
jsonargparse and PyTorch Lightning's ``LightningCLI``.
"""


def get_class(path):
    """Import and return the object referred to by a dotted path string.

    Handles plain ``"package.module.ClassName"`` as well as paths that dig
    into attributes of an imported object, e.g. ``"pkg.mod.Outer.Inner"``.
    """
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Could not import {path!r}")


def _is_sequence(obj):
    """True for iterables we treat as positional args (not str/mapping)."""
    return isinstance(obj, (list, tuple))


def instantiate(obj):
    """Recursively resolve a nested config into live Python objects.

    Rules
    -----
    * A mapping with a ``"class_path"`` key is imported and instantiated.
      Its ``"init_args"`` (if present) are resolved first, then used as
      ``**kwargs`` (mapping), ``*args`` (list/tuple) or a single positional
      argument (scalar). With no ``"init_args"`` the class is called with
      no arguments.
    * Any other mapping is rebuilt as a ``dict`` with each value resolved.
    * Lists and tuples are rebuilt (type preserved) with each element resolved.
    * Strings and other scalars are returned unchanged.
    """
    if isinstance(obj, Mapping):
        if "class_path" in obj:
            cls = get_class(obj["class_path"])
            if "init_args" not in obj:
                return cls()
            args = obj["init_args"]
            if isinstance(args, Mapping):
                return cls(**{k: instantiate(v) for k, v in args.items()})
            if _is_sequence(args):
                return cls(*(instantiate(v) for v in args))
            return cls(instantiate(args))
        return {k: instantiate(v) for k, v in obj.items()}

    if isinstance(obj, str):
        return obj

    if _is_sequence(obj):
        return type(obj)(instantiate(v) for v in obj)

    return obj
