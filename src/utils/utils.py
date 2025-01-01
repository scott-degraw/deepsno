import math
import re
from pathlib import Path
from typing import Any

import h5py
import jsonargparse
import torch


def copy_if_tensor(x: Any | torch.Tensor) -> torch.Tensor:
    # Mostly here to avoid warnings about copying tensors with torch.tensor. The performance here shouldn't really
    # matter at all.
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return torch.tensor(x)


def convert_byte_units(size: int, unit: str, original_unit: str = "B"):
    unit_values = {
        "B": 1,
        "KB": 1000,
        "KiB": 1024,
        "MB": 1000**2,
        "MiB": 1024**2,
        "GB": 1000**3,
        "GiB": 1024**3,
        "TB": 1000**4,
        "TiB": 1000**4,
        "PB": 1000**5,
        "PiB": 1000**5,
    }

    if unit not in unit_values:
        raise ValueError(f"Invalid unit '{unit}'. Valid units: {unit_values.keys()}")
    if original_unit not in unit_values:
        raise ValueError(f"Invalid unit '{original_unit}'. Valid units: {unit_values.keys()}")

    return size * unit_values[original_unit] / unit_values[unit]


def get_best_ckpt(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory: '{checkpoint_dir}' is not an existing directory")

    loss_pattern = re.compile(r".*?val_loss=(-?\d+(\.\d+)?).*?\.pt")

    best_ckpt = None
    min_loss = math.inf

    for path in checkpoint_dir.iterdir():
        if path.is_file():
            match = re.match(loss_pattern, path.name)
            if match is None:
                continue

            loss = float(match.group(1))

            if loss < min_loss:
                min_loss = loss
                best_ckpt = path

    if best_ckpt is None:
        raise RuntimeError(
            (
                f"No checkpoint with valid filename found in {checkpoint_dir}. ",
                "Filename must contain 'val_loss=<val-loss>' substring.",
            )
        )

    return best_ckpt


def write_config_to_h5(h5_group: h5py.Group, config_obj: dict):
    for key, item in config_obj.items():
        if isinstance(item, dict):
            sub_group = h5_group.create_group(key)
            write_config_to_h5(sub_group, item)
        elif isinstance(item, jsonargparse.Path):
            item = str(item)
        elif item is not None:
            h5_group.attrs[key] = item
