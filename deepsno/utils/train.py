import math
import re
from pathlib import Path
from typing import Any

import torch


def copy_if_tensor(x: Any | torch.Tensor) -> torch.Tensor:
    # Mostly here to avoid warnings about copying tensors with torch.tensor. The performance here shouldn't really
    # matter at all.
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return torch.tensor(x)


def convert_units(
    value: float | int, unit: str, original_unit: str, unit_values: dict[str : int | float]
) -> float | int:
    if unit not in unit_values:
        raise ValueError(f"Invalid unit '{unit}'. Valid units: {unit_values.keys()}")
    if original_unit not in unit_values:
        raise ValueError(f"Invalid unit '{original_unit}'. Valid units: {unit_values.keys()}")

    return value * unit_values[original_unit] / unit_values[unit]


def convert_time_units(time: float | int, unit: str, original_unit: str = "s"):
    unit_values = {"s": 1}
    unit_values["min"] = 60
    unit_values["hr"] = 60 * unit_values["min"]
    unit_values["day"] = 24 * unit_values["hr"]
    unit_values |= {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}

    return convert_units(time, unit, original_unit, unit_values)


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

    return convert_units(size, unit, original_unit, unit_values)


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
