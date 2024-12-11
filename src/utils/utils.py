import math
import re
from pathlib import Path

import h5py
import jsonargparse


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
