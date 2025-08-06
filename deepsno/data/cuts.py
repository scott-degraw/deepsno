#!/usr/bin/env python
from pathlib import Path

import h5py
import numpy as np
from jsonargparse import CLI
from tqdm import trange


def cuts(
    h5_path: str | Path,
    save_path: str | Path,
    save_prefix: str,
    min_nhits: int | None = None,
    max_nhits: int | None = None,
    min_energy: float | None = None,
    max_energy: float | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    block_size: int = 100_000,
):
    save_path = Path(save_path)
    if not save_path.is_dir():
        raise ValueError("'save_path' is not a directory")
    if not save_path.exists():
        raise ValueError(f"'save_path' directory '{save_path}' does not exist")

    h5_file = h5py.File(h5_path)

    event = h5_file["event"]
    dset_len = event["av_offset"].shape[0]

    selector = np.zeros(dset_len, dtype=np.bool)

    save_name = save_prefix

    if min_nhits is not None:
        save_name = save_name + f"_nhits>={min_nhits}"
    if max_nhits is not None:
        save_name = save_name + f"_nhits<={max_nhits}"
    if min_radius is not None:
        save_name = save_name + f"_r>={min_radius}"
    if max_radius is not None:
        save_name = save_name + f"_r<={max_radius}"
    if min_energy is not None:
        save_name = save_name + f"_energy>={min_energy}"
    if max_energy is not None:
        save_name = save_name + f"_energy<{max_energy}"

    start_row = 0

    n_blocks = (dset_len - 1) // block_size + 1

    for _ in trange(n_blocks, desc="Block number"):
        block_slice = slice(start_row, min(start_row + block_size, dset_len))
        pos_block = np.stack([event["pos" + c][block_slice] for c in ["x", "y", "z"]], axis=-1)
        energy_block = event["energy"][block_slice]

        pos_block -= event["av_offset"][block_slice]
        r_block = np.linalg.vector_norm(pos_block, axis=-1)

        block_selector = np.ones(block_slice.stop - block_slice.start, dtype=np.bool)

        if min_radius is not None:
            block_selector &= r_block >= min_radius
        if max_radius is not None:
            block_selector &= r_block < max_radius
        if min_energy is not None:
            block_selector &= energy_block >= min_energy
        if max_energy is not None:
            block_selector &= energy_block < max_energy

        selector[block_slice] = block_selector
        start_row += block_size

    cut_indices = np.nonzero(selector)[0]
    save_name = save_name + ".h5"

    print(f"Length of original dataset: {dset_len}. Length of cut dataset: {len(cut_indices)}.")

    with h5py.File(save_path / save_name, "w") as h5_save:
        h5_save.attrs["checksum"] = h5_file.attrs["checksum"]
        h5_save.create_dataset("cut_indices", data=cut_indices)


if __name__ == "__main__":
    CLI(cuts)
