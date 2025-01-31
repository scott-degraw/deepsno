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
    min_radius: float | None = None,
    max_radius: float | None = None,
    min_hit_time: float | None = None,
    max_hit_time: float | None = None,
    block_size: int = 100_000,
):
    save_path = Path(save_path)
    if not save_path.is_dir():
        raise ValueError("'save_path' is not a directory")
    if not save_path.exists():
        raise ValueError(f"'save_path' directory '{save_path}' does not exist")

    h5_file = h5py.File(h5_path)

    id_dset = h5_file["cal_pmt_events/ids"]
    pos_group = h5_file["mc_truth/position"]
    hit_time_dset = h5_file["cal_pmt_events/hit_times"]
    dset_len = id_dset.shape[0]

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
    if min_hit_time is not None:
        save_name = save_name + f"_hit_time>={min_hit_time}"
    if max_hit_time is not None:
        save_name = save_name + f"_hit_time<={max_hit_time}"

    start_row = 0

    n_blocks = (dset_len - 1) // block_size

    for _ in trange(n_blocks, desc="Block number"):
        block_slice = slice(start_row, min(start_row + block_size, dset_len))
        id_block = id_dset[block_slice]
        pos_block = np.stack([pos_group[c][block_slice] for c in ["x", "y", "z"]])
        r_block = np.linalg.vector_norm(pos_block, axis=0)

        if min_hit_time is not None or max_hit_time is not None:
            hit_time_block = hit_time_dset[block_slice]
            if min_hit_time is not None:
                id_block[hit_time_block < min_hit_time] = 0
            if max_hit_time is not None:
                id_block[hit_time_block > max_hit_time] = 0

        block_nhits = np.count_nonzero(id_block, axis=1)
        block_selector = np.ones(block_slice.stop - block_slice.start, dtype=np.bool)

        if min_nhits is not None:
            block_selector &= block_nhits >= min_nhits
        if max_nhits is not None:
            block_selector &= block_nhits <= max_nhits
        if min_radius is not None:
            block_selector &= r_block >= min_radius
        if max_radius is not None:
            block_selector &= r_block <= max_radius

        selector[block_slice] = block_selector
        start_row += block_size

    cut_indices = np.nonzero(selector)[0]
    save_name = save_name + ".h5"

    print(f"Length of original dataset: {dset_len}. Length of cut dataset: {len(cut_indices)}.")

    with h5py.File(save_path / save_name, "w") as h5_save:
        h5_save.create_dataset("cut_indices", data=cut_indices)


if __name__ == "__main__":
    CLI(cuts)
