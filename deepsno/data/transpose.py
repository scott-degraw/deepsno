#!/usr/bin/env python

from pathlib import Path
from typing import Iterable

import h5py
import numba as nb
import numpy as np
from jsonargparse import CLI
from tqdm import trange


@nb.njit
def transpose_block(id_block: np.ndarray, dset_block: np.ndarray, dset_buffer: np.ndarray) -> np.ndarray:
    per_id_end = np.zeros(dset_buffer.shape[0], dtype=np.int64)
    for i in range(len(id_block)):
        pmt_id = id_block[i]
        if pmt_id > 0:
            dset_buffer[pmt_id, per_id_end[pmt_id]] = dset_block[i]
            per_id_end[pmt_id] += 1

    return per_id_end


@nb.njit
def count_pmt_ids(id_block: np.ndarray, id_counts: np.ndarray) -> None:
    for pmt_id in id_block:
        id_counts[pmt_id] += 1


def transpose(
    h5_path: str | Path,
    block_size: int = 1_000_000,
    groups: Iterable[str] = ["cal"],
):
    with h5py.File(h5_path, mode="r+") as h5_file:
        id_dset = h5_file["cal/pmt_ids"]

        dset_len = id_dset.shape[0]
        n_pmts = h5_file["pmt_info/pos"].shape[0]

        n_blocks = (dset_len - 1) // block_size + 1

        id_counts = np.zeros(n_pmts, dtype=np.int64)

        start_row = 0
        for _ in trange(n_blocks, desc="Finding number of events per PMT"):
            id_block = np.concat(id_dset[start_row : min(start_row + block_size, dset_len)])
            count_pmt_ids(id_block=id_block, id_counts=id_counts)

            start_row += block_size

        id_counts[0] = 0

        transpose_group = h5_file.create_group("transpose")
        for group in groups:
            h5_group = h5_file[group]
            output_h5_group = transpose_group.create_group(group)

            dset_buffers = []
            for dset_key in h5_group.keys():
                dset_group = output_h5_group.create_group(dset_key)
                dset_dtype = h5_group[dset_key][0].dtype
                for pmt_id in range(n_pmts):
                    dset_group.create_dataset(
                        str(pmt_id),
                        shape=(id_counts[pmt_id],),
                        dtype=dset_dtype,
                    )
                dset_buffers.append(np.zeros((n_pmts, block_size), dtype=dset_dtype))

            start_row = 0
            per_id_start_row = np.zeros(n_pmts, dtype=np.int64)
            for _ in trange(n_blocks, desc=f"Transposing {group}"):
                id_block = np.concat(id_dset[start_row : min(start_row + block_size, dset_len)])
                for dset_key, dset_buffer in zip(h5_group.keys(), dset_buffers, strict=True):
                    dset_block = np.concat(h5_group[dset_key][start_row : min(start_row + block_size, dset_len)])

                    per_id_end = transpose_block(id_block=id_block, dset_block=dset_block, dset_buffer=dset_buffer)

                    for pmt_id in range(n_pmts):
                        end = per_id_end[pmt_id]
                        start = per_id_start_row[pmt_id]
                        output_h5_group[dset_key][str(pmt_id)][start : start + end] = dset_buffer[pmt_id, :end]

                per_id_start_row += per_id_end
                start_row += block_size


if __name__ == "__main__":
    CLI(transpose)
