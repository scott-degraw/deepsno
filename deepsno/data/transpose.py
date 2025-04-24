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
    save_path: str | Path,
    block_size: int = 1_000_000,
    dset_keys: Iterable[str] | None = None,
    exclude_dset_keys: Iterable[str] | None = None,
):
    with h5py.File(h5_path, mode="r") as h5_file:
        id_dset = h5_file["cal_pmt_events/ids"]

        if dset_keys is None:
            dset_keys = h5_file["cal_pmt_events"].keys()

        if exclude_dset_keys is not None:
            dset_keys = set(dset_keys) - set(exclude_dset_keys)

        dset_keys = set(dset_keys) - set(["ids"])

        dset_dtypes: list[np.dtype] = [h5_file[f"cal_pmt_events/{dset_key}"].dtype for dset_key in dset_keys]

        dset_len = id_dset.shape[0]
        n_pmts = h5_file["pmt_info/position/x"].shape[0]

        n_blocks = (dset_len - 1) // block_size + 1

        id_counts = np.zeros(n_pmts, dtype=np.int64)

        start_row = 0
        for _ in trange(n_blocks, desc="Finding number of events per PMT"):
            id_block = id_dset[start_row : min(start_row + block_size, dset_len)].ravel()
            count_pmt_ids(id_block=id_block, id_counts=id_counts)

            start_row += block_size

        id_counts[0] = 0

        with h5py.File(save_path, mode="w") as output_h5:
            for dset_key, dset_dtype in zip(dset_keys, dset_dtypes, strict=True):
                output_h5.create_group(dset_key)
                for pmt_id in range(n_pmts):
                    output_h5[dset_key].create_dataset(
                        str(pmt_id),
                        shape=(id_counts[pmt_id],),
                        dtype=dset_dtype,
                    )

            dset_buffers = [np.zeros((n_pmts, block_size), dtype=dset_dtype) for dset_dtype in dset_dtypes]

            start_row = 0
            per_id_start_row = np.zeros(n_pmts, dtype=np.int64)
            for _ in trange(n_blocks, desc="Transposing data"):
                id_block = id_dset[start_row : min(start_row + block_size, dset_len)].ravel()
                for dset_key, dset_buffer in zip(dset_keys, dset_buffers, strict=True):
                    dset_block = h5_file[f"cal_pmt_events/{dset_key}"][start_row : min(start_row + block_size, dset_len)].ravel()

                    per_id_end = transpose_block(id_block=id_block, dset_block=dset_block, dset_buffer=dset_buffer)

                    for pmt_id in range(n_pmts):
                        end = per_id_end[pmt_id]
                        start = per_id_start_row[pmt_id]
                        output_h5[dset_key][str(pmt_id)][start : start + end] = dset_buffer[pmt_id, :end]

                per_id_start_row += per_id_end
                start_row += block_size


if __name__ == "__main__":
    CLI(transpose)
