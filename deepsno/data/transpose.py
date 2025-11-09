#!/usr/bin/env python

import pickle
from pathlib import Path

import awkward as ak
import h5py
import hist as h
import numba as nb
import numpy as np
import uproot as ur
from jsonargparse import CLI
from tqdm import tqdm


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
    path: str | Path,
    min_qhs: float = 0.0,
    max_qhs: float = 300.0,
    step_size: str | int = "100 MB",
    q: float = 0.98,
    epsilon: float = 1e-9,
):
    path = Path(path)
    if q < 0 or q > 1:
        raise ValueError("Quantile 'q' must be between 0 and 1.")
    with ur.open({path: "pmt_info"}) as pmt_info:
        n_pmts = pmt_info.num_entries

    # The actual QHS values are stored in the centres of the bins
    hist = h.Hist(
        h.axis.Integer(0, n_pmts, name="pmt_id", label="PMT ID"),
        h.axis.Variable(
            np.arange(min_qhs - 0.25, max_qhs + 0.3, 0.5),  # 0.3 to include max_qhs in the last bin
            name="qhs",
            label="QHS (cap)",
        ),
    )
    for chunk in tqdm(
        ur.iterate({path: "event"}, step_size=step_size, library="ak", expressions=["pmt_id", "pmt_qhs"])
    ):
        hist.fill(ak.flatten(chunk["pmt_id"]), ak.flatten(chunk["pmt_qhs"]))

    hist_path = path.parent / "pmt_qhs_hist.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(hist, f)
    print(f"Saved histogram for pmt_qhs to {hist_path}")




if __name__ == "__main__":
    CLI(transpose)
