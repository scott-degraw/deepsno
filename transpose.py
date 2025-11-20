#!/usr/bin/env python

import pickle
from pathlib import Path

import awkward as ak
import h5py
import hist as h
import numpy as np
import uproot as ur
from jsonargparse import CLI
from tqdm import tqdm


def transpose(
    path: str | Path,
    branch_hist_kwargs: dict = {"pmt_qhs": {"bins": 150, "start": 0, "stop": 300, "label": "QHS (cap)"}},
    step_size: str | int = "100 MB",
    q: float = 0.98,
    epsilon: float = 1e-9,
):
    path = Path(path)
    if q < 0 or q > 1:
        raise ValueError("Quantile 'q' must be between 0 and 1.")
    hists = {}
    with ur.open({path: "pmt_info"}) as pmt_info:
        n_pmts = pmt_info.num_entries

    for branch, kwargs in branch_hist_kwargs.items():
        hist = h.Hist(
            h.axis.Integer(0, n_pmts, name="pmt_id", label="PMT ID"),
            h.axis.Regular(name=branch, **kwargs, transform=h.axis.transform.sqrt),
            # h.axis.Regular(name=branch, **kwargs),
        )
        for chunk in tqdm(
            ur.iterate({path: "event"}, step_size=step_size, library="ak", expressions=["pmt_id", branch])
        ):
            hist.fill(ak.flatten(chunk["pmt_id"]), ak.flatten(chunk[branch]))
        hists[branch] = hist

    for branch, hist in hists.items():
        hist_path = Path(path).parent / f"{branch}_hist.pkl"
        with open(hist_path, "wb") as f:
            pickle.dump(hist, f)
        print(f"Saved histogram for {branch} to {hist_path}")

    hist: h.Hist = hists["pmt_qhs"]
    qhs_counts = hist.values()
    qhs_widths = hist.axes[1].widths[None, :]
    # bad_ids = qhs_counts.sum(axis=1) < min_counts
    qhs_density = qhs_counts / (np.sum(qhs_counts, axis=1, keepdims=True) + epsilon) / qhs_widths
    cumul = np.cumsum(qhs_density, axis=1)
    quantile_i = np.zeros(n_pmts, dtype=np.int64)
    for pmt_id in range(n_pmts):
        quantile_i[pmt_id] = np.searchsorted(cumul[pmt_id, :], q * cumul[pmt_id, -1])
    n_pmts = qhs_density.shape[0]
    weights = 1 / (qhs_density + epsilon)
    # Normalize per PMT, avoid weighting by occupancy

    for pmt_id in range(n_pmts):
        weights[pmt_id, :] = np.clip(weights[pmt_id, :], 0, weights[pmt_id, quantile_i[pmt_id]])

    weights /= weights.sum(axis=1, keepdims=True)

    last_weight_column = weights[:, -2:-1]
    weights = np.concat([last_weight_column, weights, last_weight_column], axis=1)
    bad_pmt_weights = np.sum(~np.isfinite(weights), axis=1).astype(np.bool)
    weights[bad_pmt_weights] = 0.0
    assert np.all(np.isfinite(weights)), "Weights contain NaN or Inf values."
    qhs_bins = hist.axes[1].edges

    with h5py.File(path.parent / "qhs_weights.h5", "w") as h5_file:
        h5_file.create_dataset("qhs_bins", data=qhs_bins)
        h5_file.create_dataset("weights", data=weights)


if __name__ == "__main__":
    CLI(transpose)
