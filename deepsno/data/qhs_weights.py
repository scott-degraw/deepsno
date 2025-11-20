from pathlib import Path
from 

import h5py
import hist as h
import numpy as np


def main(qhs_hist: h.Hist, q: float, epsilon: float = 1e-9) -> h.Hist:
    qhs_counts = qhs_hist.values()
    n_pmts = qhs_density.shape[0]
    qhs_widths = qhs_hist.axes[1].widths[None, :]
    # bad_ids = qhs_counts.sum(axis=1) < min_counts
    qhs_density = qhs_counts / (np.sum(qhs_counts, axis=1, keepdims=True) + epsilon) / qhs_widths
    cumul = np.cumsum(qhs_density, axis=1)
    quantile_i = np.zeros(n_pmts, dtype=np.int64)
    for pmt_id in range(n_pmts):
        quantile_i[pmt_id] = np.searchsorted(cumul[pmt_id, :], q * cumul[pmt_id, -1])
    weights = 1 / (qhs_density + epsilon)
    # Normalize per PMT, avoid weighting by occupancy

    for pmt_id in range(n_pmts):
        weights[pmt_id, :] = np.clip(weights[pmt_id, :], 0, weights[pmt_id, quantile_i[pmt_id]])
    weights /= weights.sum(axis=1, keepdims=True)
    last_weight_column = weights[:, -2:-1]
    weights = np.concat([last_weight_column, weights, last_weight_column], axis=1)
    bad_pmt_weights = np.sum(~np.isfinite(weights), axis=1).astype(bool)
    weights[bad_pmt_weights] = 0.0
    assert np.all(np.isfinite(weights)), "Weights contain NaN or Inf values."
    qhs_bins = qhs_hist.axes[1].edges

    with h5py.File(path.parent / "qhs_weights.h5", "w") as h5_file:
        h5_file.create_dataset("qhs_bins", data=qhs_bins)
        h5_file.create_dataset("weights", data=weights)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute QHS weights from histogram.")
    parser.add_argument("path", type=Path, help="Path to output file")
    par

    args = parser.parse_args()

    main(args.path, hist, args.q, args.epsilon)
