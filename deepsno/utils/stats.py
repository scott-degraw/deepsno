import awkward as ak
import numba as nb
import numpy as np


@nb.njit
def clip(x: np.ndarray, a_min: float | int, a_max: float | int, out=None):
    return np.min(np.max(x, a_min), a_max)


@nb.njit
def digitize(x: float | np.ndarray, bins: np.ndarray):
    # No overflow bins are used
    # Assumes constant bin spacing
    # Returns i satsifying: bins[i - 1] <= x < bins[i]
    # If outside the bounds 0, or len(bins) is returned

    bin_width = bins[1] - bins[0]

    bin_i = int((x - bins[0]) / bin_width + 1)
    # bin_i = np.clip(bin_i, np.array(0), np.array(len(bins)))
    bin_i = clip(bin_i, 0, len(bins))

    return bin_i


@nb.njit
def hist_jagged(x: ak.Array, bins: np.ndarray, dtype=None) -> np.ndarray:
    hist = np.zeros((len(x), len(bins) - 1), dtype=dtype)

    for i in range(len(x)):
        row = x[i]
        for j in range(len(row)):
            bin_i = np.digitize(row[j], bins)
            if (0 < bin_i) and (bin_i < len(bins)):
                hist[i, bin_i - 1] += 1

    return hist
