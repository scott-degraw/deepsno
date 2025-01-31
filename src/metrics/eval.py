from warnings import warn

import numpy as np


def fwhm(counts: np.ndarray, bin_arrays: np.ndarray) -> float:
    if len(counts) + 1 != len(bin_arrays):
        raise ValueError(
            (
                "Expected 'len(counts) + 1 == len(bin_arrays)' but instead "
                f"got 'len(counts) == {len(counts)}' and 'len(bin_arrays) == {len(bin_arrays)}"
            )
        )
    if np.any(np.isnan(counts)):
        return np.nan
    bin_centers = np.convolve(bin_arrays, [0.5, 0.5], mode="valid")

    max_i = np.argmax(counts)
    if max_i == 0 or max_i == len(counts) - 1:
        warn(RuntimeWarning("Data does not have a local maximum. Returning NaN."))
        return np.nan
    maximum = counts[max_i]

    right_half_i = max_i + 1 + np.argmin(abs(counts[max_i + 1 :] - maximum / 2))
    left_half_i = np.argmin(abs(counts[:max_i] - maximum / 2))

    return bin_centers[right_half_i] - bin_centers[left_half_i]
