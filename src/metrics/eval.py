import numpy as np


def fwhm(hist: np.ndarray, bin_arrays: np.ndarray):
    bin_centers = np.convolve(bin_arrays, [0.5, 0.5], mode="valid")

    max_i = np.argmax(hist)
    maximum = hist[max_i]

    right_half_i = max_i + 1 + np.argmin(abs(hist[max_i + 1 :] - maximum / 2))
    left_half_i = np.argmin(abs(hist[:max_i] - maximum / 2))

    return bin_centers[right_half_i] - bin_centers[left_half_i]
