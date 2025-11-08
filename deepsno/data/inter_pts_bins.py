import hist as h
import numpy as np
import tqdm


def find_min_range(values: np.array, q: float) -> tuple[int, int]:
    values /= np.sum(values)
    if len(values) < 3:
        return 0, len(values) - 1
    i = 0
    j = 1
    s = 0
    s += values[i]
    min_range = (0, len(values))
    while s < q and j < len(values):
        j += 1
        s += values[j - 1]
    while j < len(values):
        j += 1
        while s - values[i] >= q:
            i += 1
        if min_range[1] - min_range[0] > j - i:
            min_range = (i, j)

    return min_range


def weighted_counts(hist: h.Hist) -> np.ndarray:
    w_counts = hist.axes[0].edges[:-1] * hist.values()
    if np.all(w_counts != 0):
        w_counts /= hist.sum(flow=True)
    return w_counts


def inter_pts_bins(hists: h.Hist, min_occupancy: int = 100) -> np.ndarray:
    all_edges = []
    count_shift = 50
    first_q = 0.7
    second_q = 0.9
    n_peak_sections = 8

    n_peak_pts = n_peak_sections // 2 + n_peak_sections // 4

    # tail_edge_offsets = np.array([50, 100, 200, 300])
    tail_edge_offsets = np.array([])

    # Regions are defined by edges
    # First edge describes min value at which interpolation will be attempted
    # Then we have 4 points describing the first part of the peak
    # 2 points describing the rest of the peak
    # 4 points for the high charge region
    # Last edge describes the max value at which interpolation will be attempted
    n_pts = n_peak_pts + len(tail_edge_offsets)
    all_edges = np.zeros([hists.shape[0], (1 + n_pts + 1) + 1])

    for i in tqdm.tqdm(range(len(hists.axes["pmt_id"]))):
        hist = hists[i, :]
        pt_edges = all_edges[i]
        if hist.sum() < min_occupancy:
            continue
        min_range = find_min_range(weighted_counts(hist), first_q)

        end_qhs = hist[min_range[1]] + count_shift
        hist = hist[: h.loc(end_qhs)]

        if hist.sum() < min_occupancy:
            continue
        peak_range = find_min_range(weighted_counts(hist), second_q)

        edges = hist.axes[0].edges
        peak_sections = np.linspace(
            edges[peak_range[0]],
            edges[peak_range[1]],
            n_peak_sections + 1,
            endpoint=True,
        )

        assert len(peak_sections) % 2 == 1
        pt_edges[0] = (peak_sections[0] + peak_sections[1]) / 2 - (peak_sections[1] - peak_sections[0])

        split_point = len(peak_sections) // 2 + 1
        peak_edges = pt_edges[1 : 1 + n_peak_pts + 1]
        peak_edges[:split_point] = peak_sections[:split_point]
        peak_edges[split_point:] = peak_sections[split_point + 1 :: 2]

        tail_edges = peak_sections[-1] + tail_edge_offsets
        pt_edges[1 + n_peak_pts + 1 : -1] = tail_edges

        pt_edges[-1] = pt_edges[-2] + (pt_edges[-2] - pt_edges[-3]) / 2

    return all_edges
