import numpy as np
import uproot as ur

ALL_PASS = np.uint32(0x0)
ALL_FAIL = np.uint32(0xFFFFFFFF)


def filter_pmts(path: str, min_occupancy: float = 0.0, max_occupancy: float = 1.0):
    if min_occupancy < 0.0:
        raise ValueError("Minimum occupancy must be non-negative")
    if max_occupancy > 1.0:
        raise ValueError("Maximum occupancy must be at most 1.0")
    if min_occupancy > max_occupancy:
        raise ValueError("Minimum occupancy must be less than or equal to maximum occupancy")

    with ur.open({path: "transpose"}) as tree:
        transpose_tree = {key: value.array() for key, value in tree.items()}
        pmt_counts = transpose_tree["pmt_counts"]

    occupancy = pmt_counts / np.sum(pmt_counts)
    valid = (occupancy > min_occupancy) & (occupancy <= max_occupancy)
    status = np.where(valid, ALL_PASS, ALL_FAIL)
    transpose_tree["status"] = status

    with ur.update(path) as direc:
        del direc["transpose"]
        direc["transpose"] = transpose_tree


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter bad PMTs")

    parser.add_argument("path", type=str, help="Path to the merged dataset")
    parser.add_argument(
        "--min_occupancy",
        type=float,
        default=0.0,
        help="Minimum occupancy for PMT to be selected",
    )
    parser.add_argument(
        "--max_occupancy",
        type=float,
        default=1.0,
        help="Maximum occupancy for PMT to be selected",
    )

    args = parser.parse_args()

    filter_pmts(path=args.path, min_occupancy=args.min_occupancy, max_occupancy=args.max_occupancy)
