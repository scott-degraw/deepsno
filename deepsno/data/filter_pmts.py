import h5py
import numpy as np


def filter_pmts(h5_path: str, min_occupancy: float, max_occupancy: float):
    assert min_occupancy >= 0.0, "Minimum occupancy must be non-negative"
    assert max_occupancy <= 1.0, "Maximum occupancy must be at most 1.0"
    assert min_occupancy < max_occupancy, "Minimum occupancy must be less than maximum occupancy"

    if min_occupancy == 0.0 and max_occupancy == 1.0:
        print("No occupancy cut applied")

    with h5py.File(h5_path, "r+") as h5_file:
        trans_qhs_group = h5_file["transpose/pmt/qhs"]
        n_pmts = h5_file["pmt_info/pos"].shape[0]
        id_counts = np.zeros(n_pmts, dtype=np.int64)
        for pmt_id in trans_qhs_group.keys():
            id_counts[int(pmt_id)] = trans_qhs_group[pmt_id].shape[0]

        print(f"Starting with {len(id_counts)} PMTs")

        occupancy = id_counts / np.sum(id_counts)

        statuses = np.zeros(len(occupancy), dtype=np.bool)
        statuses[(occupancy >= min_occupancy) & (occupancy <= max_occupancy)] = True

        print(f"{statuses.sum()} PMTs left after occupancy cut")

        for pmt_id in trans_qhs_group.keys():
            statuses[int(pmt_id)] *= np.std(trans_qhs_group[pmt_id][:]) > 0

        print(f"{statuses.sum()} PMTs left after bad QHS cut")

        if "statuses" not in h5_file["pmt_info"]:
            h5_file["pmt_info"].create_dataset("statuses", data=statuses)

        h5_file["pmt_info/statuses"][:] = statuses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter bad PMTs")

    parser.add_argument("h5_path", type=str, help="Path to the HDF5 file")
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

    filter_pmts(h5_path=args.h5_path, min_occupancy=args.min_occupancy, max_occupancy=args.max_occupancy)
