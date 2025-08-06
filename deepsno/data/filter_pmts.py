import h5py
import numpy as np
import utils

def filter_pmts(h5_path: str, min_occupancy: float, max_occupancy: float):
    assert min_occupancy >= 0.0, "Minimum occupancy must be non-negative"
    assert max_occupancy <= 1.0, "Maximum occupancy must be at most 1.0"
    assert min_occupancy < max_occupancy, "Minimum occupancy must be less than maximum occupancy"

    if min_occupancy == 0.0 and max_occupancy == 1.0:
        print("No occupancy cut applied")

    with h5py.File(h5_path, "r+") as h5_file:
        id_counts = h5_file["pmt_info/id_counts"]

        print(f"Starting with {len(id_counts)} PMTs")

        occupancy = id_counts / np.sum(id_counts)

        valid_pmts = np.zeros(len(occupancy), dtype=np.bool)
        valid_pmts[(occupancy >= min_occupancy) & (occupancy <= max_occupancy)] = True

        print(f"{valid_pmts.sum()} PMTs left after occupancy cut")

        trans_qhs_group = h5_file["transpose/pmt/qhs"]
        for pmt_id in trans_qhs_group.keys():
            valid_pmts[int(pmt_id)] *= np.std(trans_qhs_group[pmt_id][:]) > 0

        print(f"{valid_pmts.sum()} PMTs left after bad QHS cut")

        # Convert these bools to 32 bit int words
        all_pass = np.uint32(0x0)
        all_fail = np.uint32(0xffffffff)
        status = np.where(valid_pmts, all_pass, all_fail)
        if "status" not in h5_file["pmt_info"]:
            h5_file["pmt_info"].create_dataset("status", data=status)
        else:
            h5_file["pmt_info/status"][:] = status
        
        print("Producing checksum")
        utils.checksum_h5_file(h5_path)


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
