#!/usr/bin/env python3

import numpy as np
import pandas as pd
from jsonargparse import CLI


def active_pmt_remap(
    csv_path: str,
    active_types: tuple[str, ...] = ("NORMAL", "HQE"),
) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact PMT remapping derived from pmt_info.csv.

    Args:
        csv_path: Path to the CSV produced by pmt_info_dump.
        active_types: PMT type names to treat as active.

    Returns:
        active_ids: Sorted raw PMT IDs that are active, shape (n_active,).
        remap: Array of length max_raw_id + 1 where remap[raw_id] = compact_id
               for active PMTs and -1 for inactive ones.
    """
    df = pd.read_csv(csv_path, index_col="pmt_id")
    active_ids = np.sort(df.index[df["type_name"].isin(active_types)].to_numpy())
    remap = np.full(int(df.index.max()) + 1, -1, dtype=np.int64)
    remap[active_ids] = np.arange(len(active_ids), dtype=np.int64)
    return active_ids, remap


def rat_import():
    from rat import RAT
    return RAT


def pmt_info_dump(output_path: str):
    RAT = rat_import()
    RAT.DU.Utility.Get().LoadDBAndBeginRun()
    pmt_info = RAT.DU.Utility.Get().GetPMTInfo()

    EPMTType = RAT.DU.PMTInfo.EPMTType
    pmt_type_int_2_name = {}
    for attribute in dir(EPMTType):
        if isinstance(getattr(EPMTType, attribute), EPMTType):
            pmt_type_int_2_name[int(getattr(EPMTType, attribute))] = attribute

    n_pmts = pmt_info.GetCount()
    pmt_positions = np.zeros((n_pmts, 3), dtype=np.float64)
    pmt_type_ints = np.zeros((n_pmts,), dtype=np.int32)
    pmt_type_names = []

    for pmt_id in range(n_pmts):
        pmt_positions[pmt_id] = pmt_info.GetPosition(pmt_id)
        pmt_type = pmt_info.GetType(pmt_id)
        pmt_type_ints[pmt_id] = pmt_type
        pmt_type_names.append(pmt_type_int_2_name[pmt_type])

    df = pd.DataFrame(
        {
            "position_x": pmt_positions[:, 0],
            "position_y": pmt_positions[:, 1],
            "position_z": pmt_positions[:, 2],
            "type_int": pmt_type_ints,
            "type_name": pmt_type_names,
        }
    )

    df.index.name = "pmt_id"
    df.to_csv(output_path)


if __name__ == "__main__":
    CLI(pmt_info_dump)
