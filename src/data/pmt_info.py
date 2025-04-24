#!/usr/bin/env python3

import numpy as np
import pandas as pd
from jsonargparse import CLI
from rat import RAT


def pmt_info_dump(output_path: str):
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
