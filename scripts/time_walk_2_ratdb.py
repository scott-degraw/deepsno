#!/usr/bin/env python
import json
import math
from pathlib import Path

import numpy as np

from deepsno.models.hit_time_autoencoder import HitTimeAutoEncoder
from deepsno.utils.train import get_best_ckpt

all_pass = np.uint32(0x0)
all_fail = np.uint32(0xFFFFFFFF)


def convert_nans(l: list):
    for i, num in enumerate(l):
        if math.isnan(num):
            l[i] = -999_999.0

    return l


invalid_value = -999_999.0


def time_walk_2_ratdb(checkpoint: str, ratdb_output_name: str, config_file: str | None = None, run_range: list[int, int] | None = None) -> None:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        if checkpoint.name != "ckpt":
            checkpoint = checkpoint / "ckpt"
        checkpoint = get_best_ckpt(checkpoint)

    config_file = Path(config_file) if config_file else checkpoint.parent.parent / "config.yaml"

    tw = HitTimeAutoEncoder.time_walk_from_ckpt(checkpoint)
    status = tw["status"]

    invalid_pmts = (status & all_fail).astype(bool)

    tw["time_scale"][invalid_pmts] = invalid_value
    tw["qhs_scale"][invalid_pmts] = invalid_value
    tw["gradient"][invalid_pmts] = invalid_value
    tw["intercept"] -= np.median(tw["intercept"][~invalid_pmts])
    tw["intercept"][invalid_pmts] = invalid_value

    ratdb_table = {
        "type": "PCA_TW",
        "run_range": tw["run_range"],
        "version": 1,
        "pass": 0,
        "comment": "Exponential time walk parameters. Model used is: timewalk(q) = a * exp(-q / b) + c * q + d",
        "tw_type": "exponential",
        "min_qhs": 0.0,
        "max_qhs": 1000,
        "min_time_scale": 0.0,
        "max_time_scale": 100.0,
        "min_qhs_scale": 0.0,
        "min_gradient": -0.1,
        "max_gradient": 0.0,
        "min_intercept": -200.0,
        "max_intercept": 200.0,
        "time_scale": convert_nans(tw["time_scale"].tolist()),
        "qhs_scale": convert_nans(tw["qhs_scale"].tolist()),
        "gradient": convert_nans(tw["gradient"].tolist()),
        "intercept": convert_nans(tw["intercept"].tolist()),
        "PCATW_status": convert_nans(tw["status"].tolist()),
    }

    with open(ratdb_output_name, "w") as f:
        json.dump(ratdb_table, f, allow_nan=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert time walk parameters to RATDB format")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("ratdb_output_name", type=str, help="Output RATDB file name")
    parser.add_argument("--config_file", type=str, default=None, help="Path to the config file")
    parser.add_argument("--run_range", type=int, nargs=2, help="Minimum run number")

    args = parser.parse_args()

    time_walk_2_ratdb(args.checkpoint, args.ratdb_output_name, args.config_file, run_range=args.run_range)
