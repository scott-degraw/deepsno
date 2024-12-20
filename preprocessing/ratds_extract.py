#!/usr/bin/env python3
import argparse
from pathlib import Path

import ROOT
from rat import RAT

parser = argparse.ArgumentParser(description="Extract data from RATDS root files to an h5 file.")
parser.add_argument("-m", "--macro", type=str, help="Path to ROOT macro", required=True)
parser.add_argument("-i", "--input_file", type=str, help="Path to input ROOT file.", required=True)
parser.add_argument("-o", "--output_file", type=str, help="Path to output h5 file.", required=True)
parser.add_argument(
    "--min_hit_time",
    type=float,
    help="Minimum allowed hit time. Values larger than this will be clipped",
    required=True,
)
parser.add_argument(
    "--max_hit_time",
    type=float,
    help="Maximum allowed hit time. Values smaller than this will be clipped",
    required=True,
)
parser.add_argument(
    "--context_window", type=int, help="Maximum number of calibrated hit PMTs to be extracted.", required=True
)

args = parser.parse_args()

input_path = Path(args.input_file)
output_path = Path(args.output_file)


def check_file(path: Path):
    if not path.is_file():
        if path.is_dir():
            raise FileNotFoundError(f"'{str(path.resolve())} is a directory.")
        raise FileNotFoundError(f"File '{str(path.resolve())}' does not exist.")


check_file(input_path)
if not output_path.parent.is_dir():
    raise FileNotFoundError(f"Output_file '{str(output_path)}' does not have parent directory that exists.")

ROOT.gROOT.SetBatch(True)
RAT.DU.Utility.Get().LoadDBAndBeginRun()  # Database will not be loaded unless this is run

ROOT.gROOT.LoadMacro(args.macro)

ROOT.ratds_extract(
    str(input_path.resolve()), str(output_path.resolve()), args.min_hit_time, args.max_hit_time, args.context_window
)
