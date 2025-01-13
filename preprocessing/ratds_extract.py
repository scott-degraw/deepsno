#!/usr/bin/env python3
import argparse
from pathlib import Path

import ROOT
from rat import RAT

parser = argparse.ArgumentParser(description="Extract data from RATDS root files to an h5 file.")
parser.add_argument("-m", "--macro", type=str, help="Path to ROOT macro", required=True)
parser.add_argument("-i", "--input_files", type=Path, nargs="+", help="Paths to input ROOT file.", required=True)
parser.add_argument("-o", "--output_files", type=Path, nargs="+", help="Paths to output h5 file.", required=True)
parser.add_argument(
    "--min_hit_time",
    type=float,
    help="Minimum allowed hit time. Values larger than this will be clipped.",
    required=True,
)
parser.add_argument(
    "--max_hit_time",
    type=float,
    help="Maximum allowed hit time. Values smaller than this will be clipped.",
    required=True,
)
parser.add_argument(
    "--context_window", type=int, help="Maximum number of calibrated hit PMTs to be extracted.", required=True
)

args = parser.parse_args()

for input_path in args.input_files:
    if not input_path.is_file():
        if input_path.is_dir():
            raise FileNotFoundError(f"'{str(input_path.resolve())} is a directory.")
        raise FileNotFoundError(f"File '{str(input_path.resolve())}' does not exist.")

for output_path in args.output_files:
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output_file '{str(output_path)}' does not have parent directory that exists.")

ROOT.gROOT.SetBatch(True)
RAT.DU.Utility.Get().LoadDBAndBeginRun()  # Database will not be loaded unless this is run

ROOT.gROOT.LoadMacro(args.macro)

for input_path, output_path in zip(args.input_files, args.output_files):
    ROOT.ratds_extract(
        str(input_path.resolve()), str(output_path.resolve()), args.min_hit_time, args.max_hit_time, args.context_window
    )
