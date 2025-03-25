#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from time import time

sleep_time = 30  # seconds

parser = argparse.ArgumentParser(description="Extract data from RATDS root files to an h5 file.")
parser.add_argument("-m", "--macro", type=str, help="Path to ROOT macro", required=True)
parser.add_argument("-i", "--input_files", type=Path, nargs="+", help="Paths to input ROOT file.", required=True)
parser.add_argument("-o", "--output_files", type=Path, nargs="+", help="Paths to output h5 file.", required=True)
parser.add_argument(
    "--context_window", type=int, help="Maximum number of calibrated hit PMTs to be extracted.", required=True
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=2,
    help="Maximum number of attempts to extract data. Will reattempt if error is thrown by the macro.",
)
parser.add_argument("--ratdb_url", type=str, help="URL to the RATDB server.", required=False)

args = parser.parse_args()

for input_path in args.input_files:
    if not input_path.is_file():
        if input_path.is_dir():
            raise FileNotFoundError(f"'{str(input_path.resolve())} is a directory.")
        raise FileNotFoundError(f"File '{str(input_path.resolve())}' does not exist.")

for output_path in args.output_files:
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output_file '{str(output_path)}' does not have parent directory that exists.")

if args.ratdb_url is not None:
    print(f"Setting RATDB server URL: {args.ratdb_url}")
    os.environ["RATDBSERVER"] = args.ratdb_url

import ROOT  # noqa: E402
from rat import RAT  # noqa: E402

ROOT.gROOT.SetBatch(True)
RAT.DU.Utility.Get().LoadDBAndBeginRun()  # Database will not be loaded unless this is run

now = time()

ROOT.gROOT.LoadMacro(args.macro + "++")
print(f"Time to compile: {time() - now}")


def exception_loop(input_path: Path, output_path: Path, max_attempts: int) -> None:
    for attempt_num in range(max_attempts):
        try:
            ROOT.ratds_extract(str(input_path.resolve()), str(output_path.resolve()), args.context_window)
            break
        except Exception as e:
            if attempt_num < max_attempts - 1:
                print(f"An exception occurred: {e}")
                print(f"Trying again after {sleep_time} seconds, attempt_num: {attempt_num + 1}")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Error with extraction after {max_attempts} attempts: {e}")


for input_path, output_path in zip(args.input_files, args.output_files):
    exception_loop(input_path, output_path, max_attempts=args.max_attempts)
