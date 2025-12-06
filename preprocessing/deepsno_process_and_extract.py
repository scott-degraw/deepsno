#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from gridutils import apptainer_runner, download

"""
Steps
1. Download zdab
2. Download ntuple
3. Produce event lists from cut and ntuple
4. Perform 2nd pass processing with eventlist
5. Perform PMT hit extraction on processed data that also includes the ntuple info
"""

parser = argparse.ArgumentParser()

parser.add_argument("--ratdb_url", type=str)
parser.add_argument("--container", type=str)
parser.add_argument("--max_retries", type=int, default=5)

parser.add_argument("--ntuple", type=str)
parser.add_argument("--ntuple_adler", type=str)
parser.add_argument("--filter", type=str, default="")

parser.add_argument("--zdab", type=str)
parser.add_argument("--zdab_adler", type=str)
parser.add_argument("--rat_args", type=str, nargs="+")

parser.add_argument("--extract_input", type=Path, required=True)
parser.add_argument("--min_ht", type=float, required=True)
parser.add_argument("--max_ht", type=float, required=True)
parser.add_argument("--min_qhs", type=float, required=True)
parser.add_argument("--max_qhs", type=float, required=True)

args = parser.parse_args()

env = {"RATDBSERVER": args.ratdb_url}

# Download zdab and ntuple
download(args.zdab, args.zdab_adler, max_retries=args.max_retries)
if args.ntuple is not None:
    download(args.ntuple, args.ntuple_adler, max_retries=args.max_retries)

ntuple = Path(args.ntuple).name

# Create eventlist.txt
apptainer_args = [
    "root",
    "-l",
    "-q",
    "-b",
    "-x",
    f'ntuple_2_eventlist.C("{ntuple}", "{args.filter}")',
]
apptainer_runner(container=args.container, args=apptainer_args, timeout=60 * 60 * 1)

# Check if eventlist.txt is empty
with open("eventlist.txt", "r") as f:
    if not f.read().strip():
        print("'eventlist.txt' is empty, exiting.")
        sys.exit(0)

# Perform second pass processing

apptainer_args = ["rat", *args.rat_args]
apptainer_runner(container=args.container, args=apptainer_args, timeout=60 * 60 * 12, env=env)

# Perform PMT hit extraction
macro_args = [
    f'"{str(args.extract_input)}"',
    f'"{Path(args.extract_input).with_suffix(".pmt.root")}"',
    str(args.min_ht),
    str(args.max_ht),
    str(args.min_qhs),
    str(args.max_qhs),
    f'"{ntuple}"',
    f'"{args.filter}"',
    "1",  # bool
]
macro_args = ", ".join(macro_args)
apptainer_args = ["root", "-l", "-b", "-q", "-x", f"ratds_extract.C({macro_args})"]
apptainer_runner(container=args.container, args=apptainer_args, timeout=60 * 60 * 2, env=env)
