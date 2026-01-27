#!/usr/bin/env python3
import os
import random
import sys

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        parser = argparse.ArgumentParser(
            prog="split_files", description="Groups and splits files into different directories"
        )

        parser.add_argument("-f", "--files", type=Path, nargs="+", help="Files to group and split.")
        parser.add_argument(
            "-s", "--splits", type=float, nargs="+", help="Fractions to split files. Only one can be negative.", required=True
        )
        parser.add_argument("-d", "--dirs", type=Path, nargs="+", help="Directories to put files into.", required=True)
        parser.add_argument("-S", "--seed", type=int, default=47281, help="Random seed")

        args = parser.parse_args()

        if len(args.splits) != len(args.dirs):
            raise ValueError(
                (
                    f"splits and dirs must be the same size"
                    f"but are sizes {len(args.splits)} and {len(args.dirs)} respectively"
                )
            )

        files = args.files

        remainder = 0
        negative_split_i = -1
        for i, split in enumerate(args.splits):
            if split >= 0:
                remainder += split
            else:
                if negative_split_i >= 0:
                    raise ValueError("More than one split is negative")
                negative_split_i = i

        if negative_split_i >= 0:
            args.splits[negative_split_i] = 1 - remainder

        cum_sum = 0.0
        cum_splits = [0]
        for split in args.splits:
            cum_sum += split
            cum_splits.append(cum_sum)

        random.shuffle(files)

        split_indices = [int(cum_split * len(files)) for cum_split in cum_splits]

        for dest_dir, low_i, high_i in zip(args.dirs, split_indices[:-1], split_indices[1:]):
            split_files = files[low_i:high_i]
            for file in split_files:
                symlink_path = dest_dir / file.name
                symlink_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src=file, dst=symlink_path)

    except Exception as e:
        print(f"Exception encounted: {e}", file=sys.stderr)
        sys.exit(1)
