#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path
from typing import Iterable


def split_files(files: list[str | Path], splits: Iterable[float], seed: int = 47281) -> list[list[Path]]:
    remainder = 0
    negative_split_i = -1
    files = [Path(f) for f in files]
    for i, split in enumerate(splits):
        if split >= 0:
            remainder += split
        else:
            if negative_split_i >= 0:
                raise ValueError("More than one split is negative")
            negative_split_i = i

    splits = list(splits)

    if negative_split_i >= 0:
        splits[negative_split_i] = 1 - remainder

    cum_sum = 0.0
    cum_splits = [0]
    for split in splits:
        cum_sum += split
        cum_splits.append(cum_sum)

    random.shuffle(files)

    split_indices = [int(cum_split * len(files)) for cum_split in cum_splits]

    return [files[low_i:high_i] for low_i, high_i in zip(split_indices[:-1], split_indices[1:])]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        parser = argparse.ArgumentParser(
            prog="split_files", description="Groups and splits files into different directories"
        )

        parser.add_argument("-f", "--files", type=Path, nargs="+", help="Files to group and split.")
        parser.add_argument(
            "-s",
            "--splits",
            type=float,
            nargs="+",
            help="Fractions to split files. Only one can be negative.",
            required=True,
        )
        parser.add_argument("-d", "--dirs", type=Path, nargs="+", help="Directories to put files into.", required=True)
        parser.add_argument("-S", "--seed", type=int, default=47281, help="Random seed")
        parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing links.")

        args = parser.parse_args()

        if len(args.splits) != len(args.dirs):
            raise ValueError(
                (
                    f"splits and dirs must be the same size"
                    f"but are sizes {len(args.splits)} and {len(args.dirs)} respectively"
                )
            )

        splits = split_files(args.files, args.splits, args.seed)

        for dest_dir, split_files in zip(args.dirs, splits):
            if dest_dir.exists() and not args.force:
                raise FileExistsError(f"Destination directory {dest_dir} exists. Use --force to overwrite.")
            dest_dir.unlink(missing_ok=True)
            for file in split_files:
                symlink_path = dest_dir / file.name
                symlink_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src=file, dst=symlink_path)

    except Exception as e:
        print(f"Exception encounted: {e}", file=sys.stderr)
        sys.exit(1)
