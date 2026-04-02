#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path
from typing import Iterable


def shuffle_files(files: list[str | Path], splits: Iterable[float], seed: int = 47281) -> list[list[Path]]:
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


def split_files(
    splits: list[float], dirs: list[str | Path], files: list[str | Path], seed: int = 47281, force: bool = False
):
    if len(splits) != len(dirs):
        raise ValueError(
            (f"splits and dirs must be the same sizebut are sizes {len(splits)} and {len(dirs)} respectively")
        )

    splits = shuffle_files(files, splits, seed)

    for dest_dir, split_files in zip(dirs, splits):
        dest_dir = Path(dest_dir)
        if dest_dir.exists() and not force:
            raise FileExistsError(f"Destination directory {dest_dir} exists. Use --force to overwrite.")
        dest_dir.unlink(missing_ok=True)
        for file in split_files:
            symlink_path = dest_dir / file.name
            symlink_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src=file.resolve(), dst=symlink_path.resolve())


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="split_files", description="Groups and splits files into different directories"
    )

    parser.add_argument("-f", "--files", type=Path, nargs="+", help="Files to group and split.")
    parser.add_argument(
        "-g",
        "--glob",
        type=str,
        help="Glob pattern to find files (e.g. '/data/dir/*.h5'). Used if --files is not provided.",
    )
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
    parser.add_argument("--force", action="store_true", help="Overwrite existing links.")

    args = parser.parse_args()

    if args.files:
        files = args.files
    elif args.glob:
        import glob

        files = [Path(p) for p in glob.glob(args.glob)]
        if not files:
            print(f"No files matched glob pattern: {args.glob}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: must provide --files or --glob.", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    split_files(splits=args.splits, dirs=args.dirs, files=files, seed=args.seed, force=args.force)
