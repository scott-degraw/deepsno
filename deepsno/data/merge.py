#!/usr/bin/env -S python3 -u
import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List

import awkward as ak
import h5py
import numpy as np
import tqdm
import uproot as ur
from filter_pmts import filter_pmts
from transpose import transpose
from utils import checksum_file, str_from_many_paths


def check_files(input_paths: Iterable[str], tree: str, branches: Iterable[str]) -> None:
    for input_path in tqdm.tqdm(input_paths):
        with ur.open(f"{input_path}:{tree}") as t:
            for branch in branches:
                if branch not in t:
                    raise KeyError(f"Branch '{branch}' not found in {input_path}")


def merge(
    input_paths: str,
    output_path: str,
    branches: Iterable[str],
    pmt_branches: Iterable[str],
    cut: str = "",
    pmt_info_branches: Iterable[str] = None,
    parameters: Iterable[str] = None,
) -> None:
    runs = set()
    all_branches = branches + pmt_branches

    total_n_events = 0
    for input_path in input_paths:
        with ur.open(f"{input_path}:event") as events_tree:
            arrays = events_tree.arrays("runID", cut=cut)
            total_n_events += len(arrays)
            runs.add(arrays[0]["runID"])

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["run_range"] = [min(runs), max(runs)]
        h5_file.attrs["cut"] = cut if cut else ""
        event_group = h5_file.create_group("event")
        pmt_group = h5_file.create_group("pmt")
        if parameters is not None:
            with ur.open(input_paths[0]) as input_file:
                for parameter in parameters:
                    h5_file.attrs[parameter] = input_file[parameter].value

        events_tree = ur.open(f"{input_paths[0]}:event")
        arrays = events_tree.arrays(all_branches, library="np", entry_start=0, entry_stop=1)
        for dset_name in branches:
            dset = arrays[dset_name]
            dtype = dset.dtype
            if dset.ndim == 1:
                shape = (total_n_events,)
            elif dset.ndim == 2:
                shape = (total_n_events, dset.shape[1])
            else:
                raise ValueError(f"Unexpected dimension {dset.ndim} for dataset '{dset_name}' in {input_paths[0]}")

            event_group.create_dataset(dset_name, shape=shape, dtype=dtype)

        for pmt_branch in pmt_branches:
            dset_name = pmt_branch.replace("pmt_", "")
            dset = arrays[pmt_branch]
            dtype = h5py.vlen_dtype(dset[0].dtype)
            shape = (total_n_events,)
            pmt_group.create_dataset(dset_name, shape=shape, dtype=dtype)

        start_row = 0
        for input_path in tqdm.tqdm(input_paths):
            with ur.open(f"{input_path}:event") as events_tree:
                arrays = events_tree.arrays(all_branches, library="np", cut=cut)

                for dset_name, array in arrays.items():
                    if dset_name.startswith("pmt_"):
                        dset_name = dset_name.replace("pmt_", "")
                        dset = pmt_group[dset_name]
                    else:
                        dset = event_group[dset_name]
                    block_size = array.shape[0]
                    dset[start_row : start_row + block_size] = array
            start_row += block_size

        input_path = input_paths[0]
        pmt_info_group = h5_file.create_group("pmt_info")
        with ur.open(f"{input_path}:pmt_info") as pmt_info_tree:
            dsets = pmt_info_tree.arrays(library="ak")
            for pmt_info_ident in pmt_info_branches:
                pmt_info_group.create_dataset(pmt_info_ident, data=ak.to_numpy(dsets[pmt_info_ident]))


def main(
    input_paths: List[str],
    train_output_path: str,
    cut: str = "",
    train_test_split: float = 1.0,
    test_output_path: str | None = None,
    only_count: bool = True,
    min_occupancy: float = 0.0,
    max_occupancy: float = 1.0,
    seed: int = 487391,
    condor_transfer_input_files: bool = False,
    condor_transfer_output_files: bool = False,
) -> None:
    pmt_branches = ["pmt_hit_time", "pmt_qhs", "pmt_id"]
    branches = ["av_offset", "posx", "posy", "posz", "energy"]
    parameters = ["inner_av_radius", "av_thickness"]

    pmt_info_idents = ["pos"]

    create_test_set = train_test_split < 1.0 and test_output_path is not None

    if condor_transfer_input_files:
        condor_scratch_dir = Path(os.environ["_CONDOR_SCRATCH_DIR"])
        print("Copying files to Condor scratch disk")
        for i in tqdm.trange(len(input_paths)):
            shutil.copy(input_paths[i], condor_scratch_dir)

        final_train_output_path = train_output_path
        if create_test_set:
            final_test_output_path = test_output_path

        input_paths = [condor_scratch_dir / Path(path).name for path in input_paths]

    if condor_transfer_output_files:
        train_output_path = condor_scratch_dir / Path(train_output_path).name
        if create_test_set:
            test_output_path = condor_scratch_dir / Path(test_output_path).name

    generator = np.random.default_rng(seed)
    input_path_indices = generator.choice(np.arange(len(input_paths)), size=len(input_paths), replace=False)

    split_index = np.floor(train_test_split * len(input_path_indices)).astype(np.int64)

    train_path_indices = input_path_indices[:split_index]
    test_path_indices = input_path_indices[split_index:]

    train_input_paths = [input_paths[i] for i in train_path_indices]
    test_input_paths = [input_paths[i] for i in test_path_indices]

    if len(test_input_paths) == 0 and train_test_split < 1.0:
        raise RuntimeError(f"Not enough files. No test dataset for {train_test_split:.3g} train-test split.")

    print(f"Train input paths: {str_from_many_paths(train_input_paths)}")

    print("Merge train files")
    merge(
        input_paths=train_input_paths,
        output_path=train_output_path,
        cut=cut,
        branches=branches,
        pmt_branches=pmt_branches,
        parameters=parameters,
        pmt_info_branches=pmt_info_idents,
    )

    if test_output_path is not None:
        print("Merge test files")
        if train_test_split < 1.0 and test_output_path is not None:
            print("Merge test files")
            merge(
                input_paths=test_input_paths,
                output_path=test_output_path,
                cut=cut,
                branches=branches,
                pmt_branches=pmt_branches,
                parameters=parameters,
                pmt_info_branches=pmt_info_idents,
            )

    print("Transpose train files")
    transpose(train_output_path, only_count=only_count)

    print("Filter PMTs")

    filter_pmts(h5_path=train_output_path, min_occupancy=min_occupancy, max_occupancy=max_occupancy)

    if test_output_path is not None:
        print("Hasing test dataset")
        checksum_file(test_output_path)

    if condor_transfer_output_files:
        print("Transferring output files back")
        shutil.copy(train_output_path, final_train_output_path)
        if create_test_set:
            shutil.copy(test_output_path, final_test_output_path)

    print("Finished merging and proprocessing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and preprocess DeepSNO data files.")

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("min_run", type=int)
    parser.add_argument("max_run", type=int)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--min_occupancy", type=float, default=0.0)
    parser.add_argument("--max_occupancy", type=float, default=1.0)
    parser.add_argument("-t", "--transpose", action="store_true")
    parser.add_argument("-c", "--cut", type=str)
    parser.add_argument("--transfer", action="store_true")

    args = parser.parse_args()

    all_input_paths = args.input_dir.glob("**/*.pmt.root")

    min_run = args.min_run
    max_run = args.max_run
    dest = args.dest

    if dest.is_file():
        raise ValueError(f"Destination {dest} is a file, but it should be a directory.")

    input_paths = []
    runs = set()
    for input_path in all_input_paths:
        match = re.search(r"(?<=_r)(\d+)", str(input_path))
        if match:
            run = int(match.group(0))
            runs.add(run)
            if min_run <= run <= max_run:
                input_paths.append(str(input_path))
        else:
            raise ValueError(f"Could not find run number in {input_path}")

    print(f"Found {len(input_paths)} input files in range [{min_run}, {max_run}]")

    # with ur.recreate(args.dest) as output_file:
    #     tree_created = False
    #     for chunk in tqdm.tqdm(
    #         ur.iterate(
    #             {f: "event" for f in input_paths},
    #             step_size="100 MB",
    #             library="ak",
    #             expressions=["av_offset", "pmt_hit_time"],
    #         )
    #     ):
    #         if not tree_created:
    #             output_file["event"] = {"event": chunk}
    #             tree_created = True
    #         else:
    #             output_file["event"].extend({"event": chunk})

    main(
        input_paths=input_paths,
        train_output_path=dest / f"train_dset_{min(runs)}-{max(runs)}.h5",
        cut=args.cut,
        only_count=not args.transpose,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy,
        condor_transfer_input_files=args.transfer,
        condor_transfer_output_files=args.transfer,
    )
