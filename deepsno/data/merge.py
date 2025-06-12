#!/usr/bin/env -S python3 -u
import hashlib
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
from jsonargparse import CLI
from transpose import transpose


def hash_file(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def str_from_many_paths(paths: tuple[str], n=3) -> str:
    paths = sorted(paths)
    if len(paths) > n:
        paths = paths[: n - 1] + paths[n - 1 :]

    output_str = paths[0]
    for path in paths[1 : n - 1]:
        output_str = f"{output_str}, {path}"

    if len(paths) > n:
        output_str = f"{output_str}, ..."
        output_str = f"{output_str}, {paths[-1]}"

    return output_str


def check_files(input_paths: Iterable[str], groups: Iterable[str]) -> None:
    for input_path in input_paths:
        with ur.open(f"{input_path}") as file:
            if "events" not in file:
                raise KeyError(f"Events tree not found in {input_path}")
            tree = file["events"]
            for group in groups:
                if group not in tree:
                    raise KeyError(f"Group '{group}' not found in {input_path}")


def merge(
    input_paths: str,
    output_path: str,
    groups: Iterable[str],
    pmt_idents: Iterable[str],
    parameters: Iterable[str] = None,
    pmt_info_idents: Iterable[str] = None,
) -> None:
    runs = []
    for path in input_paths:
        match = re.search(r"(?<=_r)(\d+)", str(path))
        if match:
            runs.append(int(match.group(0)))
        else:
            raise ValueError(f"Could not find run number in {path}")

    min_run = min(runs)
    max_run = max(runs)

    total_n_events = 0
    group = groups[0]
    for input_path in input_paths:
        with ur.open(f"{input_path}:events") as events_tree:
            total_n_events += events_tree.num_entries

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["min_run"] = min_run
        h5_file.attrs["max_run"] = max_run

        if parameters is not None:
            with ur.open(input_paths[0]) as input_file:
                for parameter in parameters:
                    h5_file.attrs[parameter] = input_file[parameter].value

        events_tree = ur.open(f"{input_paths[0]}:events")
        for group in groups:
            arrays = events_tree[group]
            for dset_name in arrays.keys():
                if dset_name in pmt_idents:
                    dset = arrays[dset_name].array(library="np")
                    dtype = h5py.vlen_dtype(dset[0].dtype)
                    shape = (total_n_events,)
                else:
                    dset = ak.to_numpy(arrays[dset_name].array(library="ak"))
                    dtype = dset.dtype
                    shape = (total_n_events, dset.shape[1])

                h5_file.create_dataset(f"{group}/{dset_name}", shape=shape, dtype=dtype)

        start_row = 0
        for input_path in tqdm.tqdm(input_paths):
            with ur.open(f"{input_path}:events") as events_tree:
                for group in groups:
                    arrays = events_tree[group]
                    for dset_name in arrays.keys():
                        if dset_name in pmt_idents:
                            dset = arrays[dset_name].array(library="np")
                        else:
                            dset = ak.to_numpy(arrays[dset_name].array(library="ak"))

                        block_size = dset.shape[0]
                        h5_file[f"{group}/{dset_name}"][start_row : start_row + block_size] = dset
            start_row += block_size

        input_path = input_paths[0]
        pmt_info_group = h5_file.create_group("pmt_info")
        with ur.open(f"{input_path}:pmt_info") as pmt_info_tree:
            dsets = pmt_info_tree.arrays(library="ak")
            for pmt_info_ident in pmt_info_idents:
                pmt_info_group.create_dataset(pmt_info_ident, data=ak.to_numpy(dsets[pmt_info_ident]))


def main(
    input_paths: List[str],
    train_output_path: str,
    train_test_split: float = 1.0,
    test_output_path: str | None = None,
    min_occupancy: float = 0.0,
    max_occupancy: float = 1.0,
    seed: int = 487391,
    condor_transfer_input_files: bool = False,
    condor_transfer_output_files: bool = False,
) -> None:
    groups = ["pmt", "event"]
    pmt_idents = ["hit_time", "qhs", "id"]
    parameters = ["inner_av_radius", "av_thickness"]

    pmt_info_idents = ["pos"]

    print("Checking files")
    check_files(input_paths, groups)
    print("Checking successful")

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
        groups=groups,
        pmt_idents=pmt_idents,
        parameters=parameters,
        pmt_info_idents=pmt_info_idents,
    )

    if test_output_path is not None:
        print("Merge test files")
        if train_test_split < 1.0 and test_output_path is not None:
            print("Merge test files")
            merge(
                input_paths=test_input_paths,
                output_path=test_output_path,
                groups=groups,
                pmt_idents=pmt_idents,
                parameters=parameters,
                pmt_info_idents=pmt_info_idents,
            )

    print("Transpose train files")
    transpose(train_output_path)

    print("Filter PMTs")

    filter_pmts(h5_path=train_output_path, min_occupancy=min_occupancy, max_occupancy=max_occupancy)

    print("Hashing files")
    paths = [train_output_path] 
    if test_output_path is not None:
        paths.append(test_output_path)
    for path in paths:
        train_hash = hash_file(path)
        with h5py.File(path, "r+") as h5_file:
            h5_file.attrs["hash"] = train_hash

    if condor_transfer_output_files:
        print("Transferring output files back")
        shutil.copy(train_output_path, final_train_output_path)
        if create_test_set:
            shutil.copy(test_output_path, final_test_output_path)

    print("Finished merging and proprocessing")


if __name__ == "__main__":
    CLI(main, as_positional=False)
