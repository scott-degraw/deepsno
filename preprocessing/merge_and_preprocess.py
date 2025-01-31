#!/usr/bin/env -S python3 -u
import os
import shutil
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from jsonargparse import CLI


def str_from_many_paths(paths: tuple[str], n=3) -> str:
    if len(paths) > n:
        paths = paths[: n - 1] + paths[n - 1 :]

    output_str = paths[0]
    for path in paths[1 : n - 1]:
        output_str = f"{output_str}, {path}"

    if len(paths) > n:
        output_str = f"{output_str}, ..."

    output_str = f"{output_str}, {paths[-1]}"
    return output_str


def find_norms(
    dataset: h5py.Dataset, block_size: int, pmt_id_dataset: Optional[h5py.Dataset] = None, n_blocks: int = None
) -> None:
    # This function relies on masked values having a value of 0
    n_events = dataset.shape[0]
    max_n_blocks = n_events // block_size

    if max_n_blocks == 0:
        raise ValueError(
            (
                f"The value of 'block_size' ({block_size}) is too large. "
                f"It is smaller than length of data dataset: {n_events}."
            )
        )
    if n_blocks is None:
        # For ease of good numerical calculation of means I only consider evenly sized blocks
        n_blocks = max_n_blocks
    if n_blocks > max_n_blocks:
        raise ValueError(
            f"The value of 'n_blocks' is {n_blocks}. The maximum value of 'n_blocks' for this dataset is {max_n_blocks}"
        )

    block_means = np.full(n_blocks, dtype=np.float64, fill_value=np.nan)

    start_row: int = 0
    print("Finding mean")
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        block_slice = slice(start_row, min(start_row + block_size, n_events))
        data_block = dataset[block_slice]
        if pmt_id_dataset is not None:
            data_block = data_block[pmt_id_dataset[block_slice] != 0]
        block_means[block_num] = np.mean(data_block)
        start_row += block_size

    mean = np.mean(block_means)

    block_variances = np.full(n_blocks, dtype=np.float64, fill_value=np.nan)

    start_row = 0
    print("Finding root mean square deviation")
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        block_slice = slice(start_row, min(start_row + block_size, n_events))
        data_block = dataset[block_slice]
        if pmt_id_dataset is not None:
            data_block = data_block[pmt_id_dataset[block_slice] != 0]
        block_variances[block_num] = np.mean(np.square(data_block - mean))
        start_row += block_size

    rmsd = np.sqrt(np.mean(block_variances))

    dataset.attrs["mean"] = mean
    dataset.attrs["root_mean_square_deviation"] = rmsd


def preprocess_hit_time(output_path: str, min_hit_time: float, max_hit_time: float, block_size: int):
    if not isinstance(output_path, h5py.Group):
        output_path = h5py.File(output_path, "r+")

    hit_time_dset = output_path["cal_pmt_events/hit_times"]
    id_dset = output_path["cal_pmt_events/ids"]

    n_events = hit_time_dset.shape[0]
    n_blocks = (n_events - 1) // block_size + 1
    start_row = 0
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        block_slice = slice(start_row, min(start_row + block_size, n_events))
        hit_time_block = hit_time_dset[block_slice]
        id_block = id_dset[block_slice]

        selector = (hit_time_block > max_hit_time) | (hit_time_block < min_hit_time)
        hit_time_block[selector] = 0.0
        id_block[selector] = 0

        hit_time_dset[block_slice] = hit_time_block
        id_dset[block_slice] = id_block
        start_row += block_size


def merge_h5(
    input_paths: List[str],
    output_path: str,
    dataset_identifiers: List[str],
    pmt_info_identifiers: List[str],
) -> None:
    print(f"Merging {str_from_many_paths(input_paths)} to {output_path}.")
    with h5py.File(output_path, "w", libver="latest") as merged_h5:
        for dataset_identifier in dataset_identifiers:
            dataset_dims = []

            with h5py.File(next(iter(input_paths))) as h5_file:
                dataset_dtype = h5_file[dataset_identifier].dtype

            for input_path in input_paths:
                with h5py.File(input_path) as h5_file:
                    dataset_dims.append(h5_file[dataset_identifier].shape)

            dataset_dims = np.array(dataset_dims)
            if dataset_dims.shape[1] > 1:
                assert np.all(dataset_dims[0, 1:] == dataset_dims[:, 1:]), "Dataset dimensions are not compatible"

            n_events = dataset_dims[:, 0].sum()
            other_dims = dataset_dims[0, 1:]

            merged_dataset = merged_h5.create_dataset(
                dataset_identifier, shape=(n_events, *other_dims), dtype=dataset_dtype
            )

            start_row_i: int = 0
            for input_file_num, input_path in enumerate(input_paths):
                print(f"Merging {dataset_identifier} for file {input_file_num + 1}/{len(input_paths)}")
                with h5py.File(input_path) as input_h5:
                    dataset = input_h5[dataset_identifier]
                    merged_dataset[start_row_i : start_row_i + dataset.shape[0]] = dataset[:]
                start_row_i += dataset.shape[0]

        # Checking if all the pmt info is the same across the datasets
        for pmt_info_ident in pmt_info_identifiers:
            with h5py.File(next(iter(input_paths))) as h5_file:
                pmt_info_item = h5_file[pmt_info_ident]
                pmt_info_array = pmt_info_item[:]
                base_fname = h5_file.filename

                for input_path in input_paths:
                    with h5py.File(input_path) as test_h5_file:
                        if np.all(pmt_info_array != test_h5_file[pmt_info_ident][:]):
                            raise ValueError(
                                (
                                    f"{pmt_info_ident} in {test_h5_file.filename}"
                                    f" does not match corresponding entry in {base_fname}"
                                )
                            )

                merged_h5.copy(source=pmt_info_item, dest=merged_h5, name=pmt_info_ident)


def merge_and_norm(
    input_paths: List[str],
    train_output_path: str,
    test_output_path: str,
    train_test_split: float,
    min_hit_time: float,
    max_hit_time: float,
    positions: List[str] = ["x", "y", "z"],
    block_size: int = 100_000,
    seed: int = 487391,
    condor_transfer_files: bool = False,
) -> None:
    if condor_transfer_files:
        condor_scratch_dir = Path(os.environ["_CONDOR_SCRATCH_DIR"])
        print("Copying files to Condor scratch disk")
        for file_num, path in enumerate(input_paths):
            print(f"File {file_num + 1}/{len(input_paths)}")
            shutil.copy(path, condor_scratch_dir)

        final_train_output_path = train_output_path
        final_test_output_path = test_output_path

        input_paths = [condor_scratch_dir / Path(path).name for path in input_paths]
        train_output_path = condor_scratch_dir / Path(train_output_path).name
        test_output_path = condor_scratch_dir / Path(test_output_path).name

    dataset_identifiers = ["cal_pmt_events/hit_times", "cal_pmt_events/ids", "mc_truth/kinetic_energy"]
    dataset_identifiers += [f"mc_truth/position/{c}" for c in positions]

    pmt_info_identifiers = [f"pmt_info/position/{c}" for c in positions]

    generator = np.random.default_rng(seed)
    input_path_indices = generator.choice(np.arange(len(input_paths)), size=len(input_paths), replace=False)

    split_index = np.floor(train_test_split * len(input_path_indices)).astype(np.int64)

    train_path_indices = input_path_indices[:split_index]
    test_path_indices = input_path_indices[split_index:]

    train_input_paths = [input_paths[i] for i in train_path_indices]
    test_input_paths = [input_paths[i] for i in test_path_indices]

    if len(test_input_paths) == 0:
        raise RuntimeError(f"Not enough files. No test dataset for {train_test_split:.3g} train-test split.")

    print("Merge train files")
    merge_h5(
        input_paths=train_input_paths,
        output_path=train_output_path,
        dataset_identifiers=dataset_identifiers,
        pmt_info_identifiers=pmt_info_identifiers,
    )
    print("Preprocess train dataset")
    preprocess_hit_time(train_output_path, min_hit_time=min_hit_time, max_hit_time=max_hit_time, block_size=block_size)

    # Add in the mean and root mean square deviation normalization

    with h5py.File(train_output_path, "r+") as train_h5:
        print("Finding hit time norms")
        find_norms(
            train_h5["cal_pmt_events/hit_times"], block_size=block_size, pmt_id_dataset=train_h5["cal_pmt_events/ids"]
        )
        for c in positions:
            print(f"Finding {c} position norms")
            find_norms(train_h5[f"mc_truth/position/{c}"], block_size=block_size)

    print("Merge test files")
    merge_h5(
        input_paths=test_input_paths,
        output_path=test_output_path,
        dataset_identifiers=dataset_identifiers,
        pmt_info_identifiers=pmt_info_identifiers,
    )

    print("Preprocess test dataset")
    preprocess_hit_time(test_output_path, min_hit_time=min_hit_time, max_hit_time=max_hit_time, block_size=block_size)

    if condor_transfer_files:
        print("Transferring output files back")
        shutil.copy(train_output_path, final_train_output_path)
        shutil.copy(test_output_path, final_test_output_path)

    print("Finished merging and proprocessing")


if __name__ == "__main__":
    CLI(merge_and_norm, as_positional=False)
