#!/usr/bin/env -S python3 -u
import itertools
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    if block_size > n_events:
        block_size = n_events

    max_n_blocks = n_events // block_size

    if n_blocks is None or n_blocks > max_n_blocks:
        # For ease of good numerical calculation of means I only consider evenly sized blocks
        n_blocks = max_n_blocks

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


def preprocess_hit_time(dset: np.ndarray, pmt_id_dset: np.ndarray, min_hit_time: float, max_hit_time: float):
    selector = (dset > max_hit_time) | (dset < min_hit_time)
    dset[selector] = 0.0
    pmt_id_dset[selector] = 0


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


def check_files(input_paths: List[str], dataset_identifiers: List[str], pmt_info_identifiers: List[str]):
    for input_path in input_paths:
        with h5py.File(input_path) as h5_file:
            for dataset_identifier in dataset_identifiers:
                if dataset_identifier not in h5_file:
                    raise KeyError(f"Dataset identifier '{dataset_identifier}' not found in {input_path}")
            for pmt_info_identifier in pmt_info_identifiers:
                if pmt_info_identifier not in h5_file:
                    raise KeyError(f"PMT information identifier '{pmt_info_identifier}' not found in {input_path}")


@dataclass
class GroupAndAttr:
    group: str
    attr: str
    dest_group: str | None = None


def merge_and_preprocess(
    input_paths: str,
    output_path: str,
    dset_idents: Tuple[str],
    pmt_id_ident: str,
    pmt_info_idents: Tuple[str],
    min_hit_time: float,
    max_hit_time: float,
    group_and_attrs: Tuple[GroupAndAttr],
    per_event_group_and_attrs: Tuple[GroupAndAttr],
):
    all_input_h5 = [h5py.File(path) for path in input_paths]
    total_n_events = 0
    for input_h5 in all_input_h5:
        total_n_events += input_h5.attrs["number_of_events"]
    try:
        with h5py.File(output_path, "w", libver="latest") as merged_h5:
            for dset_ident in itertools.chain(dset_idents, [pmt_id_ident]):
                dset_dtype = all_input_h5[0][dset_ident].dtype
                dset_dims = []
                for input_h5 in all_input_h5:
                    dset_dims.append(input_h5[dset_ident].shape)

                dset_dims = np.array(dset_dims)

                if dset_dims.shape[1] > 1:
                    assert np.all(dset_dims[0, 1:] == dset_dims[:, 1:]), (
                        f"Dataset '{dset_ident}' dimensions are not compatible"
                    )

                n_events = dset_dims[:, 0].sum()
                assert n_events == total_n_events, f"Number of events in datasets do not match for {dset_ident}"
                other_dims = dset_dims[0, 1:]
                merged_h5.create_dataset(dset_ident, shape=(total_n_events, *other_dims), dtype=dset_dtype)

            input_h5 = all_input_h5[0]
            for group_and_attr in group_and_attrs:
                attr = input_h5[group_and_attr.group].attrs[group_and_attr.attr]
                merged_h5.attrs[group_and_attr.attr] = attr
            for group_and_attr in per_event_group_and_attrs:
                attr = input_h5[group_and_attr.group].attrs[group_and_attr.attr]
                merged_h5.create_dataset(
                    group_and_attr.dest_group, shape=(total_n_events, *attr.shape), dtype=attr.dtype
                )

            start_row = 0
            for input_file_num, input_h5 in enumerate(all_input_h5):
                print(f"Merging file {input_file_num + 1}/{len(input_paths)}")
                n_events = input_h5.attrs["number_of_events"]
                for group_and_attr in group_and_attrs:
                    attr = input_h5[group_and_attr.group].attrs[group_and_attr.attr]
                    assert merged_h5.attrs[group_and_attr.attr] == attr, (
                        f"Attribute '{group_and_attr.attr}' does not match between files"
                    )
                for group_and_attr in per_event_group_and_attrs:
                    attr = input_h5[group_and_attr.group].attrs[group_and_attr.attr]
                    attr = np.broadcast_to(attr, (n_events, *attr.shape))
                    merged_h5[group_and_attr.dest_group][start_row : start_row + n_events] = attr

                for dset_ident in dset_idents:
                    dset = input_h5[dset_ident][:]
                    pmt_id_dset = input_h5[pmt_id_ident][:]

                    if dset_ident == "cal_pmt_events/hit_times":
                        preprocess_hit_time(
                            dset, pmt_id_dset=pmt_id_dset, min_hit_time=min_hit_time, max_hit_time=max_hit_time
                        )

                    merged_h5[dset_ident][start_row : start_row + dset.shape[0]] = dset
                    merged_h5[pmt_id_ident][start_row : start_row + dset.shape[0]] = pmt_id_dset

                start_row += dset.shape[0]

            for pmt_info_ident in pmt_info_idents:
                input_h5 = all_input_h5[0]
                pmt_info_array = input_h5[pmt_info_ident][:]
                base_fname = input_h5.filename
                for input_h5 in all_input_h5:
                    if np.all(pmt_info_array != input_h5[pmt_info_ident][:]):
                        raise ValueError(
                            (
                                f"{pmt_info_ident} in {input_h5.filename} does not match corresponding entry in {base_fname}"
                            )
                        )

                merged_h5.create_dataset(pmt_info_ident, data=pmt_info_array)

    finally:
        for h5_file in all_input_h5:
            h5_file.close()


def main(
    input_paths: List[str],
    train_output_path: str,
    test_output_path: str,
    train_test_split: float,
    min_hit_time: float,
    max_hit_time: float,
    positions: List[str] = ["x", "y", "z"],
    block_size: int = 100_000,
    n_blocks: int = None,
    seed: int = 487391,
    condor_transfer_input_files: bool = False,
    condor_transfer_output_files: bool = False,
) -> None:
    dataset_identifiers = ["cal_pmt_events/hit_times", "cal_pmt_events/QHS", "mc_truth/kinetic_energy"]
    dataset_identifiers += [f"mc_truth/position/{c}" for c in positions]
    dataset_identifiers += ["mc_truth/global_trigger_time", "cal_pmt_events/times_of_flight"]
    dataset_identifiers += ["cal_pmt_events/mc_hit_times"]
    pmt_id_ident = "cal_pmt_events/ids"

    group_and_attrs = [
        GroupAndAttr("/", "inner_av_radius"),
        GroupAndAttr("/", "av_thickness"),
        GroupAndAttr("/", "is_mc"),
    ]
    per_event_group_and_attrs = [
        GroupAndAttr("/", "av_offset", "cal_pmt_events/av_offset"),
    ]

    pmt_info_identifiers = [f"pmt_info/position/{c}" for c in positions]

    print("Checking files")
    check_files(input_paths, dataset_identifiers, pmt_info_identifiers)
    print("Checking successful")

    if condor_transfer_input_files:
        condor_scratch_dir = Path(os.environ["_CONDOR_SCRATCH_DIR"])
        print("Copying files to Condor scratch disk")
        for file_num, path in enumerate(input_paths):
            print(f"File {file_num + 1}/{len(input_paths)}")
            shutil.copy(path, condor_scratch_dir)

        final_train_output_path = train_output_path
        final_test_output_path = test_output_path

        input_paths = [condor_scratch_dir / Path(path).name for path in input_paths]

    if condor_transfer_output_files:
        train_output_path = condor_scratch_dir / Path(train_output_path).name
        test_output_path = condor_scratch_dir / Path(test_output_path).name

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

    merge_and_preprocess(
        input_paths=train_input_paths,
        output_path=train_output_path,
        dset_idents=dataset_identifiers,
        pmt_id_ident=pmt_id_ident,
        pmt_info_idents=pmt_info_identifiers,
        min_hit_time=min_hit_time,
        max_hit_time=max_hit_time,
        group_and_attrs=group_and_attrs,
        per_event_group_and_attrs=per_event_group_and_attrs,
    )

    # Add in the mean and root mean square deviation normalization

    with h5py.File(train_output_path, "r+") as train_h5:
        print("Finding hit time norms")
        find_norms(
            train_h5["cal_pmt_events/hit_times"],
            block_size=block_size,
            n_blocks=n_blocks,
            pmt_id_dataset=train_h5["cal_pmt_events/ids"],
        )
        for c in positions:
            print(f"Finding {c} position norms")
            find_norms(train_h5[f"mc_truth/position/{c}"], block_size=block_size, n_blocks=n_blocks)
        print("Finding event time norms")
        find_norms(train_h5["mc_truth/global_trigger_time"], block_size=block_size, n_blocks=n_blocks)

    print("Merge test files")
    merge_and_preprocess(
        input_paths=test_input_paths,
        output_path=test_output_path,
        dset_idents=dataset_identifiers,
        pmt_id_ident=pmt_id_ident,
        pmt_info_idents=pmt_info_identifiers,
        min_hit_time=min_hit_time,
        max_hit_time=max_hit_time,
        group_and_attrs=group_and_attrs,
        per_event_group_and_attrs=per_event_group_and_attrs,
    )

    if condor_transfer_output_files:
        print("Transferring output files back")
        shutil.copy(train_output_path, final_train_output_path)
        shutil.copy(test_output_path, final_test_output_path)

    print("Finished merging and proprocessing")


if __name__ == "__main__":
    CLI(main, as_positional=False)
