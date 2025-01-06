#!/usr/bin/env -S python3 -u
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


def find_valid_pmts(dataset: h5py.Dataset, block_size: int, n_pmts: int) -> None:
    n_events = dataset.shape[0]

    n_blocks = (n_events - 1) // block_size + 1
    start_row: int = 0

    id_dtype = dataset.dtype

    valid_ids = np.array([], dtype=id_dtype)
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        id_block = dataset[start_row : min(start_row + block_size, n_events)]
        block_unique_ids = np.unique_values(id_block)
        valid_ids = np.unique_values(np.concatenate([valid_ids, block_unique_ids]))

    valid_ids = valid_ids[valid_ids != 0]
    dataset.attrs["valid_pmt_ids"] = np.sort(valid_ids)
    all_ids = np.arange(0, n_pmts, dtype=id_dtype)
    dataset.attrs["invalid_pmt_ids"] = np.sort(np.setdiff1d(all_ids, valid_ids))


def find_norms(dataset: h5py.Dataset, block_size: int, pmt_id_dataset: Optional[h5py.Dataset] = None) -> None:
    # This function counts on masked values having a value of 0
    n_events = dataset.shape[0]

    n_blocks = (n_events - 1) // block_size + 1

    data_sum = 0

    n_values: int = 0
    start_row: int = 0

    print("Finding mean")
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        block_slice = slice(start_row, min(start_row + block_size, n_events))
        data_block = dataset[block_slice]
        if pmt_id_dataset is not None:
            mask_block = pmt_id_dataset[block_slice] == 0
            data_block = data_block[~mask_block]
            n_values += mask_block.size - mask_block.sum()
        else:
            n_values += data_block.size
        data_sum += data_block.sum()
        start_row += block_size

    mean = data_sum / n_values

    residual_sum = 0

    start_row = 0
    print("Finding root mean square deviation")
    for block_num in range(n_blocks):
        print(f"Block {block_num + 1}/{n_blocks}")
        block_slice = slice(start_row, min(start_row + block_size, n_events))
        data_block = dataset[block_slice]
        if pmt_id_dataset is not None:
            mask_block = pmt_id_dataset[block_slice] == 0
            data_block = data_block[~mask_block]
        residual_sum += np.sum(np.square(data_block - mean))
        start_row += block_size

    root_mean_square_deviation = np.sqrt(residual_sum / n_values)

    dataset.attrs["mean"] = mean
    dataset.attrs["root_mean_square_deviation"] = root_mean_square_deviation


def merge_h5(
    input_paths: List[str], output_path: str, dataset_identifiers: List[str], pmt_info_identifiers: List[str]
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
    positions: List[str] = ["x", "y", "z"],
    block_size: int = 100_000_000,
    seed: int = 487391,
) -> None:
    dataset_identifiers = ["cal_pmt_events/hit_times", "cal_pmt_events/ids", "cal_pmt_events/qhs"]
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

    # Add in the mean and root mean square deviation normalization

    with h5py.File(train_output_path, "r+") as train_h5:
        print("Finding hit time norms")
        find_norms(
            train_h5["cal_pmt_events/hit_times"], block_size=block_size, pmt_id_dataset=train_h5["cal_pmt_events/ids"]
        )
        for c in positions:
            print(f"Finding {c} position norms")
            find_norms(train_h5[f"mc_truth/position/{c}"], block_size=block_size)
        print("Finding valid PMT IDs")
        n_pmts = len(next(iter(train_h5["pmt_info/position"].values())))
        find_valid_pmts(dataset=train_h5["cal_pmt_events/ids"], block_size=block_size, n_pmts=n_pmts)

    print("Merge test files")
    merge_h5(
        input_paths=test_input_paths,
        output_path=test_output_path,
        dataset_identifiers=dataset_identifiers,
        pmt_info_identifiers=pmt_info_identifiers,
    )

    print("Finished merging and proprocessing")


if __name__ == "__main__":
    CLI(merge_and_norm, as_positional=False)
