from pathlib import Path

import h5py
import numpy as np

block_size = 1_000_000

train_output_path = Path("/data/snoplus3/degraw/train_dset.h5")
test_output_path = Path("/data/snoplus3/degraw/test_dset.h5")

input_paths = list(Path("/data/snoplus3/degraw/extraction_test").glob("*.h5"))

train_test_split = 0.8

dataset_identifiers = ["cal_pmt_events/hit_times", "cal_pmt_events/masks"]
dataset_identifiers += [f"mc_truth/position/{c}" for c in ["x", "y", "z"]]

pmt_info_identifiers = ["pmt_info/inward_pmt_ids" "pmt_info/pmt_id_2_index"]
pmt_info_identifiers = [f"pmt_info/position/{c}" for c in ["x", "y", "z"]]


def find_norms(dataset: h5py.Dataset, block_size: int, masks: h5py.Dataset):
    n_events = dataset.shape[0]

    n_blocks = (n_events - 1) // block_size + 1

    data_sum = 0

    n_hits: int = 0
    start_row: int = 0

    for _ in range(n_blocks):
        block_slice = slice(start_row, min(start_row + block_size, n_events - 1))
        data_block = dataset[block_slice]
        mask_block = masks[block_slice]
        data_sum += data_block.sum()
        n_hits += mask_block.size - mask_block.sum()
        start_row += block_size

    mean = data_sum / n_hits

    residual_sum = 0

    start_row = 0
    for _ in range(n_blocks):
        data_block = dataset[start_row : min(start_row + block_size, n_events - 1)]
        residual_sum += np.sum(np.square(data_block - mean))
        start_row += block_size

    # Root mean square deviation
    root_mean_square_deviation = np.sqrt(residual_sum / n_hits)

    dataset.attrs["mean"] = mean
    dataset.attrs["root_mean_square_deviation"] = root_mean_square_deviation


def merge_h5(input_paths: Path, output_path: Path, dataset_identifiers: list[str], pmt_info_identifiers: list[str]):
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
            for input_path in input_paths:
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
                                f"{pmt_info_ident} in {test_h5_file.filename} does not match corresponding entry in {base_fname}"
                            )

                merged_h5.create_dataset_like(pmt_info_ident, pmt_info_item)


if __name__ == "__main__":
    input_path_indices = np.random.choice(np.arange(len(input_paths)), size=len(input_paths), replace=False)

    split_index = np.floor(train_test_split * len(input_path_indices)).astype(np.int64)

    train_path_indices = input_path_indices[:split_index]
    test_path_indices = input_path_indices[split_index:]

    train_input_paths = [input_paths[i] for i in train_path_indices]
    test_input_paths = [input_paths[i] for i in test_path_indices]

    if len(test_input_paths) == 0:
        raise RuntimeError(f"Not enough files. No test dataset for {train_test_split:.3g} train-test split.")

    merge_h5(
        input_paths=train_input_paths,
        output_path=train_output_path,
        dataset_identifiers=dataset_identifiers,
        pmt_info_identifiers=pmt_info_identifiers,
    )

    merge_h5(
        input_paths=test_input_paths,
        output_path=test_output_path,
        dataset_identifiers=dataset_identifiers,
        pmt_info_identifiers=pmt_info_identifiers,
    )

    # Add in the mean and root mean square deviation normalization

    with h5py.File(train_output_path, "r+") as train_h5:
        find_norms(train_h5["cal_pmt_events/hit_times"], block_size=block_size, masks=train_h5["cal_pmt_events/masks"])
