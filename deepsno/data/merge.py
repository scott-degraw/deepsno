#!/usr/bin/env -S python3 -u
import argparse
import re
from typing import List

import awkward as ak
import numpy as np
import tqdm
import uproot as ur

from utils import str_from_many_paths
from filter_pmts import filter_pmts




def main(
    input_paths: List[str],
    train_output_path: str,
    tree_2_branches: dict[str, List[str]],
    parameters: List[str],
    cut: str | None = None,
    train_test_split: float = 1.0,
    test_output_path: str | None = None,
    seed: int = 487391,
    step_size: str | int = "100 MB",
) -> None:
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

    if test_input_paths:
        print(f"Test input paths: {str_from_many_paths(test_input_paths)}")

    output_paths = [train_output_path]
    if test_output_path is not None:
        output_paths.append(test_output_path)

    input_paths = [train_input_paths]
    if test_input_paths:
        input_paths.append(test_input_paths)

    names = ["train"]
    if test_output_path is not None:
        names.append("test")

    input_path = input_paths[0][0]
    with ur.open(f"{input_path}:pmt_info") as tree_name:
        n_pmts = len(tree_name["pos"].array())

    pmt_counts = np.zeros(n_pmts, dtype=np.int64)

    pattern = re.compile(r"_r(\d+)")
    runs = set()
    for path in train_input_paths:
        match = pattern.search(path)
        if match:
            runs.add(int(match.group(1)))
        else:
            raise ValueError(f"Input path {path} does not match expected pattern for run number extraction.")

    metadata = {"cut": cut, "run_range": [min(runs), max(runs)]}
    if cut is None:
        metadata["cut"] = ""

    with ur.open(f"{input_path}") as direc:
        for parameter in parameters:
            if parameter in direc.keys(cycle=False):
                metadata[parameter] = direc[parameter].value

    for name, input_paths, output_path in zip(names, input_paths, output_paths):
        with ur.recreate(output_path) as output_file:
            tree_name = "event"
            branches = tree_2_branches[tree_name]
            tree_created = False
            for chunk in tqdm.tqdm(
                ur.iterate(
                    {f: tree_name for f in input_paths},
                    step_size=step_size,
                    library="ak",
                    expressions=branches,
                    cut=cut,
                ),
                total=len(input_paths),
                desc=f"Merging {name} tree {tree_name}",
            ):
                dict_chunk = {field: chunk[field] for field in chunk.fields}
                if tree_created:
                    output_file[tree_name].extend(dict_chunk)
                else:
                    output_file[tree_name] = dict_chunk
                    tree_created = True

                if "pmt_id" in branches:
                    pmt_counts += np.bincount(ak.to_numpy(ak.flatten(chunk["pmt_id"])), minlength=n_pmts)

            tree_name = "pmt_info"
            branches = tree_2_branches[tree_name]
            tree = ur.open({next(iter(input_paths)): tree_name})
            arrays = tree.arrays(branches, library="ak")
            output_file[tree_name] = {field: arrays[field] for field in arrays.fields}

            output_file["transpose"] = {"pmt_counts": pmt_counts}

            output_file["metadata"] = {key: np.array([value]) for key, value in metadata.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and preprocess DeepSNO data files.")
    parser.add_argument("input_paths", type=str, nargs="+", help="Paths to input files.")
    parser.add_argument("train_output_path", type=str, help="Path to the output training dataset.")
    parser.add_argument("--test_output_path", type=str, default=None, help="Path to the output test dataset.")
    parser.add_argument("--cut", type=str, help="Cut expression for filtering events.")
    parser.add_argument("--train_test_split", type=float, default=1.0, help="Fraction of data used for training.")
    parser.add_argument("--min_occupancy", type=float, default=0.0, help="Minimum occupancy for PMT to be selected.")
    parser.add_argument("--max_occupancy", type=float, default=1.0, help="Maximum occupancy for PMT to be selected.")
    parser.add_argument("--seed", type=int, default=487391, help="Random seed for reproducibility.")
    args = parser.parse_args()

    tree_2_branches = {
        "event": [
            "av_offset",
            "posx",
            "posy",
            "posz",
            "posr_av",
            "nhits",
            "energy",
            "pmt_hit_time",
            "pmt_qhs",
            "pmt_id",
        ],
        "pmt_info": ["pos"],
    }
    parameters = ["av_thickness", "inner_av_radius"]

    path = args.input_paths[0]
    with ur.open({path: "event"}) as tree:
        if "mc" in tree.keys():
            mc_branches = ["global_trigger_time", "times_of_flight"]
            mc_branches = [f"mc/{branch}" for branch in mc_branches]
            mc_branches += ["mcPosx", "mcPosy", "mcPosz", "mcke1", "mctime1"]
            tree_2_branches["event"] += mc_branches
    
    main(
        input_paths=args.input_paths,
        train_output_path=args.train_output_path,
        tree_2_branches=tree_2_branches,
        parameters=parameters,
        test_output_path=args.test_output_path,
        cut=args.cut,
        train_test_split=args.train_test_split,
        seed=args.seed,
    )

    filter_pmts(
        path=args.train_output_path,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy,
    )
