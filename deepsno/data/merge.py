#!/usr/bin/env -S python3 -u
import argparse
import re
from pathlib import Path
from typing import Iterable, List

import awkward as ak
import numpy as np
import tqdm
import uproot as ur
from filter_pmts import filter_pmts


def merge(
    input_paths: List[str],
    output_paths: str | Iterable,
    tree_2_branches: dict[str, List[str]],
    parameters: List[str],
    cut: str | None = None,
    splits: float | Iterable = 1.0,
    seed: int = 487391,
) -> None:
    if not isinstance(splits, Iterable):
        splits = []
    if not isinstance(output_paths, Iterable):
        output_paths = [output_paths]

    if len(splits) != len(output_paths):
        raise ValueError("Number of splits must match number of output paths.")

    negative_split = None
    for i in range(len(splits)):
        if splits[i] < 0:
            if negative_split is not None:
                raise ValueError("Only one split can be negative.")
            negative_split = i

    splits[negative_split] = 1.0 - sum(splits) + splits[negative_split]

    if not np.isclose(sum(splits), 1.0, atol=1e-4):
        raise ValueError("Splits must sum to 1.")

    generator = np.random.default_rng(seed)
    input_path_indices = generator.choice(np.arange(len(input_paths)), size=len(input_paths), replace=False)

    split_indices = [int(split * len(input_path_indices)) for split in np.cumsum(splits)[:-1]]
    path_indices = np.split(input_path_indices, split_indices)
    all_input_paths = [[input_paths[i] for i in path_indices_row] for path_indices_row in path_indices]

    input_path = all_input_paths[0][0]
    with ur.open(f"{input_path}:pmt_info") as tree_name:
        n_pmts = len(tree_name["pos"].array())

    pmt_counts = np.zeros(n_pmts, dtype=np.int64)

    pattern = re.compile(r"_r(\d+)")
    runs = set()
    for row in all_input_paths:
        for path in row:
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

    for input_paths_row, output_path in zip(all_input_paths, output_paths):
        total_size = 0
        for path in input_paths_row:
            total_size += Path(path).stat().st_size

        with ur.recreate(output_path, compression=ur.ZLIB(0)) as output_file:
            tree_name = "event"
            branches = tree_2_branches[tree_name]
            tree_created = False

            with tqdm.tqdm(total=len(input_paths_row), desc=f"{Path(output_path).name}: file number") as pbar:
                curr_n_events = 0
                curr_file_sizes = 0
                for input_path in input_paths_row:
                    pbar.update(1)
                    with ur.open({input_path: tree_name}) as tree:
                        branches = [branch for branch in branches if branch in tree]
                        chunk = tree.arrays(branches, library="ak", cut=cut)
                        curr_n_events += len(chunk)
                        curr_file_sizes += Path(input_path).stat().st_size

                        pred_n_events = curr_n_events / curr_file_sizes * total_size
                        pbar.set_postfix({"Predicted total events": f"{round(pred_n_events):,}"})

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
                tree = ur.open({next(iter(input_paths_row)): tree_name})
                arrays = tree.arrays(branches, library="ak")
                output_file[tree_name] = {field: arrays[field] for field in arrays.fields}

                output_file["transpose"] = {"pmt_counts": pmt_counts}

                output_file["metadata"] = {key: np.array([value]) for key, value in metadata.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and preprocess DeepSNO data files.")
    parser.add_argument("--input_paths", type=str, nargs="+", help="Paths to input files.", required=True)
    parser.add_argument("--output_paths", type=str, nargs="+", help="Paths to output files.", required=True)
    parser.add_argument("--splits", type=float, nargs="+", help="Fractions for splitting the data.", required=True)
    parser.add_argument("--cut", type=str, help="Cut expression for filtering events.")
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

    merge(
        input_paths=args.input_paths,
        output_paths=args.output_paths,
        splits=args.splits,
        tree_2_branches=tree_2_branches,
        parameters=parameters,
        cut=args.cut,
        seed=args.seed,
    )

    for output_path in args.output_paths:
        filter_pmts(
            path=output_path,
            min_occupancy=args.min_occupancy,
            max_occupancy=args.max_occupancy,
        )
