import glob
import json
import os
import shutil
from pathlib import Path
from typing import Hashable, Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
import uproot as ur
from torch.utils.data import IterableDataset

from deepsno.data.multihit import UprootMultiFileDataset
from deepsno.models.hit_time_autoencoder import exp_time_walk


def good_pmt_status(status: np.ndarray, status_mask: int) -> np.ndarray:
    return ~(status_mask & status).astype(bool)


class ChunkedUprootDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        expressions: Iterable[str] = set(),
        condor_scratch: bool = False,
        batch_size: int | None = None,
        chunk_size: int = 10000,
        buffer_size: int = 10000,
        cut: str | None = None,
        max_num_entries: int | None = None,
    ):
        self.path = Path(path)
        self.expressions = set(expressions)
        self.cut = cut
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.event_tree = None
        self.buffer_size = buffer_size
        self.generator = None
        self.worker_id = 0
        self.num_workers = 1
        self.max_num_entries = max_num_entries

        self.n_events = 0

        if condor_scratch and "_CONDOR_SCRATCH_DIR" in os.environ:
            condor_scratch_path = Path(os.environ["_CONDOR_SCRATCH_DIR"]) / Path(path).name
            if not condor_scratch_path.exists():
                print(f"Copying dataset at {str(path)} to condor scratch directory...")
                shutil.copy(path, condor_scratch_path)
            self.path = condor_scratch_path
        else:
            self.path = path

        for chunk in ur.iterate({path: "event"}, expressions=["pmt_id"], cut=cut):
            self.n_events += len(chunk)

    def __len__(self) -> int:
        return self.n_events

    def __iter__(self) -> Iterator[dict[Hashable, np.ndarray]]:
        if not self.expressions:
            raise ValueError("No expressions specified for ChunkedUprootDataset")

        if self.event_tree is None:
            self.event_tree = ur.open({self.path: "event"})
        if self.generator is None:
            self.generator = np.random.default_rng()

        tree_size = self.event_tree.num_entries

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        global_worker_id = rank * self.num_workers + self.worker_id
        total_workers = world_size * self.num_workers

        worker_block_size = tree_size // total_workers
        worker_begin_index = global_worker_id * worker_block_size
        if global_worker_id == total_workers - 1:
            worker_end_index = tree_size
            worker_block_size = worker_end_index - worker_begin_index
        else:
            worker_end_index = (global_worker_id + 1) * worker_block_size

        self.buffer_size = min(self.buffer_size, worker_block_size)
        buffer = {field: np.empty(self.buffer_size, dtype=np.object_) for field in self.expressions}
        buffer_i_init = 0

        chunk_splits = np.arange(worker_begin_index, worker_end_index, self.chunk_size)
        chunk_splits = np.append(chunk_splits, worker_end_index)

        total_events_retrieved = 0
        total_events_yielded = 0

        for chunk_split_i in self.generator.permutation(len(chunk_splits) - 1):
            entry_start = chunk_splits[chunk_split_i]
            entry_stop = chunk_splits[chunk_split_i + 1]
            chunk = self.event_tree.arrays(
                expressions=self.expressions,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="np",
                cut=self.cut,
            )

            # If a cut is applied, we may have fewer events than expected
            chunk_len = len(next(iter(chunk.values())))
            if chunk_len == 0:
                raise ValueError("Did not read any data")

            total_events_retrieved += chunk_len
            sample_indices = self.generator.permutation(chunk_len)

            for i in sample_indices:
                if buffer_i_init < self.buffer_size:
                    for field in chunk:
                        buffer[field][buffer_i_init] = chunk[field][i]
                    buffer_i_init += 1
                    continue

                buffer_index = self.generator.integers(self.buffer_size)
                event = {field: value[buffer_index] for field, value in buffer.items()}

                total_events_yielded += 1
                yield event

                for field in chunk:
                    buffer[field][buffer_index] = chunk[field][i]

        # If the buffer wasn't filled fully treat the rest of the buffer as padding
        self.buffer_size = buffer_i_init

        # Eat through remainder of buffer
        buffer_indices = self.generator.permutation(self.buffer_size)
        for buffer_index in buffer_indices:
            event = {field: value[buffer_index] for field, value in buffer.items()}
            total_events_yielded += 1
            yield event

        if total_events_yielded != total_events_retrieved:
            raise RuntimeError(
                (
                    "Mismatch between yielded number of events and number of retrieved events\n"
                    f"Yielded {total_events_yielded} but retrieved {total_events_retrieved}"
                )
            )


class PositionRecoDataset(UprootMultiFileDataset):
    def __init__(
        self,
        pmt_valid_file: str,
        context_len: int,
        truth_expressions: list = [],
        status_mask: int = 0xFFFFFFFF,
        seed=74819,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.context_len = context_len
        self.status_mask = status_mask
        self.seed = seed
        self.truth_expressions = truth_expressions
        self.expressions.update(truth_expressions)
        self.expressions.update(["pmt_ids", "pmt_hit_times", "pmt_qhs"])

        self.pmt_valid = np.loadtxt(pmt_valid_file, dtype=bool)
        print(f"Using {np.sum(self.pmt_valid)} / {len(self.pmt_valid)} PMTs in dataset")

    def __iter__(self) -> Iterator[dict[Hashable, torch.Tensor]]:
        for entry, _ in super().__iter__():
            pmt_ids = entry["pmt_ids"].to_numpy()
            pmt_ids *= self.pmt_valid[pmt_ids]
            hit_times = self.pmt_valid[pmt_ids] * entry["pmt_hit_times"].to_numpy()
            qhs = self.pmt_valid[pmt_ids] * entry["pmt_qhs"].to_numpy()

            non_zero_pmt_indices = np.nonzero(pmt_ids)[0]
            if len(non_zero_pmt_indices) == 0:
                raise ValueError("Input has no valid PMTs")

            if len(non_zero_pmt_indices) > self.context_len:
                pmt_indices = np.sort(self.generator.choice(non_zero_pmt_indices, size=self.context_len, replace=False))
                pmt_ids = pmt_ids[pmt_indices]
                hit_times = hit_times[pmt_indices]
                qhs = qhs[pmt_indices]
            else:
                pmt_ids = pmt_ids[non_zero_pmt_indices]
                hit_times = hit_times[non_zero_pmt_indices]
                pmt_ids = np.pad(pmt_ids, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
                hit_times = np.pad(hit_times, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))

                qhs = qhs[non_zero_pmt_indices]
                qhs = np.pad(qhs, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))

            pmt_ids = torch.from_numpy(pmt_ids).long()
            hit_times = torch.from_numpy(hit_times)
            qhs = torch.from_numpy(qhs)

            inputs = {"hit_times": hit_times, "pmt_ids": pmt_ids, "qhs": qhs}

            truth = {field: torch.from_numpy(entry[field].to_numpy()) for field in self.truth_expressions}

            yield inputs, truth


class HitTimeAEDataset(PositionRecoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expressions.add("av_offset")

        if isinstance(self.file_paths, str):
            file_paths = glob.glob(self.file_paths)
        else:
            file_paths = self.file_paths

        with ur.open({next(iter(file_paths)): "pmt_info"}) as pmt_info:
            pos = pmt_info["pos"].array(library="np")
            self.n_pmts = pos.shape[0]
            self._pmt_positions = torch.from_numpy(pos)

    def __iter__(self) -> Iterator[dict[Hashable, torch.Tensor]]:
        for inputs, truth in super().__iter__():
            inputs["av_offset"] = truth["av_offset"]
            inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

            inputs["uncal_hit_times"] = inputs.pop("hit_times")

            yield inputs, truth


class TimeWalkDataset(HitTimeAEDataset):
    def __init__(
        self,
        ratdb_path: str | Path,
        model_save_dir: str | Path,
        noise: float = 1.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        model_save_dir = Path(model_save_dir)

        self.noise = noise

        with open(ratdb_path) as f:
            time_walk_params = json.load(f)

        bool_time_walk_status = good_pmt_status(
            np.array(time_walk_params["PCATW_status"]), status_mask=self.status_mask
        )

        param_names = ["time_scale", "qhs_scale", "gradient", "intercept"]
        for name in param_names:
            time_walk_params[name] = np.array(time_walk_params[name], dtype=np.float32)

        time_walk_params["max_qhs_scale"] = np.inf
        for param_name in param_names:
            bool_time_walk_status &= time_walk_params[f"max_{param_name}"] >= time_walk_params[param_name]
            bool_time_walk_status &= time_walk_params[f"min_{param_name}"] < time_walk_params[param_name]

        # Ids that are valid in dataset but are not valid in given time walk table
        bool_ids_to_fill = self.pmt_statuses & (~bool_time_walk_status)
        if np.sum(bool_time_walk_status) == np.sum(self.pmt_statuses):
            raise RuntimeError("All PMTs are marked bad in time walk table")

        ids_to_fill = np.nonzero(bool_ids_to_fill)[0]

        crate_size = 512

        def find_next_id(id):
            while not bool_time_walk_status[id]:
                id += 1
                if id >= len(bool_time_walk_status):
                    return -1
            if id % crate_size == 0:
                return -1
            return id

        def find_prev_id(id):
            while not bool_time_walk_status[id]:
                id -= 1
                if id < 0:
                    return -1
            if id % crate_size == crate_size - 1:
                return -1
            return id

        def linear_interpolate(x, x1, y1, x2, y2):
            if x1 == x2:
                return y1
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

        for id in ids_to_fill:
            id1 = find_prev_id(id)
            id2 = find_next_id(id)
            no_interpolate = False
            if id1 == -1:
                if id2 == -1:
                    # Crate is empty
                    no_interpolate = True
                # Try to extrapolate instead
                id1 = find_next_id(id2)
                if id1 == -1:
                    # Give up
                    no_interpolate = True
            if id2 == -1:
                # Try to extrapolate
                id2 = find_prev_id(id1)
                if id2 == -1:
                    no_interpolate = True
            if no_interpolate:
                self.pmt_statuses[id] = False
            else:
                for param_name in param_names:
                    params = time_walk_params[param_name]
                    params[id] = linear_interpolate(id, x1=id1, y1=params[id1], x2=id2, y2=params[id2])
                    params[id] = np.clip(
                        params[id],
                        time_walk_params[f"min_{param_name}"],
                        time_walk_params[f"max_{param_name}"],
                    )

        model_save_dir.mkdir(parents=True, exist_ok=True)
        with open(model_save_dir / "filled_time_walk.json", "w") as f:
            saved_time_walk_params = {}
            for name, params in time_walk_params.items():
                if isinstance(params, np.ndarray):
                    saved_time_walk_params[name] = params.tolist()
                else:
                    saved_time_walk_params[name] = params

            json.dump(saved_time_walk_params, f)

        for param_name in param_names:
            setattr(self, param_name, torch.from_numpy(time_walk_params[param_name]))

    def __iter__(self) -> Iterable[dict[Hashable, torch.Tensor]]:
        for inputs, truth in super().__iter__():
            truth["weights"] = torch.ones_like(inputs["qhs"])

            ids = inputs["pmt_ids"]
            time_walk = exp_time_walk(
                q=inputs["qhs"],
                a=self.time_scale[ids],
                b=self.qhs_scale[ids],
                c=self.gradient[ids],
                d=self.intercept[ids],
            )
            inputs["uncal_hit_times"] += time_walk
            inputs["uncal_hit_times"] += self.noise * torch.randn_like(inputs["uncal_hit_times"])

            yield inputs, truth
