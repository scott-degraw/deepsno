import json
import os
import shutil
from pathlib import Path
from typing import Hashable, Iterator

import numpy as np
import torch
import uproot as ur
from torch.utils.data import Dataset, Sampler

from deepsno.models.hit_time_autoencoder import exp_time_walk


class BlockedRandomSampler(Sampler[int]):
    def __init__(self, data_source: Dataset, block_size: int = 10_000):
        self.data_source = data_source
        block_indices = torch.arange(0, len(data_source), block_size)
        block_indices = torch.concat([block_indices, torch.tensor([len(data_source) - 1])])
        self.ranges = torch.concat([block_indices[:-1].unsqueeze(1), block_indices[1:].unsqueeze(1)], dim=1)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        for block_num in torch.randperm(self.ranges.shape[0]):
            block_size = self.ranges[block_num, 1] - self.ranges[block_num, 0]
            perm_indices = self.ranges[block_num, 0] + torch.randperm(block_size)
            for index in perm_indices:
                yield index.item()


class PositionRecoDataset(Dataset):
    expressions = ["pmt_id", "pmt_hit_time"]

    def __init__(
        self,
        path: str | Path,
        context_len: int,
        condor_scratch: bool = False,
        checkpoint_dir: str | Path = None,
        cut: str | None = None,
        max_n_events: int | None = None,
        trigger_offset: float = 0,
        qhs: bool = False,
        status_mask: int = 0xFFFFFFFF,
        seed=74819,
    ):
        super().__init__()
        if condor_scratch:
            condor_scratch_path = Path(os.environ["_CONDOR_SCRATCH_DIR"]) / Path(path).name
            if not condor_scratch_path.exists():
                print(f"Copying dataset at {str(path)} to condor scratch directory...")
                shutil.copy(path, condor_scratch_path)
            self._path = condor_scratch_path
        else:
            self._path = path

        self.context_len = context_len
        self.generator = np.random.default_rng(seed)
        self.trigger_offset = trigger_offset

        print(f"Loading dataset from {self._path} with context length {self.context_len}")

        with ur.open(path) as direc:
            transpose = direc["transpose"]
            status = transpose["status"].array(library="np")
            valid_pmts = ~(status_mask & status)
            self.pmt_statuses = valid_pmts.astype(np.bool)

        with ur.open({path: "event"}) as event_tree:
            self.n_events = event_tree.num_entries
            self.read_qhs = qhs and "pmt_qhs" in event_tree
            if "mc/global_trigger_time" in event_tree:
                self.expressions.append("mc/global_trigger_time")
            if "mc/times_of_flight" in event_tree:
                self.expressions.append("mc/times_of_flight")
            mc_pos_names = {"mcPosx", "mcPosy", "mcPosz"}
            if self.read_qhs:
                self.expressions.append("pmt_qhs")
            print("Opening event tree with expressions:", self.expressions)
            self.event_arrays = event_tree.arrays(self.expressions, cut=cut, library="np", entry_stop=max_n_events)
            self.n_events = (
                min(max_n_events, event_tree.num_entries) if max_n_events is not None else event_tree.num_entries
            )
            if mc_pos_names < set(event_tree.keys()):
                self.event_arrays["mc_pos"] = np.stack(
                    [event_tree[name].array(library="np") for name in ["mcPosx", "mcPosy", "mcPosz"]], axis=-1
                )

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        pmt_ids = self.event_arrays["pmt_id"][index]
        pmt_ids *= self.pmt_statuses[pmt_ids]
        hit_times = self.event_arrays["pmt_hit_time"][index]
        if "mc/times_of_flight" in self.event_arrays:
            times_of_flight = self.event_arrays["mc/times_of_flight"][index]
        else:
            times_of_flight = None
        if self.read_qhs:
            qhs = self.event_arrays["pmt_qhs"][index]

        non_zero_pmt_indices = np.nonzero(pmt_ids)[0]
        if len(non_zero_pmt_indices) == 0:
            raise ValueError("Input has no valid PMTs")

        if len(non_zero_pmt_indices) > self.context_len:
            pmt_indices = np.sort(self.generator.choice(non_zero_pmt_indices, size=self.context_len, replace=False))
            pmt_ids = pmt_ids[pmt_indices]
            hit_times = hit_times[pmt_indices]
            if times_of_flight is not None:
                times_of_flight = times_of_flight[pmt_indices]
            if self.read_qhs is not None:
                qhs = qhs[pmt_indices]
        else:
            pmt_ids = pmt_ids[non_zero_pmt_indices]
            hit_times = hit_times[non_zero_pmt_indices]
            if times_of_flight is not None:
                times_of_flight = times_of_flight[non_zero_pmt_indices]
            pmt_ids = np.pad(pmt_ids, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            hit_times = np.pad(hit_times, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            if self.read_qhs:
                qhs = qhs[non_zero_pmt_indices]
                qhs = np.pad(qhs, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            if times_of_flight is not None:
                times_of_flight = np.pad(times_of_flight, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))

        truth = {}
        if times_of_flight is not None:
            truth["times_of_flight"] = torch.from_numpy(times_of_flight)

        if "mc_pos" in self.event_arrays:
            truth["position"] = torch.from_numpy(self.event_arrays["mc_pos"][index])

        if "mc/global_trigger_time" in self.event_arrays:
            truth["event_times"] = self.trigger_offset - self.event_arrays["mc/global_trigger_time"][index].item()

        hit_times -= np.median(hit_times)

        pmt_ids = torch.from_numpy(pmt_ids).long()
        hit_times = torch.from_numpy(hit_times)

        inputs = {"hit_times": hit_times, "pmt_ids": pmt_ids}

        if self.read_qhs:
            inputs["qhs"] = torch.from_numpy(qhs)

        return inputs, truth


class HitTimeAEDataset(PositionRecoDataset):
    def __init__(self, *args, **kwargs):
        self.expressions.append("av_offset")
        super().__init__(*args, **kwargs)

        with ur.open({self._path: "pmt_info"}) as pmt_info:
            pos = pmt_info["pos"].array(library="np")
            self.n_pmts = pos.shape[0]
            self._pmt_positions = torch.from_numpy(pos)

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, truth = super().__getitem__(index)

        inputs["av_offset"] = torch.from_numpy(self.event_arrays["av_offset"][index])
        inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

        inputs["uncal_hit_times"] = inputs.pop("hit_times")
        truth["uncal_hit_times"] = inputs["uncal_hit_times"]

        truth["pmt_positions"] = inputs["pmt_positions"]
        truth["pmt_ids"] = inputs["pmt_ids"]

        return inputs, truth


class CableDelayDataset(HitTimeAEDataset):
    def __init__(self, delays_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if delays_file is not None:
            self.cable_delays = torch.from_numpy(np.loadtxt(delays_file, dtype=np.float32))
            assert len(self.cable_delays) == self.n_pmts, (
                f"Cable delays from {delays_file} is length {len(self.cable_delays)}, "
                "which does not match {self.n_pmts}"
            )
        else:
            self.cable_delays = None

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, truth = super().__getitem__(index)

        if self.cable_delays is not None:
            inputs["uncal_hit_times"] += self.cable_delays[inputs["pmt_ids"]]

        return inputs, truth


class TimeWalkDataset(HitTimeAEDataset):
    def __init__(
        self,
        a_mean: float = 0.0,
        a_std: float = 3.0,
        b_mean: float = 0.0,
        b_std: float = 2.0,
        c_mean: float = 50,
        c_std: float = 10,
        noise: float = 1.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.add_noise = True

        self.a = torch.from_numpy(self.generator.normal(a_mean, a_std, self.n_pmts))
        self.b = torch.from_numpy(self.generator.normal(b_mean, b_std, self.n_pmts))
        self.c = torch.from_numpy(self.generator.normal(c_mean, c_std, self.n_pmts))

        self.a[0] = 0.0
        self.b[0] = 0.0

        self.noise = noise

    def time_walk(self, pmt_ids: torch.LongTensor, qhs: torch.FloatTensor):
        times = self.a[pmt_ids] + self.b[pmt_ids] * torch.exp(-qhs / self.c[pmt_ids])

        if self.add_noise:
            times += torch.from_numpy(self.generator.normal(0, self.noise, times.shape))

        return times

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, truth = super().__getitem__(index)

        inputs["uncal_hit_times"] += self.time_walk(pmt_ids=inputs["pmt_ids"], qhs=inputs["qhs"])

        return inputs, truth
