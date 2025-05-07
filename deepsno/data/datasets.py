from pathlib import Path
from typing import Hashable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


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
    def __init__(
        self,
        path: str | Path,
        context_len: int,
        cut_index_file: str | Path | None = None,
        positions: list[str] = ["x", "y", "z"],
        trigger_offset: float = 0,
        qhs: bool = False,
        seed=74819,
    ):
        super().__init__()
        self._path = Path(path)
        self.context_len = context_len
        self._h5_file = h5py.File(path)

        self.positions: list = positions

        self._hit_times_dset: h5py.Dataset = self._h5_file["cal_pmt_events/hit_times"]
        if qhs and "cal_pmt_events/QHS" in self._h5_file:
            self._qhs_dset: h5py.Dataset = self._h5_file["cal_pmt_events/QHS"]
        else:
            self._qhs_dset = None
        self._times_of_flight_dset = self._h5_file["cal_pmt_events/times_of_flight"]
        self.n_events = self._hit_times_dset.shape[0]

        if "global_trigger_time" in self._h5_file["mc_truth"]:
            self._trigger_time_dset: h5py.Dataset = self._h5_file["mc_truth/global_trigger_time"]
            self.trigger_offset = trigger_offset
        else:
            self._trigger_time_dset = None
            self.trigger_offset = None

        if "mean" in self._hit_times_dset.attrs:
            self.hit_time_mean = self._hit_times_dset.attrs["mean"]
        else:
            self.hit_time_mean = None
        if "root_mean_square_deviation" in self._hit_times_dset.attrs:
            self.hit_time_rmsd = self._hit_times_dset.attrs["root_mean_square_deviation"]
        else:
            self.hit_time_rmsd = None

        self._pmt_ids_dset: h5py.Dataset = self._h5_file["cal_pmt_events/ids"]
        self._mc_truth_pos_group: h5py.Group = self._h5_file["mc_truth/position"]

        self.position_numpy_dtype = self._mc_truth_pos_group[self.positions[0]].dtype
        self.position_torch_dtype = torch.from_numpy(self._mc_truth_pos_group[self.positions[0]][0:1]).dtype

        self.position_means = torch.empty(len(self.positions), dtype=self.position_torch_dtype)
        self.position_rmsds = torch.empty(len(self.positions), dtype=self.position_torch_dtype)
        for i, c in enumerate(self.positions):
            mc_pos_dset = self._mc_truth_pos_group[c]
            if "mean" in mc_pos_dset.attrs:
                self.position_means[i] = mc_pos_dset.attrs["mean"].item()
            if "root_mean_square_deviation" in mc_pos_dset.attrs:
                self.position_rmsds[i] = mc_pos_dset.attrs["root_mean_square_deviation"].item()

        self.generator = np.random.default_rng(seed)
        self.available_indices = np.arange(0, self._hit_times_dset.shape[1])

        if cut_index_file is not None:
            with h5py.File(cut_index_file) as cut_index_h5:
                self.cut_indices = torch.from_numpy(cut_index_h5["cut_indices"][:])
                self.n_events = len(self.cut_indices)
        else:
            self.cut_indices = None

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        if self.cut_indices is not None:
            index = self.cut_indices[index].item()

        pmt_ids = self._pmt_ids_dset[index]
        hit_times = self._hit_times_dset[index]
        if self._qhs_dset is not None:
            qhs = self._qhs_dset[index]
        times_of_flight = self._times_of_flight_dset[index]

        non_zero_pmt_indices = np.nonzero(pmt_ids)[0]
        if len(non_zero_pmt_indices) == 0:
            raise ValueError("Input has no valid PMTs")

        if len(non_zero_pmt_indices) > self.context_len:
            pmt_indices = np.sort(self.generator.choice(non_zero_pmt_indices, size=self.context_len, replace=False))
            pmt_ids = pmt_ids[pmt_indices]
            hit_times = hit_times[pmt_indices]
            times_of_flight = times_of_flight[pmt_indices]
            if self._qhs_dset is not None:
                qhs = qhs[pmt_indices]
        else:
            pmt_ids = pmt_ids[non_zero_pmt_indices]
            hit_times = hit_times[non_zero_pmt_indices]
            times_of_flight = times_of_flight[non_zero_pmt_indices]
            pmt_ids = np.pad(pmt_ids, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            hit_times = np.pad(hit_times, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            if self._qhs_dset is not None:
                qhs = qhs[non_zero_pmt_indices]
                qhs = np.pad(qhs, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
        hit_times -= np.median(hit_times)

        pmt_ids = torch.from_numpy(pmt_ids).long()
        hit_times = torch.from_numpy(hit_times)

        inputs = {"hit_times": hit_times, "pmt_ids": pmt_ids}

        if self._qhs_dset is not None:
            inputs["qhs"] = torch.from_numpy(qhs)

        return inputs, truth


class HitTimeAEDataset(PositionRecoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_pmts = self._h5_file[f"pmt_info/position/{self.positions[0]}"].shape[0]
        self._pmt_positions = torch.zeros((self.n_pmts, len(self.positions)), dtype=self.position_torch_dtype)
        for i, c in enumerate(self.positions):
            self._pmt_positions[:, i] = torch.from_numpy(self._h5_file[f"pmt_info/position/{c}"][:])
            self._pmt_positions[0, i] = 0.0

        if "cal_pmt_events/av_offset" in self._h5_file:
            self._av_offset_dset = self._h5_file["cal_pmt_events/av_offset"]
        else:
            self._av_offset_dset = None

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, truth = super().__getitem__(index)

        inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

        inputs["uncal_hit_times"] = inputs.pop("hit_times")
        truth["uncal_hit_times"] = inputs["uncal_hit_times"]

        truth["pmt_positions"] = inputs["pmt_positions"]
        truth["pmt_ids"] = inputs["pmt_ids"]

        if self._av_offset_dset is not None:
            inputs["av_offset"] = self._av_offset_dset[index]

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
