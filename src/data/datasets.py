from pathlib import Path
from typing import Hashable
from warnings import warn

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PositionRecoDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        context_len: int,
        cut_index_file: str | Path | None = None,
        min_hit_time: float | None = None,
        max_hit_time: float | None = None,
        delays_file: str | Path = None,
        positions: list[str] = ["x", "y", "z"],
        trigger_offset: float = 0,
        seed=74819,
    ):
        super().__init__()
        self._path = Path(path)
        self.context_len = context_len
        self._h5_file = h5py.File(path)

        self.positions: list = positions

        self._hit_times_dset: h5py.Dataset = self._h5_file["cal_pmt_events/hit_times"]
        self._times_of_flight_dset = self._h5_file["cal_pmt_events/times_of_flight"]
        self.n_events = self._hit_times_dset.shape[0]
        n_pmts = self._h5_file[f"pmt_info/position/{positions[0]}"].shape[0]

        if delays_file is not None:
            self.cable_delays = torch.from_numpy(np.loadtxt(delays_file, dtype=np.float32))
            assert len(self.cable_delays) == n_pmts, (
                f"Cable delays from {delays_file} is length {len(self.cable_delays)}, which does not match {n_pmts}"
            )
        else:
            self.cable_delays = None

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

        self.min_hit_time = min_hit_time
        self.max_hit_time = max_hit_time

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        if self.cut_indices is not None:
            index = self.cut_indices[index].item()

        pmt_ids = self._pmt_ids_dset[index]
        hit_times = self._hit_times_dset[index]
        times_of_flight = self._times_of_flight_dset[index]

        if self.min_hit_time is not None:
            pmt_ids[hit_times < self.min_hit_time] = 0
        if self.max_hit_time is not None:
            pmt_ids[hit_times > self.max_hit_time] = 0

        non_zero_pmt_indices = np.nonzero(pmt_ids)[0]
        if len(non_zero_pmt_indices) == 0:
            warn("Input has no valid PMTs", UserWarning)

        if len(non_zero_pmt_indices) > self.context_len:
            pmt_indices = np.sort(self.generator.choice(non_zero_pmt_indices, size=self.context_len, replace=False))
            pmt_ids = pmt_ids[pmt_indices]
            hit_times = hit_times[pmt_indices]
            times_of_flight = times_of_flight[pmt_indices]
        else:
            pmt_ids = pmt_ids[non_zero_pmt_indices]
            hit_times = hit_times[non_zero_pmt_indices]
            times_of_flight = times_of_flight[non_zero_pmt_indices]
            pmt_ids = np.pad(pmt_ids, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            hit_times = np.pad(hit_times, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))
            times_of_flight = np.pad(times_of_flight, pad_width=(0, self.context_len - len(non_zero_pmt_indices)))

        pmt_ids = torch.from_numpy(pmt_ids).long()
        hit_times = torch.from_numpy(hit_times)
        times_of_flight = torch.from_numpy(times_of_flight)

        truth_position = np.zeros(len(self.positions), dtype=self.position_numpy_dtype)
        for i, c in enumerate(self.positions):
            self._mc_truth_pos_group[c].read_direct(truth_position, index, i)
        truth_position = torch.from_numpy(truth_position)

        inputs = {"hit_times": hit_times, "pmt_ids": pmt_ids}
        truth = {"positions": truth_position, "times_of_flight": times_of_flight}

        if self._trigger_time_dset is not None:
            truth["event_times"] = self.trigger_offset - self._trigger_time_dset[index]

        if self.cable_delays is not None:
            inputs["hit_times"] += self.cable_delays[inputs["pmt_ids"]]

        return inputs, truth


class CableDelaysPositionRecoDataset(PositionRecoDataset):
    def __init__(
        self,
        path: str | Path,
        context_len: int,
        min_hit_time: float | None = None,
        max_hit_time: float | None = None,
        delays_file: str | Path = None,
        cut_index_file: str | Path = None,
        trigger_offset: float = 0,
        positions: list[str] = ["x", "y", "z"],
        seed=74819,
    ):
        super().__init__(
            path=path,
            context_len=context_len,
            min_hit_time=min_hit_time,
            max_hit_time=max_hit_time,
            delays_file=delays_file,
            cut_index_file=cut_index_file,
            trigger_offset=trigger_offset,
            positions=positions,
            seed=seed,
        )

        n_pmts = self._h5_file[f"pmt_info/position/{positions[0]}"].shape[0]
        self._pmt_positions = torch.zeros((n_pmts, len(self.positions)), dtype=self.position_torch_dtype)
        for i, c in enumerate(self.positions):
            self._pmt_positions[:, i] = torch.from_numpy(self._h5_file[f"pmt_info/position/{c}"][:])
            self._pmt_positions[0, i] = 0.0

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, truth = super().__getitem__(index)

        inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

        inputs["uncal_hit_times"] = inputs.pop("hit_times")
        truth["uncal_hit_times"] = inputs["uncal_hit_times"]

        truth["pmt_positions"] = inputs["pmt_positions"]
        truth["pmt_ids"] = inputs["pmt_ids"]

        return inputs, truth
