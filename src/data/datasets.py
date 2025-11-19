from pathlib import Path
from typing import Hashable

import h5py
import numpy as np
import torch
import torch.distributions as dist
from torch.utils.data import Dataset

# TODO: We might be able to make this quicker. Implement custom that uses a slice of indices. This index slice
# could be fed into the h5py to more efficiently load the data


class PositionRecoDataset(Dataset):
    def __init__(self, path: str | Path, context_len: int, positions: list[str] = ["x", "y", "z"]):
        super().__init__()
        self._path = Path(path)
        self.context_len = context_len
        self._h5_file = h5py.File(path)
        self.positions: list = positions

        self._hit_times_dset: h5py.Dataset = self._h5_file["cal_pmt_events/hit_times"]
        self.n_events = self._hit_times_dset.shape[0]

        if "mean" in self._hit_times_dset.attrs:
            self.hit_time_mean = self._hit_times_dset.attrs["mean"]
        else:
            self.hit_time_mean = None
        if "root_mean_square_deviation" in self._hit_times_dset.attrs:
            self.hit_time_rmsd = self._hit_times_dset.attrs["root_mean_square_deviation"]
        else:
            self.hit_time_rmsd = None

        self._pmt_ids_dset: h5py.Dataset = self._h5_file["cal_pmt_events/ids"]
        self._mc_truth_pos_group: h5py.Dataset = self._h5_file["mc_truth/position"]

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

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        hit_times = torch.from_numpy(self._hit_times_dset[index, : self.context_len])
        # PMT id of 0 indicates a masked PMT
        # TODO: maybe use torch int instead of long. This will decrease the amount of data to pass to GPU but we will
        # have to convert it into long at the gpu which may not be worth it
        pmt_ids = torch.from_numpy(self._pmt_ids_dset[index, : self.context_len]).long()

        truth_position = np.zeros(len(self.positions), dtype=self.position_numpy_dtype)
        for i, c in enumerate(self.positions):
            self._mc_truth_pos_group[c].read_direct(truth_position, index, i)
        truth_position = torch.from_numpy(truth_position)

        return {"hit_times": hit_times, "pmt_ids": pmt_ids}, truth_position

    def __del__(self):
        self._h5_file.close()


class CableDelaysPositionRecoDataset(PositionRecoDataset):
    def __init__(
        self,
        path: str | Path,
        context_len: int,
        mean_delay: float | None = None,
        std_delay: float | None = None,
        delays_save_path: str | Path | None = None,
        delays_file: str | Path | None = None,
        positions: list[str] = ["x", "y", "z"],
    ):
        super().__init__(path=path, positions=positions, context_len=context_len)

        n_pmts = self._h5_file[f"pmt_info/position/{positions[0]}"].shape[0]
        self._pmt_positions = torch.zeros((n_pmts, len(self.positions)), dtype=self.position_torch_dtype)
        for i, c in enumerate(self.positions):
            self._pmt_positions[:, i] = torch.from_numpy(self._h5_file[f"pmt_info/position/{c}"][:])

        if delays_file is not None:
            self.cable_delays = torch.from_numpy(np.loadtxt(delays_file, dtype=np.float32))
            assert (
                len(self.cable_delays) == n_pmts
            ), f"Cable delays from {delays_file} is length {len(self.cable_delays)}, which does not match {n_pmts}"
        else:
            gauss_dist = torch.distributions.Normal(loc=mean_delay, scale=std_delay)

            self.cable_delays: torch.FloatTensor = gauss_dist.sample([n_pmts])
            self.cable_delays[0] = 0.0  # PMT with ID 0 corresponds to masked PMT

            if delays_save_path is not None:
                np.savetxt(delays_save_path, self.cable_delays.numpy())

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, _ = super().__getitem__(index)

        inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

        inputs["hit_times"] += self.cable_delays[inputs["pmt_ids"]]
        inputs["uncal_hit_times"] = inputs.pop("hit_times")

        truth = inputs["uncal_hit_times"]

        return inputs, truth


class TimeWalkDataset(PositionRecoDataset):
    def __init__(
        self,
        path: str | Path,
        context_len: int,
        noise: float,
        charge_scale: float,
        time_scale: float,
        tail_slope: float,
        tail_intercept: float,
        positions: list[str] = ["x", "y", "z"],
    ):
        super().__init__(path=path, positions=positions, context_len=context_len)

        self._qhs_dset = self._h5_file["cal_pmt_events/qhs"]

        n_pmts = self._h5_file[f"pmt_info/position/{positions[0]}"].shape[0]
        self._pmt_positions = torch.zeros((n_pmts, len(self.positions)), dtype=self.position_torch_dtype)
        for i, c in enumerate(self.positions):
            self._pmt_positions[:, i] = torch.from_numpy(self._h5_file[f"pmt_info/position/{c}"][:])

        self.noise = noise
        self.charge_scale = torch.full((n_pmts,), charge_scale)
        self.time_scale = torch.full((n_pmts,), time_scale)
        self.tail_slope = torch.full((n_pmts,), tail_slope)

        tail_norm = dist.Normal(loc=tail_intercept, scale=3)
        self.tail_intercept = tail_norm.sample((n_pmts,))

        self.norm_dist = dist.Normal(loc=0, scale=self.noise)

    def time_walk_generate(self, charges: torch.Tensor, pmt_ids: torch.Tensor, truth: bool = False):
        time_walk = (
            self.time_scale[pmt_ids] * torch.exp(-charges / self.charge_scale[pmt_ids])
            + self.tail_slope[pmt_ids] * charges
            + self.tail_intercept[pmt_ids]
        )

        if not truth:
            time_walk += self.norm_dist.sample(time_walk.shape)

        return time_walk

    def __getitem__(self, index: int) -> dict[Hashable, torch.Tensor]:
        inputs, _ = super().__getitem__(index)

        inputs["pmt_positions"] = self._pmt_positions[inputs["pmt_ids"]]

        qhs = torch.from_numpy(self._qhs_dset[index, : self.context_len])

        inputs["hit_times"] += self.time_walk_generate(qhs, inputs["pmt_ids"])
        inputs["uncal_hit_times"] = inputs.pop("hit_times")

        inputs["qhs"] = qhs

        truth = inputs["uncal_hit_times"]

        return inputs, truth
