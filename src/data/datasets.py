from pathlib import Path
from typing import Hashable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: We might be able to make this quicker. Implement custom that uses a slice of indices. This index slice
# could be fed into the h5py to more efficiently load the data


class PositionRecoDataset(Dataset):
    def __init__(self, path: str | Path, positions: list[str] = ["x", "y", "z"]):
        super().__init__()
        self._path = Path(path)
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
        hit_times = torch.from_numpy(self._hit_times_dset[index])
        # PMT id of 0 indicates a masked PMT
        # TODO: maybe use torch int instead of long. This will decrease the amount of data to pass to GPU but we will
        # have to convert it into long at the gpu which may not be worth it
        pmt_ids = torch.from_numpy(self._pmt_ids_dset[index]).long()

        truth_position = np.zeros(len(self.positions), dtype=self.position_numpy_dtype)
        for i, c in enumerate(self.positions):
            self._mc_truth_pos_group[c].read_direct(truth_position, index, i)
        truth_position = torch.from_numpy(truth_position)

        return {"hit_times": hit_times, "pmt_ids": pmt_ids}, truth_position

    def __del__(self):
        self._h5_file.close()
