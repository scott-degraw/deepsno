import typing
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: We might be able to make this quicker. Implement custom that uses a slice of indices. This index slice
# could be fed into the h5py to more efficiently load the data


class PositionRecoDataset(Dataset):
    def __init__(self, path: str | Path, positions: list[str] = ["x", "y", "z"]):
        super().__init__()
        self.path = Path(path)
        self.h5_file = h5py.File(path)
        self.positions: list = positions

        self.hit_times_dset: h5py.Dataset = self.h5_file["cal_pmt_events/hit_times"]
        self.n_events = self.hit_times_dset.shape[0]

        if "mean" in self.hit_times_dset.attrs:
            self.hit_time_mean = self.hit_times_dset.attrs["mean"]
        else:
            self.hit_time_mean = None
        if "root_mean_square_deviation" in self.hit_times_dset.attrs:
            self.hit_time_rmsd = self.hit_times_dset.attrs["root_mean_square_deviation"]
        else:
            self.hit_time_rmsd = None

        self.pmt_ids_dset: h5py.Dataset = self.h5_file["cal_pmt_events/ids"]
        self.mc_truth_pos_group: h5py.Dataset = self.h5_file["mc_truth/position"]

        self.position_means = np.empty(len(self.positions), dtype=np.float32)
        self.position_rmsds = np.empty(len(self.positions), dtype=np.float32)
        for i, c in enumerate(self.positions):
            mc_pos_dset = self.mc_truth_pos_group[c]
            if "mean" in mc_pos_dset.attrs:
                self.position_means[i] = mc_pos_dset.attrs["mean"]
            if "root_mean_square_deviation" in mc_pos_dset.attrs:
                self.position_rmsds[i] = mc_pos_dset.attrs["root_mean_square_deviation"]

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int) -> dict[typing.Hashable, torch.Tensor]:
        hit_times = torch.from_numpy(self.hit_times_dset[index])
        pmt_ids = torch.from_numpy(self.pmt_ids_dset[index]).long()  # PMT id of -1 indicates a masked PMT

        position_dtype = self.mc_truth_pos_group[self.positions[0]].dtype
        truth_position = np.zeros(len(self.positions), dtype=position_dtype)
        for i, c in enumerate(self.positions):
            self.mc_truth_pos_group[c].read_direct(truth_position, index, i)
        truth_position = torch.from_numpy(truth_position)

        return {"hit_times": hit_times, "pmt_ids": pmt_ids}, truth_position

    def __del__(self):
        self.h5_file.close()
