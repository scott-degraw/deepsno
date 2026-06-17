"""Synthetic datasets with no dependency on real detector data or a GPU.

Useful for smoke-testing the training/predict loops and the Hydra config
plumbing without needing ROOT files or CUDA.
"""

import torch
from torch.utils.data import Dataset


class RandomPositionRecoDataset(Dataset):
    """Random (pmt_ids, hit_times) inputs and random vertex positions for `PositionReco`."""

    def __init__(self, n_pmts: int, context_len: int, length: int = 1000, seed: int = 0):
        self.n_pmts = n_pmts
        self.context_len = context_len
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        generator = torch.Generator().manual_seed(self.seed + index)
        pmt_ids = torch.randint(1, self.n_pmts, (self.context_len,), generator=generator)
        hit_times = torch.rand(self.context_len, generator=generator) * 400 + 20
        positions = (torch.rand(3, generator=generator) - 0.5) * 6000
        return {"pmt_ids": pmt_ids, "hit_times": hit_times}, {"positions": positions}
