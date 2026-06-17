from typing import Iterable

import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, scale: float = 1.0, shift: float = 0.0, log_norm: bool = False):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.log_norm = log_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_norm:
            x = torch.log(x)
        x = (x - self.scale) / self.shift


class ListSequential(nn.Sequential):
    def __init__(self, modules: Iterable[nn.Module]):
        super().__init__(*modules)
