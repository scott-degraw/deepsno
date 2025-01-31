import math
from abc import ABC, abstractmethod
from typing import Hashable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.metrics.eval import fwhm


class MetricMonitor(ABC):
    @abstractmethod
    def __init__(self, dset_len):
        pass

    @abstractmethod
    def update(self, predict: dict, truth: dict):
        pass

    @abstractmethod
    def compute(self):
        pass

    def reset(self):
        pass


class PositionMonitor(MetricMonitor):
    def __init__(
        self,
        writer: SummaryWriter,
        min_residual: float = -math.inf,
        max_residual: float = math.inf,
        bins: int = 100,
        name_prefix: str = "validation_metrics",
    ):
        self.writer = writer
        self.min_residual = min_residual
        self.max_residual = max_residual
        self.bins = bins

        self.bin_array = np.linspace(self.min_residual, self.max_residual, num=bins + 1, endpoint=True)
        self.counts = np.zeros((bins, 3), dtype=np.int64)

        self.residual_sum = np.zeros(3, dtype=np.double)
        self.n_points: int = 0
        self.name_prefix = name_prefix

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        residuals = predict["positions"].cpu().numpy() - truth["positions"].cpu().numpy()
        self.n_points += residuals.shape[0]
        batch_counts = np.apply_along_axis(
            lambda x: np.histogram(np.clip(x, self.min_residual, self.max_residual), bins=self.bin_array)[0],
            axis=0,
            arr=residuals,
        )
        self.counts += batch_counts

        self.residual_sum += residuals.sum(0)

    def reset(self) -> None:
        self.counts = 0
        self.residual_sum = 0
        self.n_points = 0

    def compute(self, global_step: int) -> None:
        fig, axis = plt.subplots()
        positions = ["x", "y", "z"]
        for count_per_coord, c in zip(self.counts.T, positions):
            axis.stairs(count_per_coord, self.bin_array, label=c)
        axis.set_xlabel("Position residual (mm)")
        axis.set_ylabel(f"Counts / {self.bin_array[1] - self.bin_array[0]:.2g}")
        axis.legend()
        self.writer.add_figure(f"{self.name_prefix}/position_residuals", fig, global_step=global_step)

        residual_bias = self.residual_sum / self.n_points
        for bias, c in zip(residual_bias, positions):
            self.writer.add_scalar(f"{self.name_prefix}/bias/{c}-mm", bias, global_step=global_step)

        residual_fwhm = np.apply_along_axis(lambda arr: fwhm(arr, self.bin_array), axis=0, arr=self.counts)

        for fwhm_value, c in zip(residual_fwhm, positions):
            self.writer.add_scalar(f"{self.name_prefix}/fwhm/{c}-mm", fwhm_value, global_step=global_step)
