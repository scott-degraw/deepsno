import math
from abc import ABC, abstractmethod
from typing import Hashable, Iterable

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch.utils.tensorboard import SummaryWriter

from src.metrics.eval import fwhm


class MetricMonitor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, predict: dict, truth: dict):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class MonitorCollection(MetricMonitor):
    def __init__(self, monitors: Iterable[MetricMonitor]):
        self.monitors = monitors

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        for monitor in self.monitors:
            monitor.update(predict, truth)

    def reset(self) -> None:
        for monitor in self.monitors:
            monitor.reset()

    def compute(self, global_step: int) -> None:
        for monitor in self.monitors:
            monitor.compute(global_step)


class PositionMonitor(MetricMonitor):
    def __init__(
        self,
        writer: SummaryWriter,
        min_residual: float = -4000,
        max_residual: float = 4000,
        bins: int = 100,
        name_prefix: str = "validation_metrics",
    ):
        self.writer = writer
        self.min_residual = min_residual
        self.max_residual = max_residual
        self.bins = bins

        self.residual_hists = [
            bh.Histogram(bh.axis.Regular(bins, min_residual, max_residual, overflow=False, underflow=False))
            for _ in range(3)
        ]

        self.residual_sum = np.zeros(3, dtype=np.double)
        self.n_points: int = 0
        self.name_prefix = name_prefix

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        all_residuals = predict["positions"].cpu().numpy() - truth["positions"].cpu().numpy()
        for residuals, hist in zip(all_residuals.T, self.residual_hists):
            hist.fill(residuals)

        self.n_points += all_residuals.shape[0]
        self.residual_sum += all_residuals.sum(0)

    def reset(self) -> None:
        for hist in self.residual_hists:
            hist[:] = 0
        self.residual_sum = 0
        self.n_points = 0

    def compute(self, global_step: int) -> None:
        fig, axis = plt.subplots()
        positions = ["x", "y", "z"]
        for hist, c in zip(self.residual_hists, positions):
            axis.stairs(hist.values(), hist.axes[0].edges, label=c)
        axis.set_xlabel("Position residual (mm)")
        axis.set_ylabel("Counts")
        axis.legend()
        self.writer.add_figure(f"{self.name_prefix}/position_residuals", fig, global_step=global_step)

        residual_bias = self.residual_sum / self.n_points
        for bias, c in zip(residual_bias, positions):
            self.writer.add_scalar(f"{self.name_prefix}/bias/{c}-mm", bias, global_step=global_step)

        residual_fwhm = [fwhm(hist.view(), hist.axes[0].edges) for hist in self.residual_hists]
        for fwhm_value, c in zip(residual_fwhm, positions):
            self.writer.add_scalar(f"{self.name_prefix}/fwhm/{c}-mm", fwhm_value, global_step=global_step)
