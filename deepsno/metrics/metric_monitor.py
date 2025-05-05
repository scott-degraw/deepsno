from abc import ABC, abstractmethod
from typing import Hashable, Iterable

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch.utils.tensorboard import SummaryWriter

from deepsno.metrics.eval import fwhm


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


class TimeResidualMonitor(MetricMonitor):
    def __init__(
        self,
        writer: SummaryWriter,
        effective_c: float,
        offset: float = 0.0,
        scale: float = 1.0,
        min_residual: float = -50,
        max_residual: float = 300,
        bins: int = 100,
        name_prefix: str = "validation_metrics",
    ):
        self.writer = writer
        self.min_residual = min_residual
        self.max_residual = max_residual
        self.bins = bins
        self.offset = offset
        self.scale = scale

        self.predict_hist = bh.Histogram(
            bh.axis.Regular(bins, min_residual, max_residual, overflow=True, underflow=True)
        )
        self.truth_hist = bh.Histogram(bh.axis.Regular(bins, min_residual, max_residual, overflow=True, underflow=True))

        self.name_prefix = name_prefix
        self.effective_c = effective_c

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        predict = pytree.tree_map(lambda x: x.cpu().numpy(), predict)
        truth = pytree.tree_map(lambda x: x.cpu().numpy(), truth)
        not_padding_mask = truth["pmt_ids"] != 0
        predicted_time_residuals = truth["uncal_hit_times"] - predict["times_of_flight"]
        predicted_time_residuals = predicted_time_residuals[not_padding_mask]
        predicted_time_residuals = predicted_time_residuals.ravel() * self.scale + self.offset
        predicted_time_residuals = predicted_time_residuals - np.mean(predicted_time_residuals)
        self.predict_hist.fill(predicted_time_residuals)

        truth_time_residuals = truth["uncal_hit_times"] - (truth["times_of_flight"] + truth["event_times"][..., None])
        truth_time_residuals = truth_time_residuals[not_padding_mask]
        truth_time_residuals -= np.mean(truth_time_residuals)
        self.truth_hist.fill(truth_time_residuals)

    def reset(self) -> None:
        self.truth_hist[:] = 0
        self.predict_hist[:] = 0

    def compute(self, global_step: int) -> None:
        fig, axis = plt.subplots()
        axis.stairs(self.predict_hist.values(), self.predict_hist.axes[0].edges, label="Predict")
        axis.stairs(self.truth_hist.values(), self.truth_hist.axes[0].edges, label="Truth")
        axis.legend()
        axis.set_xlabel("Time residual (ns)")
        axis.set_ylabel("Counts")
        self.writer.add_figure(f"{self.name_prefix}/time_residuals", fig, global_step=global_step)


class EffectiveCMonitor(MetricMonitor):
    def __init__(
        self, writer: SummaryWriter, position_scale: float, time_scale: float, name_prefix="validation_metrics"
    ):
        self.writer = writer
        self.name_prefix = name_prefix
        self.position_scale = position_scale
        self.time_scale = time_scale

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        self.c_av = predict["c_av"].detach().item() * self.position_scale / self.time_scale
        self.c_water = predict["c_water"].detach().item() * self.position_scale / self.time_scale

    def reset(self) -> None:
        pass

    def compute(self, global_step: int) -> None:
        self.writer.add_scalar(f"{self.name_prefix}/c_av", self.c_av, global_step=global_step)
        self.writer.add_scalar(f"{self.name_prefix}/c_water", self.c_water, global_step=global_step)
