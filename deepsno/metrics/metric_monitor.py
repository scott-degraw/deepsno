from abc import ABC, abstractmethod
from typing import Hashable, Iterable

import hist as h
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torchmetrics as tm
import wandb
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils.tensorboard import SummaryWriter

from deepsno.metrics.eval import fwhm
from deepsno.viz import COLORSCALE, time_colorscale, voxel_mesh


class MetricMonitor(ABC):
    def __init__(self, run: wandb.Run, name_prefix: str = "validation_metrics"):
        self.run = run
        self.name_prefix = name_prefix
        self.reset()

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



class MultiLossMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        multi_loss_fn: nn.Module,
        name_prefix: str = "multi_loss",
        scales: dict[str, float] | None = None,
        sqrt_scaled: bool = True,
    ):
        self.run = run
        self.name_prefix = name_prefix
        self.multi_loss_fn = multi_loss_fn
        self.scales = scales
        self.sqrt_scaled = sqrt_scaled
        self.reset()

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        losses = self.multi_loss_fn.losses(predict, truth)

        for key, value in losses.items():
            self.losses.setdefault(key, []).append(value.detach().item())

    def reset(self) -> None:
        self.losses = {}

    def compute(self, global_step: int) -> None:
        mean_losses = {key: np.mean(values) for key, values in self.losses.items()}
        self.run.log({f"{self.name_prefix}/{key}": value for key, value in mean_losses.items()}, step=global_step)

        if self.scales:
            scaled = {key: mean_losses[key] * scale**2 for key, scale in self.scales.items() if key in mean_losses}
            if self.sqrt_scaled:
                scaled = {key: value**0.5 for key, value in scaled.items()}
            self.run.log({f"{self.name_prefix}_scaled/{key}": value for key, value in scaled.items()}, step=global_step)


class BinaryClassMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        truth_key: str,
        predict_key: str,
        metrics: list[tm.Metric],
        logits: bool = True,
        threshold: float = 0.5,
        name_prefix: str = "classification_metrics",
    ):
        self.run = run
        self.name_prefix = name_prefix
        self.truth_key = truth_key
        self.predict_key = predict_key
        self.metrics = metrics
        self.logits = logits
        self.reset()

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        pred_prob = predict[self.predict_key]
        if self.logits:
            pred_prob = torch.sigmoid(pred_prob)

        truth_class = truth[self.truth_key].bool()

        for metric in self.metrics:
            metric.to(pred_prob.device)
            metric.update(pred_prob, truth_class)

    def compute(self, global_step: int) -> None:
        for metric in self.metrics:
            value = metric.compute().detach().item()
            self.run.log({f"{self.name_prefix}/{metric.__class__.__name__}": value}, step=global_step)


class PositionMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        min_residual: float = -4000,
        max_residual: float = 4000,
        bins: int = 100,
        name_prefix: str = "validation_metrics",
    ):
        self.writer = run
        self.min_residual = min_residual
        self.max_residual = max_residual
        self.bins = bins

        self.residual_hists = [
            h.Hist(h.axis.Regular(bins, min_residual, max_residual, overflow=False, underflow=False, name=""))
            for name, label in zip(["x", "y", "z"], [r"$x$", r"$y$", r"$z$"])
        ]

        self.residual_sum = np.zeros(3, dtype=np.double)
        self.n_points: int = 0
        self.name_prefix = name_prefix

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        truth_positions = np.stack([truth[f"mcPos{c}"].cpu().numpy() for c in ["x", "y", "z"]], axis=-1)
        all_residuals = predict["positions"].cpu().numpy() - truth_positions
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

        axis.axvline(0, plt.rcParams["axes.linewidth"])
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

        self.predict_hist = h.Hist(h.axis.Regular(bins, min_residual, max_residual, overflow=True, underflow=True))
        self.truth_hist = h.Hist(h.axis.Regular(bins, min_residual, max_residual, overflow=True, underflow=True))

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


class EventDisplayMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        n_events: int = 4,
        energy_threshold: float = 0.0,
        voxel_size: float = 10.0,
        position_scale: float = 1.0,
        position_shift: float = 0.0,
        time_scale: float = 1.0,
        time_shift: float = 0.0,
        energy_scale: float = 1.0,
        energy_shift: float = 0.0,
        name_prefix: str = "event_display",
    ):
        self.run = run
        self.n_events = n_events
        self.energy_threshold = energy_threshold
        self.voxel_size = voxel_size
        self.position_scale = position_scale
        self.position_shift = position_shift
        self.time_scale = time_scale
        self.time_shift = time_shift
        self.energy_scale = energy_scale
        self.energy_shift = energy_shift
        self.name_prefix = name_prefix
        self.reset()

    def reset(self) -> None:
        self._predict_events: list[dict] = []
        self._truth_events: list[dict] = []

    def update(self, predict: dict[Hashable, torch.Tensor], truth: dict[Hashable, torch.Tensor]) -> None:
        if len(self._predict_events) >= self.n_events:
            return

        needed = self.n_events - len(self._predict_events)
        batch_size = predict["position"].shape[0]
        take = min(needed, batch_size)

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().float().numpy()

        for i in range(take):
            self._predict_events.append({
                "position": _np(predict["position"][i]),
                "time": _np(predict["time"][i]),
                "energy": _np(predict["energy"][i]),
                "exists_logit": _np(predict["exists_logit"][i]),
            })
            self._truth_events.append({
                "position": _np(truth["position"][i]),
                "time": _np(truth["time"][i]),
                "energy": _np(truth["energy"][i]),
                "exists": _np(truth["exists"][i]).astype(bool),
            })

    def _unnorm(self, pos, t, e):
        pos = pos * self.position_scale + self.position_shift
        t = t * self.time_scale + self.time_shift
        e = e * self.energy_scale + self.energy_shift
        return pos, t, e

    def _make_figure(self, pred: dict, truth: dict, event_idx: int) -> go.Figure:
        normed_threshold = self.energy_threshold / self.energy_scale
        pred_mask = pred["energy"] > normed_threshold
        truth_mask = truth["exists"] & (truth["energy"] > normed_threshold)

        pred_pos, pred_t, pred_e = self._unnorm(
            pred["position"][pred_mask], pred["time"][pred_mask], pred["energy"][pred_mask]
        )
        truth_pos, truth_t, truth_e = self._unnorm(
            truth["position"][truth_mask], truth["time"][truth_mask], truth["energy"][truth_mask]
        )

        all_times = np.concatenate([pred_t, truth_t]) if (len(pred_t) and len(truth_t)) else np.array([0.0, 1.0])
        t_min, t_max = float(all_times.min()), float(all_times.max())
        e_max = float(truth_e.max()) if len(truth_e) else 1.0

        traces = []

        if len(pred_pos):
            traces.append(go.Scatter3d(
                x=pred_pos[:, 0], y=pred_pos[:, 1], z=pred_pos[:, 2],
                mode="markers",
                name="Predicted",
                marker=dict(
                    size=4,
                    color=time_colorscale(pred_t, t_min, t_max),
                    symbol="diamond",
                ),
                customdata=np.stack([pred_t, pred_e], axis=1),
                hovertemplate="t=%{customdata[0]:.2f} ns  E=%{customdata[1]:.0f}<extra>Predicted</extra>",
            ))

        if len(truth_pos):
            traces.append(voxel_mesh(
                centers=truth_pos,
                energies=truth_e,
                voxel_size=self.voxel_size,
                e_max=e_max,
                colorbar_title="Truth energy (photons)",
                colorbar_x=-0.15,
                opacity=0.4,
                name="Truth voxels",
            ))

        # Ghost trace to render the time colorbar
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(
                colorscale=COLORSCALE,
                cmin=t_min,
                cmax=t_max,
                showscale=True,
                colorbar=dict(title="Time (ns)", x=-0.3),
            ),
            hoverinfo="none",
            showlegend=False,
        ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Event {event_idx + 1}  |  pred: {len(pred_pos)} pts  |  truth: {len(truth_pos)} voxels",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode="data",
            ),
            height=600,
        )
        return fig

    def compute(self, global_step: int) -> None:
        figures = {
            f"{self.name_prefix}/event_{i}": wandb.Plotly(self._make_figure(pred, truth, i))
            for i, (pred, truth) in enumerate(zip(self._predict_events, self._truth_events))
        }
        if figures:
            self.run.log(figures, step=global_step)


class SinkhornConvergenceMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        loss_fn: nn.Module,
        name_prefix: str = "sinkhorn_convergence",
    ):
        self.run = run
        self.loss_fn = loss_fn
        self.name_prefix = name_prefix
        self.reset()

    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        if "max_delta_u" in predict:
            delta_u = predict["max_delta_u"]
            if isinstance(delta_u, torch.Tensor):
                delta_u = delta_u.detach().cpu().item()
            self.max_delta_u.append(delta_u)
        if "max_delta_v" in predict:
            delta_v = predict["max_delta_v"]
            if isinstance(delta_v, torch.Tensor):
                delta_v = delta_v.detach().cpu().item()
            self.max_delta_v.append(delta_v)

    def reset(self) -> None:
        self.max_delta_u = []
        self.max_delta_v = []

    def compute(self, global_step: int) -> None:
        metrics = {}
        if self.max_delta_u:
            metrics[f"{self.name_prefix}/max_delta_u"] = np.mean(self.max_delta_u)
        if self.max_delta_v:
            metrics[f"{self.name_prefix}/max_delta_v"] = np.mean(self.max_delta_v)
        
        if metrics:
            self.run.log(metrics, step=global_step)

