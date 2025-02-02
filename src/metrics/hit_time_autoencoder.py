import torch
from torch import nn

from src.metrics.metrics import Metric


class VarianceMetric(Metric):
    def __init__(self, metric_fn: nn.Module):
        self.metric_fn = metric_fn
        self.running_metric: float = 0
        self.n_points: int = 0

    def update(self, predict: dict, truth: dict) -> None:
        batch_size = truth["uncal_hit_times"].shape[0]
        self.n_points += batch_size

        self.running_metric += batch_size * self.metric_fn(predict, truth).item()

    def compute(self) -> float:
        return self.running_metric / self.n_points

    def reset(self) -> None:
        self.running_metric = 0
        self.n_points = 0


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: dict, reference: dict):
        predict: torch.FloatTensor = input["predict"]
        not_padding_masks: torch.BoolTensor = ~input["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (reference["uncal_hit_times"] - predict)

        n_hits = torch.sum(not_padding_masks, dim=-1, keepdims=True)

        batch_size = n_hits.numel()

        centered_residuals = residuals - torch.sum(residuals, dim=-1, keepdims=True) / n_hits

        return torch.sum(not_padding_masks * centered_residuals.square() / n_hits) / batch_size


class TotalVarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: dict, reference: torch.Tensor):
        predict: torch.FloatTensor = input["predict"]
        not_padding_masks: torch.BoolTensor = ~input["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (reference["uncal_hit_times"] - predict)

        total_nhits = torch.sum(not_padding_masks)

        return torch.sum((residuals - torch.sum(residuals) / total_nhits).square()) / total_nhits
