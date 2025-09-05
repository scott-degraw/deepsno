import torch
from torch import nn


class TimeResidualLoss(nn.Module):
    def __init__(self, a: float, b: float, mu: float, sigma: float, offset: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.mu = (mu - offset) / scale
        self.sigma = sigma / scale

        self.offset = offset
        self.scale = scale

    def forward(self, predict: dict, truth: dict):
        time_res: torch.FloatTensor = predict["time_residuals"]
        not_padding_masks: torch.BoolTensor = ~predict["pad_masks"]
        weights: torch.FloatTensor = truth.get("weights", torch.ones_like(time_res))

        # Normalize the weights to the size of the input
        weights = weights / torch.sum(not_padding_masks * weights)

        time_res = (time_res - self.mu) / self.sigma

        fraction = time_res / (self.a + self.b + time_res.square()).sqrt()
        log_likelihoods = (self.a + 0.5) * (1 + fraction).log() + (self.b + 0.5) * (1 - fraction).log()
        log_likelihoods = log_likelihoods * weights
        return -(not_padding_masks * log_likelihoods).sum()


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict: dict, truth: dict):
        times_of_flight: torch.FloatTensor = predict["times_of_flight"]
        not_padding_masks: torch.BoolTensor = ~predict["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (truth["uncal_hit_times"] - times_of_flight)

        n_hits = torch.sum(not_padding_masks, dim=-1, keepdims=True)

        batch_size = n_hits.numel()

        centered_residuals = residuals - torch.sum(residuals, dim=-1, keepdims=True) / n_hits

        return torch.sum(not_padding_masks * centered_residuals.square() / n_hits) / batch_size


class TotalVarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict: dict, truth: torch.Tensor):
        times_of_flight: torch.FloatTensor = predict["times_of_flight"]
        not_padding_masks: torch.BoolTensor = ~predict["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (truth["uncal_hit_times"] - times_of_flight)

        total_nhits = torch.sum(not_padding_masks)

        return torch.sum((residuals - torch.sum(residuals) / total_nhits).square()) / total_nhits
