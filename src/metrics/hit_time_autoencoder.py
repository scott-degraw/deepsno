import torch
from torch import nn


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: dict, reference: torch.Tensor):
        predict: torch.FloatTensor = input["predict"]
        not_padding_masks: torch.BoolTensor = ~input["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (reference - predict)

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
        residuals: torch.FloatTensor = not_padding_masks * (reference - predict)

        total_nhits = torch.sum(not_padding_masks)

        return torch.sum((residuals - torch.sum(residuals) / total_nhits).square()) / total_nhits
