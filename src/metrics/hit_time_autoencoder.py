import torch
from torch import nn


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: dict, reference: torch.Tensor):
        predict: torch.FloatTensor = input["predict"]
        pad_masks: torch.BoolTensor = input["pad_masks"]
        residuals: torch.FloatTensor = ~pad_masks * (reference - predict)

        n_hits: torch.Tensor = pad_masks.shape[-1] - torch.sum(pad_masks, dim=-1, keepdims=True)

        return torch.sum((residuals - torch.sum(residuals, dim=-1, keepdims=True) / n_hits).square() / n_hits)
