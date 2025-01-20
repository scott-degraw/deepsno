import torch
from torch import nn


class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: dict, reference: torch.Tensor):
        predict: torch.FloatTensor = input["predict"]
        not_padding_masks: torch.BoolTensor = ~input["pad_masks"]
        residuals: torch.FloatTensor = not_padding_masks * (reference - predict)

        total_nhits = torch.sum(not_padding_masks)

        return torch.sum(residuals - torch.sum(residuals) / total_nhits) / total_nhits