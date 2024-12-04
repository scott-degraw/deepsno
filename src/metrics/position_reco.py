import torch
from torch import nn
import torch.nn.functional as F


class ResolutionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, truth):
        return torch.sqrt(F.mse_loss(prediction, truth) / 3)
