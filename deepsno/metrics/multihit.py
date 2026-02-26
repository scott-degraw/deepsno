from typing import Callable

import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import _pytree as pytree

from deepsno.metrics.metric_monitor import MetricMonitor




class HungarianVertexLoss(nn.Module):
    def __init__(
        self,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        weights: dict[str, float] | None = None,
        ord: int | float = 2,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.positive_weight = positive_weight / (positive_weight + negative_weight)
        self.negative_weight = negative_weight / (positive_weight + negative_weight)
        self.ord = ord
        self.epsilon = epsilon
        self.weights = (
            pytree.tree_map(lambda x: x / sum(weights.values()), weights)
            if weights
            else None
        )

    def apply_weights(self, losses: dict) -> dict:
        if self.weights:
            return {k: w * losses[k] for k, w in self.weights.items()}
        return losses

    def unreduced_losses(
        self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_sigma2 = predict["log_sigma2"]
        n_truth_vertices = truth["exists"].sum(-1, keepdim=True) + self.epsilon
        truth["energy"] = truth["exists"] * truth["energy"]
        energy_norm = truth["energy"] / (
            truth["energy"].sum(-1, keepdim=True) + self.epsilon
        )
        pos_loss = F.mse_loss(predict["position"], truth["position"], reduction="none")
        # pos_loss = pos_loss / (2 * torch.exp(log_sigma2["position"])) + 0.5 * log_sigma2["position"]
        pos_loss = torch.sum(pos_loss, dim=-1)  # sum over coords
        pos_loss = pos_loss * energy_norm
        # pos_loss = truth["exists"] * pos_loss / n_truth_vertices

        time_loss = F.mse_loss(predict["time"], truth["time"], reduction="none")
        # time_loss = time_loss / (2 * torch.exp(log_sigma2["time"])) + 0.5 * log_sigma2["time"]
        time_loss = time_loss * energy_norm
        # time_loss = truth["exists"] * time_loss / n_truth_vertices

        exists_loss = F.binary_cross_entropy_with_logits(
            predict["exists_logit"], truth["exists"].float(), reduction="none"
        )
        exists_loss = (
            self.negative_weight * ~truth["exists"]
            + self.positive_weight * truth["exists"]
        ) * exists_loss
        # exists_loss = exists_loss / torch.exp(log_sigma2["exists"]) + 0.5 * log_sigma2["exists"]
        # normalize by number of vertices
        exists_loss = exists_loss / exists_loss.shape[-1]

        truth["energy"] = truth["exists"] * truth["energy"]
        energy_loss = F.mse_loss(predict["energy"], truth["energy"], reduction="none")
        energy_loss = energy_loss / energy_loss.shape[-1]

        batch_size = np.prod(truth["exists"].shape[:-1]).item()
        losses = {
            "position": pos_loss,
            "time": time_loss,
            # "exists": exists_loss,
            "energy": energy_loss,
        }
        losses = pytree.tree_map(lambda x: x / batch_size, losses)

        return losses

    @torch.no_grad
    def cross_losses(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> np.ndarray:
        log_sigma2 = predict["log_sigma2"]
        cross_predict = {
            "position": predict["position"][..., :, None, :],
            "time": predict["time"][..., :, None],
            "exists_logit": predict["exists_logit"][..., :, None],
            "energy": predict["energy"][..., :, None],
        }

        cross_truth = {
            "position": truth["position"][..., None, :, :],
            "time": truth["time"][..., None, :],
            "exists": truth["exists"][..., None, :],
            "energy": truth["energy"][..., None, :],
        }

        for pkey, tkey in zip(
            ["position", "time", "exists_logit", "energy"],
            ["position", "time", "exists", "energy"],
        ):
            cross_predict[pkey], cross_truth[tkey] = torch.broadcast_tensors(
                cross_predict[pkey], cross_truth[tkey]
            )

        n_truth_vertices = truth["exists"].sum(-1) + self.epsilon
        n_truth_vertices = n_truth_vertices[..., None, None]
        truth["energy"] = truth["exists"] * truth["energy"]
        energy_norm = cross_truth["energy"] / (
            cross_truth["energy"].sum(-1, keepdim=True) + self.epsilon
        )
        pos_loss = F.mse_loss(
            cross_predict["position"], cross_truth["position"], reduction="none"
        )
        # pos_loss = pos_loss / (2 * torch.exp(log_sigma2["position"])) + 0.5 * log_sigma2["position"]
        pos_loss = torch.sum(pos_loss, dim=-1)  # sum over coords
        pos_loss = pos_loss * energy_norm
        # pos_loss = cross_truth["exists"] * pos_loss / n_truth_vertices

        time_loss = F.mse_loss(
            cross_predict["time"], cross_truth["time"], reduction="none"
        )
        # time_loss = time_loss / (2 * torch.exp(log_sigma2["time"])) + 0.5 * log_sigma2["time"]
        time_loss = time_loss * energy_norm
        # time_loss = cross_truth["exists"] * time_loss / n_truth_vertices

        exists_loss = F.binary_cross_entropy_with_logits(
            cross_predict["exists_logit"],
            cross_truth["exists"].float(),
            reduction="none",
        )
        # exists_loss = -(cross_truth["exists"] * F.sigmoid(cross_predict["exists_logit"]))
        exists_loss = (
            self.negative_weight * ~cross_truth["exists"]
            + self.positive_weight * cross_truth["exists"]
        ) * exists_loss
        # exists_loss = exists_loss / torch.exp(log_sigma2["exists"]) + 0.5 * log_sigma2["exists"]
        exists_loss = (
            exists_loss / exists_loss.shape[-1]
        )  # normalize by number of vertices

        cross_truth["energy"] = cross_truth["exists"] * cross_truth["energy"]
        energy_loss = F.mse_loss(
            cross_predict["energy"], cross_truth["energy"], reduction="none"
        )
        energy_loss = energy_loss / energy_loss.shape[-1]

        costs = {
            "position": pos_loss,
            "time": time_loss,
            # "exists": exists_loss,
            "energy": energy_loss,
        }
        costs = self.apply_weights(costs)
        costs = sum(costs.values())  # sum over loss types
        costs = costs.cpu().numpy()

        return costs

    def losses(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        predict, truth = self.bipartite_matching(predict, truth)
        unreduced_losses = self.unreduced_losses(predict, truth)
        reduced_losses = pytree.tree_map(torch.sum, unreduced_losses)

        return reduced_losses

    def bipartite_matching(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        costs = self.cross_losses(predict, truth)

        truth_i = np.full(truth["exists"].shape, dtype=np.int64, fill_value=-1)
        pred_i = np.full(truth["exists"].shape, dtype=np.int64, fill_value=-1)

        for batch_i in np.ndindex(costs.shape[:-2]):
            pred_i[*batch_i], truth_i[*batch_i] = sp.optimize.linear_sum_assignment(
                costs[*batch_i]
            )

        device = truth["exists"].device
        truth_i = torch.from_numpy(truth_i).to(device)
        pred_i = torch.from_numpy(pred_i).to(device)

        matched_predict = {}
        matched_truth = {}

        matched_predict["position"] = torch.take_along_dim(
            predict["position"], pred_i[..., None], -2
        )
        matched_predict["time"] = torch.take_along_dim(predict["time"], pred_i, -1)
        matched_predict["exists_logit"] = torch.take_along_dim(
            predict["exists_logit"], pred_i, -1
        )
        matched_predict["energy"] = torch.take_along_dim(predict["energy"], pred_i, -1)

        matched_truth["position"] = torch.take_along_dim(
            truth["position"], truth_i[..., None], -2
        )
        matched_truth["time"] = torch.take_along_dim(truth["time"], truth_i, -1)
        matched_truth["exists"] = torch.take_along_dim(truth["exists"], truth_i, -1)
        matched_truth["energy"] = torch.take_along_dim(truth["energy"], truth_i, -1)

        return {**matched_predict, "log_sigma2": predict["log_sigma2"]}, matched_truth

    def forward(
        self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        losses = self.losses(predict, truth)
        losses = self.apply_weights(losses)
        return sum(losses.values()) / len(losses.values())


def cardinality_error(
    predict_logit: torch.Tensor,
    truth_exists: torch.Tensor,
    threshold: torch.Tensor = 0.5,
    prob_fn=torch.sigmoid,
):
    threshold = torch.tensor(
        threshold, device=predict_logit.device, dtype=predict_logit.dtype
    )
    predict_exists = prob_fn(predict_logit) > threshold
    return (predict_exists != truth_exists).float().mean()


def padded_chamfer_distance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    pad_mask1: torch.Tensor,
    pad_mask2: torch.Tensor,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
) -> torch.Tensor:
    dist = distance_fn(x1, x2)
    dist = dist + torch.max(dist) * (pad_mask1[..., :, None] + pad_mask2[..., None, :])
    not_pad_mask1 = ~pad_mask1
    not_pad_mask2 = ~pad_mask2
    dist1 = (dist.min(-1).values * not_pad_mask1).sum(-1) / not_pad_mask1.sum(-1)
    dist2 = (dist.min(-2).values * not_pad_mask2).sum(-1) / not_pad_mask2.sum(-1)
    return 0.5 * (dist1 + dist2)


