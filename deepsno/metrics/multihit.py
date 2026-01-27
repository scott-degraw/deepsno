from warnings import warn

import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import _pytree as pytree


class MultiVertexLoss(nn.Module):
    def __init__(self, position_weight: float = 1.0, time_weight: float = 1.0, physical: bool = False):
        super().__init__()
        weight_sum = position_weight + time_weight
        self.position_weight = position_weight / weight_sum
        self.time_weight = time_weight / weight_sum
        self.position_loss_fn = nn.MSELoss()
        self.time_loss_fn = nn.MSELoss()
        self.physical = physical

    def forward(self, predicted: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]) -> torch.Tensor:
        truth = pytree.tree_map(lambda x: x[:, 0], truth)

        loss = self.position_weight * F.mse_loss(predicted["position"], truth["position"])
        if self.physical:
            loss = loss / self.position_weight
            loss = torch.sqrt(loss / 3)
            return loss

        loss = loss + self.time_weight * F.mse_loss(predicted["time"], truth["time"])

        return loss


@torch.no_grad
def bipartite_matching(
    truth_pos: np.ndarray,
    pred_pos: np.ndarray,
    truth_class: np.ndarray,
    pred_class: np.ndarray,
    ord: int | float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    # pos1, pos2: (batch..., max_n_vertex, 3)
    # mask: (batch..., max_n_vertex)

    if not (truth_pos.device == pred_pos.device == truth_class.device == pred_class.device):
        raise ValueError("All inputs to bipartite_matching must be on the same device")

    assert truth_pos.shape == pred_pos.shape
    assert truth_class.shape == pred_class.shape

    displacements = truth_pos[..., :, None, :] - pred_pos[..., None, :, :]

    displacement_costs = np.sum(np.power(np.abs(displacements), ord), axis=-1) / displacements.shape[-1]
    displacement_costs *= truth_class[..., :, None]

    class_costs = -(truth_class[..., :, None] * pred_class[..., None, :])

    cost_matrices = displacement_costs + class_costs

    truth_i = np.full(truth_pos.shape[:-1], dtype=np.int64, fill_value=-1)
    pred_i = np.full(pred_pos.shape[:-1], dtype=np.int64, fill_value=-1)

    for batch_i in np.ndindex(cost_matrices.shape[:-2]):
        truth_i[*batch_i], pred_i[*batch_i] = sp.optimize.linear_sum_assignment(cost_matrices[*batch_i])

    return truth_i, pred_i


class HungarianVertexLoss(nn.Module):
    def __init__(
        self,
        device: torch.device | str = "cpu",
        position_weight: float = 0.0,
        time_weight: float = 0.0,
        class_weight: float = 0.0,
        sigma_requires_grad: bool = False,
        ord: int | float = 2,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        grad = sigma_requires_grad
        self.log_pos_sigma2 = nn.Parameter(torch.full([3], position_weight, device=device), requires_grad=grad)
        self.log_time_sigma2 = nn.Parameter(torch.tensor(time_weight, device=device), requires_grad=grad)
        self.log_class_sigma2 = nn.Parameter(torch.tensor(class_weight, device=device), requires_grad=grad)
        self.ord = ord
        self.epsilon = epsilon

    def unreduced_losses(
        self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_truth_vertices = truth["exists"].sum(-1, keepdim=True) + self.epsilon
        pos_loss = F.mse_loss(predict["position"], truth["position"], reduction="none")
        pos_loss = pos_loss / (2 * torch.exp(self.log_pos_sigma2)) + 0.5 * self.log_pos_sigma2
        pos_loss = torch.sum(pos_loss, dim=-1)  # sum over coords
        pos_loss = truth["exists"] * pos_loss / n_truth_vertices

        time_loss = F.mse_loss(predict["time"], truth["time"], reduction="none")
        time_loss = time_loss / (2 * torch.exp(self.log_time_sigma2)) + 0.5 * self.log_time_sigma2
        time_loss = truth["exists"] * time_loss / n_truth_vertices 

        exists_loss = F.binary_cross_entropy_with_logits(
            predict["exists_logit"], truth["exists"].float(), reduction="none"
        )
        exists_loss = exists_loss / torch.exp(self.log_class_sigma2) + 0.5 * self.log_class_sigma2
        exists_loss = exists_loss / exists_loss.shape[-1]  # normalize by number of vertices

        batch_size = np.prod(truth["exists"].shape[:-1]).item()
        losses = {"pos_loss": pos_loss, "time_loss": time_loss, "exists_loss": exists_loss}
        losses = pytree.tree_map(lambda x: x / batch_size, losses)

        return losses

    def losses(self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        unreduced_losses = self.unreduced_losses(predict, truth)
        reduced_losses = pytree.tree_map(lambda x: torch.sum(x, dim=1), unreduced_losses)
        for key, value in reduced_losses.items():
            batch_i = torch.nonzero(~torch.isfinite(value))
            if len(batch_i) > 0:
                batch_i = batch_i.squeeze().tolist()
                warn(f"Loss {key} is not finite for batch indices {batch_i}")
                print(f"  Corresponding predictions: {[{k: v[batch_i]} for k, v in predict.items()]}")
                print(f"  Corresponding truths: {[{k: v[batch_i]} for k, v in truth.items()]}")

        reduced_losses = pytree.tree_map(torch.sum, reduced_losses)

        return reduced_losses

    def bipartite_matching(self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]):
        cross_predict = {
            "position": predict["position"][..., None, :],
            "time": predict["time"][..., None],
            "logit_exists": predict["logit_exists"][..., None],
        }
        cross_truth = {
            "position": truth["position"][..., None, :, :],
            "time": truth["time"][..., None, :],
            "exists": truth["exists"][..., None, :],
        }

        costs = self.losses(cross_predict, cross_truth)
        costs = pytree.tree_map(lambda x: x.cpu().detach().numpy(), costs)
        cost_matrix = sum(costs.values())

        truth_i = np.full(truth["exists"].shape, dtype=np.int64, fill_value=-1)
        pred_i = np.full(truth["exists"].shape, dtype=np.int64, fill_value=-1)

        for batch_i in np.ndindex(cost_matrix.shape[:-2]):
            truth_i[*batch_i], pred_i[*batch_i] = sp.optimize.linear_sum_assignment(cost_matrix[*batch_i])

        device = truth["exists"].device
        truth_i = torch.from_numpy(truth_i).to(device)
        pred_i = torch.from_numpy(pred_i).to(device)

        matched_predict = {}
        matched_truth = {}

        matched_predict["position"] = torch.take_along_dim(predict["position"], pred_i[..., None], -2)
        matched_predict["time"] = torch.take_along_dim(predict["time"], pred_i, -1)
        matched_predict["exists_logit"] = torch.take_along_dim(predict["exists_logit"], pred_i, -1)

        matched_truth["position"] = torch.take_along_dim(truth["position"], truth_i[..., None], -2)
        matched_truth["time"] = torch.take_along_dim(truth["time"], truth_i, -1)
        matched_truth["exists"] = torch.take_along_dim(truth["exists"], truth_i, -1)

        return matched_predict, matched_truth

    def losses_old(
        self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pred_pos, truth_pos: (batch..., max_n_vertices, 3)
        truth_pos = torch.concat([truth["position"], truth["time"][..., None]], dim=-1)
        pred_pos = torch.concat([predict["position"], predict["time"][..., None]], dim=-1)

        truth_class = truth["exists"]
        pred_class = predict["exists_logit"]

        matched_truth_i, matched_pred_i = bipartite_matching(
            truth_pos=truth_pos.detach().cpu().numpy(),
            pred_pos=pred_pos.detach().cpu().numpy(),
            truth_class=truth_class.detach().cpu().numpy(),
            pred_class=F.sigmoid(predict["exists_logit"]).detach().cpu().numpy(),
        )

        matched_truth_i = torch.from_numpy(matched_truth_i).to(truth_pos.device)
        matched_pred_i = torch.from_numpy(matched_pred_i).to(truth_pos.device)

        predict["position"] = torch.take_along_dim(predict["position"], matched_pred_i[..., None], -2)
        predict["time"] = torch.take_along_dim(predict["time"], matched_pred_i, -1)
        predict["exists_logit"] = torch.take_along_dim(predict["exists_logit"], matched_pred_i, -1)
        truth["position"] = torch.take_along_dim(truth["position"], matched_truth_i[..., None], -2)
        truth["time"] = torch.take_along_dim(truth["time"], matched_truth_i, -1)
        truth["exists"] = torch.take_along_dim(truth["exists"], matched_truth_i, -1)

        truth_pos = torch.take_along_dim(truth_pos, matched_truth_i[..., None], -2)
        pred_pos = torch.take_along_dim(pred_pos, matched_pred_i[..., None], -2)
        truth_class = torch.take_along_dim(truth_class, matched_truth_i, -1)
        pred_class = torch.take_along_dim(pred_class, matched_pred_i, -1)

        metric_loss = F.mse_loss(pred_pos, truth_pos, reduction="none")
        # metric_loss = metric_loss / (2 * torch.exp(self.log_pos_sigma[None, None, :]))
        # metric_loss = torch.sum(metric_loss, dim=-1)
        metric_loss = torch.sum(truth_class[..., None] * metric_loss / truth_class.sum(-1)[..., None, None])
        # metric_loss = torch.sum(truth_class * metric_loss / truth_class.sum(-1, keepdim=True))

        cross_entropy = F.binary_cross_entropy_with_logits(pred_class, truth_class.float())

        return {"metric_loss": metric_loss, "cross_entropy": cross_entropy}

    def forward(self, predicted: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = self.losses(predicted, truth)
        return sum(losses.values())
