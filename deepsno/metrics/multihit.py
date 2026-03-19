import concurrent.futures
from typing import Callable

import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import _pytree as pytree


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
        self.weights = pytree.tree_map(lambda x: x / sum(weights.values()), weights) if weights else None
        self.costs = None

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
        energy_norm = truth["energy"] / (truth["energy"].sum(-1, keepdim=True) + self.epsilon)
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
        exists_loss = (self.negative_weight * ~truth["exists"] + self.positive_weight * truth["exists"]) * exists_loss
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
            "energy": predict["energy"][..., :, None],
        }

        cross_truth = {
            "position": truth["position"][..., None, :, :],
            "time": truth["time"][..., None, :],
            "energy": truth["energy"][..., None, :],
        }

        for pkey, tkey in zip(
            ["position", "time", "energy"],
            ["position", "time", "energy"],
        ):
            cross_predict[pkey], cross_truth[tkey] = torch.broadcast_tensors(cross_predict[pkey], cross_truth[tkey])

        energy_norm = cross_truth["energy"] / (cross_truth["energy"].sum(-1, keepdim=True) + self.epsilon)
        pos_loss = F.mse_loss(cross_predict["position"], cross_truth["position"], reduction="none")
        # pos_loss = pos_loss / (2 * torch.exp(log_sigma2["position"])) + 0.5 * log_sigma2["position"]
        pos_loss = torch.sum(pos_loss, dim=-1)  # sum over coords
        pos_loss = pos_loss * energy_norm
        # pos_loss = cross_truth["exists"] * pos_loss / n_truth_vertices

        time_loss = F.mse_loss(cross_predict["time"], cross_truth["time"], reduction="none")
        # time_loss = time_loss / (2 * torch.exp(log_sigma2["time"])) + 0.5 * log_sigma2["time"]
        time_loss = time_loss * energy_norm

        energy_loss = F.mse_loss(cross_predict["energy"], cross_truth["energy"], reduction="none")
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

        def _match(i):
            return sp.optimize.linear_sum_assignment(costs[i])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_match, range(costs.shape[0])))

        for batch_i, (p_i, t_i) in enumerate(results):
            pred_i[batch_i] = p_i
            truth_i[batch_i] = t_i

        device = truth["exists"].device
        truth_i = torch.from_numpy(truth_i).to(device)
        pred_i = torch.from_numpy(pred_i).to(device)

        matched_predict = {}
        matched_truth = {}

        matched_predict["position"] = torch.take_along_dim(predict["position"], pred_i[..., None], -2)
        matched_predict["time"] = torch.take_along_dim(predict["time"], pred_i, -1)
        matched_predict["exists_logit"] = torch.take_along_dim(predict["exists_logit"], pred_i, -1)
        matched_predict["energy"] = torch.take_along_dim(predict["energy"], pred_i, -1)

        matched_truth["position"] = torch.take_along_dim(truth["position"], truth_i[..., None], -2)
        matched_truth["time"] = torch.take_along_dim(truth["time"], truth_i, -1)
        matched_truth["exists"] = torch.take_along_dim(truth["exists"], truth_i, -1)
        matched_truth["energy"] = torch.take_along_dim(truth["energy"], truth_i, -1)

        return {**matched_predict, "log_sigma2": predict["log_sigma2"]}, matched_truth

    def forward(self, predict: dict[str, torch.Tensor], truth: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = self.losses(predict, truth)
        losses = self.apply_weights(losses)
        return sum(losses.values()) / len(losses.values())


def marginal_convergence(
    T: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    eps_num: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max relative error of the transport plan's row/col sums vs. target marginals.

    Args:
        T:  Transport plan (..., M, N)
        mu: Row marginal (..., M)
        nu: Col marginal (..., N)
    Returns:
        row_conv: max_i(|sum_j T_ij - mu_i| / mu_i)
        col_conv: max_j(|sum_i T_ij - nu_j| / nu_j)
    """
    # row_conv = ((T.sum(-1) - mu).abs() / (mu + eps_num)).max()
    # col_conv = ((T.sum(-2) - nu).abs() / (nu + eps_num)).max()
    row_conv = (T.sum(-1) - mu).abs().max()
    col_conv = (T.sum(-2) - nu).abs().max()
    return row_conv, col_conv


def sinkhorn_log(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    n_iters: int,
    eps_num: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn returning the transport plan T.

    Args:
        C:  Cost matrix  (..., M, N)
        mu: Row marginal (..., M)   – prediction side
        nu: Col marginal (..., N)   – truth side
        epsilon: Entropic regularization strength
        n_iters: Number of Sinkhorn iterations
        eps_num: Small constant for numerical safety
    Returns:
        T: Transport plan (..., M, N)
        row_conv: Max normalized row-marginal error, max_i(|T.sum(-1) - mu|_i / mu_i)
        col_conv: Max normalized col-marginal error, max_j(|T.sum(-2) - nu|_j / nu_j)
    """
    log_mu = torch.log(mu + eps_num)  # (..., M)
    log_nu = torch.log(nu + eps_num)  # (..., N)
    log_K = -C / epsilon  # (..., M, N)

    # Initialise dual variables
    u = torch.zeros_like(log_mu)  # (..., M)
    v = torch.zeros_like(log_nu)  # (..., N)
    i = torch.zeros((), dtype=torch.int32, device=C.device)

    def cond_fn(i: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return i < n_iters

    def body_fn(i: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_v = log_nu - torch.logsumexp(log_K + u[..., :, None], dim=-2)
        new_u = log_mu - torch.logsumexp(log_K + new_v[..., None, :], dim=-1)
        return i + 1, new_u, new_v

    _, u, v = torch.while_loop(cond_fn, body_fn, (i, u, v))

    log_T = log_K + u[..., :, None] + v[..., None, :]
    T = torch.exp(log_T)

    row_conv, col_conv = marginal_convergence(T, mu, nu, eps_num)
    return T, row_conv, col_conv


@torch.compile
class SinkhornVertexLoss(nn.Module):
    """Differentiable vertex loss using Sinkhorn optimal transport.

    Instead of hard bipartite matching (Hungarian), the loss is computed as a
    soft inner product between a transport plan T and the pairwise cost matrix C:

        L = sum_{ij} T_{ij} * C_{ij}

    The plan T is obtained via Sinkhorn iterations in log-space for numerical
    stability.

    Args:
        positive_weight: Weight applied to positive (existing) vertex BCE terms.
        negative_weight: Weight applied to negative BCE terms.
        weights: Optional per-loss-type weighting dict (same as HungarianVertexLoss).
        epsilon: Entropic regularisation strength (smaller ≈ closer to hard OT).
        n_iters: Number of Sinkhorn iterations.
        eps_num: Small constant for numerical safety.
    """

    def __init__(
        self,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        weights: dict[str, float] | None = None,
        epsilon: float = 0.1,
        n_iters: int = 50,
        eps_num: float = 1e-8,
    ):
        super().__init__()

        self.positive_weight = positive_weight / (positive_weight + negative_weight)
        self.negative_weight = negative_weight / (positive_weight + negative_weight)
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.eps_num = eps_num
        self.weights = pytree.tree_map(lambda x: x / sum(weights.values()), weights) if weights else None

    def apply_weights(self, losses: dict) -> dict:
        if self.weights:
            return {k: w * losses[k] for k, w in self.weights.items()}
        return losses

    def cross_costs(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Compute per-component and combined pairwise cost matrices.

        Returns:
            components: dict of individual cost matrices (..., N_pred, N_truth)
            C: combined (weighted) cost matrix (..., N_pred, N_truth)
        """
        cross_predict = {
            "position": predict["position"][..., :, None, :],
            "time": predict["time"][..., :, None],
            "energy": predict["energy"][..., :, None],
        }
        cross_truth = {
            "position": truth["position"][..., None, :, :],
            "time": truth["time"][..., None, :],
            "energy": truth["energy"][..., None, :],
        }

        for key in ["position", "time", "energy"]:
            cross_predict[key], cross_truth[key] = torch.broadcast_tensors(cross_predict[key], cross_truth[key])

        energy_norm = cross_truth["energy"] / (cross_truth["energy"].sum(-1, keepdim=True) + self.eps_num)

        pos_loss = F.mse_loss(cross_predict["position"], cross_truth["position"], reduction="none")
        pos_loss = torch.sum(pos_loss, dim=-1)  # sum over spatial coords
        pos_loss = pos_loss * energy_norm

        time_loss = F.mse_loss(cross_predict["time"], cross_truth["time"], reduction="none")
        time_loss = time_loss * energy_norm

        energy_loss = F.mse_loss(cross_predict["energy"], cross_truth["energy"], reduction="none")
        energy_loss = energy_loss / energy_loss.shape[-1]

        components = {"position": pos_loss, "time": time_loss, "energy": energy_loss}
        C = sum(self.apply_weights(components).values())  # (..., N_pred, N_truth)
        return components, C

    def losses(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute per-component Sinkhorn losses for monitoring.

        The transport plan T is computed from the combined (weighted) cost C.
        Each component loss is then <T, C_k> — the transport-weighted average
        of that component's pairwise costs.

        Returns a dict of scalar losses: {"position", "time", "energy"}.
        """
        components, C = self.cross_costs(predict, truth)
        norm_C = C / (torch.median(C.flatten(-2, -1), dim=-1).values[..., None, None] + self.eps_num)

        # ------------------------------------------------------------------
        # Build marginals from energy
        # nu (truth side):  ground-truth energy distribution over true vertices
        # mu (predict side): predicted energy distribution over predicted slots
        # ------------------------------------------------------------------
        nu = torch.ones_like(truth["energy"], dtype=torch.float32)

        not_exists = truth["energy"] == 0
        zero_energy_counts = not_exists.sum(-1, keepdim=True)
        nu[not_exists] = 0
        first_not_exists = torch.argmax(not_exists.float(), dim=-1, keepdim=True)
        nu = nu.scatter(-1, first_not_exists, zero_energy_counts.float())

        nu = nu / (nu.sum(-1, keepdim=True) + self.eps_num)

        mu = torch.ones_like(predict["time"], dtype=torch.float32)
        mu = mu / (mu.sum(-1, keepdim=True) + self.eps_num)

        T, row_conv, col_conv = sinkhorn_log(
            C=norm_C,
            mu=mu,
            nu=nu,
            epsilon=self.epsilon,
            n_iters=self.n_iters,
            eps_num=self.eps_num,
        )
        # We don't want the gradients to flow through the iterations themselves
        T = T.detach()

        # TODO: Think if you should include the regularization term here.
        batch_size = float(max(1, T.shape[0] if T.dim() > 2 else 1))
        n_vertices = truth["energy"].shape[-1]
        return {
            **{k: (T * c).sum() * n_vertices / batch_size for k, c in components.items()},
            "sinkhorn_row_conv": row_conv.detach(),
            "sinkhorn_col_conv": col_conv.detach(),
        }

    def forward(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Sinkhorn loss.

        The loss is the sum of transport-weighted per-component costs.
        Plan T sums to 1 per batch element, so <T, C> is a correctly-normalized
        weighted average of pairwise costs.

        Returns a scalar loss.
        """
        all_losses = self.losses(predict, truth)
        component_losses = {k: v for k, v in all_losses.items() if not k.startswith("sinkhorn_")}
        weighted = self.apply_weights(component_losses)
        return sum(weighted.values())


def cardinality_error(
    predict_logit: torch.Tensor,
    truth_exists: torch.Tensor,
    threshold: torch.Tensor = 0.5,
    prob_fn=torch.sigmoid,
):
    threshold = torch.tensor(threshold, device=predict_logit.device, dtype=predict_logit.dtype)
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
