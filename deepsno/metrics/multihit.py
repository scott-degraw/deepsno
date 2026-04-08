import concurrent.futures
from typing import Callable, NamedTuple

import geomloss
import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.utils import _pytree as pytree


def vertex_cost_matrix(
    predict: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    eps_num: float = 1e-8,
    return_components: bool = False,
) -> torch.Tensor | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Pairwise vertex cost matrix between predicted and truth point clouds.

    Position and time costs are weighted by the truth energy normalisation, rescaled
    so the sum over truth vertices equals N_truth.  Energy uses plain squared MSE.

    Args:
        predict: dict with "position" (..., N_pred, 3), "time" (..., N_pred), "energy" (..., N_pred).
        truth:   dict with "position" (..., N_truth, 3), "time" (..., N_truth), "energy" (..., N_truth).
        weights: optional per-component weights {"position", "time", "energy"}; normalised internally.
        eps_num: small constant for numerical safety.
        return_components: if True return (components, C), otherwise return C only.

    Returns:
        C: combined (weighted) cost matrix (..., N_pred, N_truth).
        If return_components is True, returns (components, C) where components is a dict of
        individual cost matrices with the same shape.
    """
    energy_norm = truth["energy"] / (truth["energy"].sum(-1, keepdim=True) + eps_num)
    energy_norm = energy_norm.shape[-1] * energy_norm  # rescale so that sum over truth vertices = N_truth

    pos_cost = torch.cdist(predict["position"], truth["position"], p=2) ** 2  # (..., N_pred, N_truth)
    time_cost = (predict["time"][..., :, None] - truth["time"][..., None, :]) ** 2
    energy_cost = (predict["energy"][..., :, None] - truth["energy"][..., None, :]) ** 2

    pos_cost = pos_cost * energy_norm[..., None, :]
    time_cost = time_cost * energy_norm[..., None, :]

    components = {"position": pos_cost, "time": time_cost, "energy": energy_cost}

    if weights:
        total = sum(weights.values())
        C = sum((weights[k] / total) * c for k, c in components.items())
    else:
        C = sum(components.values())

    if return_components:
        return components, C
    return C


class HungarianVertexLoss(nn.Module):
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.weights = pytree.tree_map(lambda x: x / sum(weights.values()), weights) if weights else None
        self.costs = None

    def apply_weights(self, losses: dict) -> dict:
        if self.weights:
            return {k: w * losses[k] for k, w in self.weights.items()}
        return losses

    def losses(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        truth["energy"] = truth["exists"] * truth["energy"]
        components, C = vertex_cost_matrix(
            predict, truth, weights=self.weights, eps_num=self.epsilon, return_components=True
        )
        components = {k: c / truth["energy"].shape[-1] for k, c in components.items()}
        C = C / truth["energy"].shape[-1]
        pred_i, truth_i = self.bipartite_matching(C.detach().cpu().numpy())
        batch_size = np.prod(truth["exists"].shape[:-1]).item()
        # index matched pairs from the original component cost matrices
        return {
            k: c[torch.arange(c.shape[0])[:, None], pred_i, truth_i].sum() / batch_size for k, c in components.items()
        }

    @torch.no_grad
    def bipartite_matching(self, costs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        B = costs.shape[0]
        shape = costs.shape[:2]

        pred_i = np.full(shape, dtype=np.int64, fill_value=-1)
        truth_i = np.full(shape, dtype=np.int64, fill_value=-1)

        def _match(i):
            return sp.optimize.linear_sum_assignment(costs[i])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_match, range(B)))

        for b, (p_i, t_i) in enumerate(results):
            pred_i[b] = p_i
            truth_i[b] = t_i

        return torch.from_numpy(pred_i), torch.from_numpy(truth_i)

    def match(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return matched predict/truth dicts (for external use, e.g. visualisation)."""
        with torch.no_grad():
            C = vertex_cost_matrix(predict, truth, weights=self.weights, eps_num=self.epsilon)
        pred_i, truth_i = self.bipartite_matching(C.cpu().numpy())
        device = truth["exists"].device
        pred_i = pred_i.to(device)
        truth_i = truth_i.to(device)

        matched_predict = {
            "position": torch.take_along_dim(predict["position"], pred_i[..., None], -2),
            "time": torch.take_along_dim(predict["time"], pred_i, -1),
            "exists_logit": torch.take_along_dim(predict["exists_logit"], pred_i, -1),
            "energy": torch.take_along_dim(predict["energy"], pred_i, -1),
        }
        matched_truth = {
            "position": torch.take_along_dim(truth["position"], truth_i[..., None], -2),
            "time": torch.take_along_dim(truth["time"], truth_i, -1),
            "exists": torch.take_along_dim(truth["exists"], truth_i, -1),
            "energy": torch.take_along_dim(truth["energy"], truth_i, -1),
        }
        return matched_predict, matched_truth

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
    row_conv = (T.sum(-1) - mu).abs().mean()
    col_conv = (T.sum(-2) - nu).abs().mean()
    return row_conv, col_conv


class SinkhornResult(NamedTuple):
    T: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    row_conv: torch.Tensor
    col_conv: torch.Tensor


def sinkhorn_log(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    n_iters: int,
    u_init: torch.Tensor | None = None,
    v_init: torch.Tensor | None = None,
    eps_num: float = 1e-8,
) -> SinkhornResult:
    """Log-domain Sinkhorn returning the transport plan T and dual variables.

    Args:
        C:  Cost matrix  (..., M, N)
        mu: Row marginal (..., M)   – prediction side
        nu: Col marginal (..., N)   – truth side
        epsilon: Entropic regularization strength
        n_iters: Number of Sinkhorn iterations
        u_init: Optional warm-start for row dual variable (..., M)
        v_init: Optional warm-start for col dual variable (..., N)
        eps_num: Small constant for numerical safety
    Returns:
        T: Transport plan (..., M, N)
        u: Row dual variable (..., M)
        v: Col dual variable (..., N)
        row_conv: Max normalized row-marginal error, max_i(|T.sum(-1) - mu|_i / mu_i)
        col_conv: Max normalized col-marginal error, max_j(|T.sum(-2) - nu|_j / nu_j)
    """
    log_mu = torch.log(mu + eps_num)  # (..., M)
    log_nu = torch.log(nu + eps_num)  # (..., N)
    log_K = -C / epsilon  # (..., M, N)

    # Initialise dual variables (zero or warm-start)
    u = u_init if u_init is not None else torch.zeros_like(log_mu)  # (..., M)
    v = v_init if v_init is not None else torch.zeros_like(log_nu)  # (..., N)
    i = torch.zeros((), dtype=torch.int32, device=C.device)

    def cond_fn(i: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return i < n_iters

    def body_fn(i: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = log_mu - torch.logsumexp(log_K + v[..., None, :], dim=-1)
        v = log_nu - torch.logsumexp(log_K + u[..., :, None], dim=-2)
        return i + 1, u, v

    _, u, v = torch.while_loop(cond_fn, body_fn, (i, u, v))

    log_T = log_K + u[..., :, None] + v[..., None, :]
    T = torch.exp(log_T)

    row_conv, col_conv = marginal_convergence(T, mu, nu, eps_num)
    return SinkhornResult(T=T, u=u, v=v, row_conv=row_conv, col_conv=col_conv)


@torch.compile(dynamic=False, fullgraph=True)
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
        weights: dict[str, float] | None = None,
        epsilon: float = 0.1,
        n_iters: int = 50,
        unbiased: bool = False,
        warm_start: bool = False,
        eps_num: float = 1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.n_iters = n_iters
        self.unbiased = unbiased
        self.warm_start = warm_start
        self.eps_num = eps_num
        self.weights = pytree.tree_map(lambda x: x / sum(weights.values()), weights) if weights else None
        self.register_buffer("u_avg", None)
        self.register_buffer("v_avg", None)

    def apply_weights(self, losses: dict) -> dict:
        if self.weights:
            return {k: w * losses[k] for k, w in self.weights.items()}
        return losses

    def cross_costs(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return vertex_cost_matrix(predict, truth, weights=self.weights, eps_num=self.eps_num, return_components=True)

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

        # ------------------------------------------------------------------
        # Build marginals from energy
        # nu (truth side):  ground-truth energy distribution over true vertices
        # mu (predict side): predicted energy distribution over predicted slots
        # ------------------------------------------------------------------
        nu = torch.ones_like(truth["energy"], dtype=torch.float32)

        # not_exists = truth["energy"] == 0
        # zero_energy_counts = not_exists.sum(-1, keepdim=True)
        # nu[not_exists] = 0
        # first_not_exists = torch.argmax(not_exists.float(), dim=-1, keepdim=True)
        # nu = nu.scatter(-1, first_not_exists, zero_energy_counts.float())

        nu = nu / (nu.sum(-1, keepdim=True) + self.eps_num)

        mu = torch.ones_like(predict["time"], dtype=torch.float32)
        mu = mu / (mu.sum(-1, keepdim=True) + self.eps_num)

        u_init = self.u_avg.expand_as(mu) if self.warm_start and self.u_avg is not None else None
        v_init = self.v_avg.expand_as(nu) if self.warm_start and self.v_avg is not None else None

        # with torch.no_grad():
        result = sinkhorn_log(
            C=C,
            mu=mu,
            nu=nu,
            epsilon=self.epsilon,
            n_iters=self.n_iters,
            u_init=u_init,
            v_init=v_init,
            eps_num=self.eps_num,
        )

        if self.warm_start:
            self.u_avg = result.u.detach().mean(0)
            self.v_avg = result.v.detach().mean(0)

        T = result.T.detach()

        # TODO: Think if you should include the regularization term here.
        batch_size = torch.prod(torch.tensor(T.shape[:-2], dtype=torch.float32)) if T.dim() > 2 else 1.0
        n_vertices = truth["energy"].shape[-1]
        components = {k: (T * c).sum() / batch_size for k, c in components.items()}

        if self.unbiased:
            components_self, C_self = self.cross_costs(predict, predict)
            result_self = sinkhorn_log(
                C=C_self,
                mu=mu,
                nu=mu,
                epsilon=self.epsilon,
                n_iters=self.n_iters,
                eps_num=self.eps_num,
            )

            components_self = {k: (result_self.T.detach() * c).sum() / batch_size for k, c in components_self.items()}

            components = {k: components[k] - 0.5 * components_self[k] for k in components}

        return {
            **components,
            "sinkhorn_row_conv": result.row_conv.mean(0).detach(),
            "sinkhorn_col_conv": result.col_conv.mean(0).detach(),
        }

    def forward(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Sinkhorn loss.

        Returns a scalar loss.
        """
        all_losses = self.losses(predict, truth)
        component_losses = {k: v for k, v in all_losses.items() if not k.startswith("sinkhorn_")}
        weighted = self.apply_weights(component_losses)
        return sum(weighted.values())


class GeomlossSinkhornVertexLoss(nn.Module):
    """Vertex loss using geomloss SamplesLoss (Sinkhorn OT) with the same cost as SinkhornVertexLoss.

    Position and time costs are weighted by the truth energy normalisation; energy uses a plain MSE cost:

        C_ij = energy_norm_j * (w_pos * ||pos_i - pos_j||^2 + w_time * (t_i - t_j)^2)
               + w_energy * (e_i - e_j)^2

    The Sinkhorn divergence is computed between predicted and truth point clouds with
    uniform marginals via ``geomloss.SamplesLoss``.

    Args:
        weights: Optional per-component weighting dict with keys "position", "time", "energy".
        blur: Entropic regularisation strength (geomloss ``blur`` parameter).
        p: Cost exponent (default 2 for squared Euclidean).
        scaling: geomloss multi-scale refinement parameter (range (0, 1)).
        debias: Whether to use the debiased Sinkhorn divergence.
        eps_num: Small constant for numerical safety in energy normalisation.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        blur: float = 0.1,
        p: int = 2,
        scaling: float = 0.5,
        debias: bool = True,
        eps_num: float = 1e-8,
    ):
        super().__init__()
        self.blur = blur
        self.p = p
        self.scaling = scaling
        self.debias = debias
        self.eps_num = eps_num
        self.weights = {k: w / sum(weights.values()) for k, w in weights.items()} if weights else None

        self.loss_fn = geomloss.SamplesLoss(
            loss="sinkhorn",
            backend="tensorized",
            p=self.p,
            blur=self.blur,
            scaling=self.scaling,
            debias=self.debias,
            # diameter=100.0,
            cost=self._cost_fn,
            verbose=True,
        )

    def _cost_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Unpack geomloss point clouds and delegate to vertex_cost_matrix.

        Args:
            x: predicted point cloud (..., N_pred, 5) = [pos(3), time(1), energy(1)]
            y: truth point cloud (..., N_truth, 5)
        Returns:
            C: cost matrix (..., N_pred, N_truth)
        """
        predict = {"position": x[..., :3], "time": x[..., 3], "energy": x[..., 4]}
        truth = {"position": y[..., :3], "time": y[..., 3], "energy": y[..., 4]}
        return vertex_cost_matrix(predict, truth, weights=self.weights, eps_num=self.eps_num)

    def forward(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Pack [pos, time, energy] unscaled — cost function handles all weighting
        truth["energy"] = truth["exists"] * truth["energy"]

        def _cat(d):
            return torch.cat([d["position"], d["time"].unsqueeze(-1), d["energy"].unsqueeze(-1)], dim=-1)

        pred_pts = _cat(predict)
        truth_pts = _cat(truth)

        n_pred = pred_pts.shape[-2]
        n_truth = truth_pts.shape[-2]
        d = pred_pts.shape[-1]

        pred_flat = pred_pts.reshape(-1, n_pred, d)
        truth_flat = truth_pts.reshape(-1, n_truth, d)
        batch = pred_flat.shape[0]

        mu = torch.full((batch, n_pred), 1.0 / n_pred, device=pred_pts.device, dtype=pred_pts.dtype)

        zero_energy = truth["energy"] == 0
        first_not_exists = torch.argmax(zero_energy.float(), dim=-1, keepdim=True)
        zero_energy_counts = zero_energy.sum(-1, keepdim=True)

        nu = torch.full((batch, n_truth), 1.0, device=truth_pts.device, dtype=truth_pts.dtype)
        nu[zero_energy] = 0
        nu = nu.scatter(-1, first_not_exists, zero_energy_counts.float())
        nu = nu / (nu.sum(-1, keepdim=True) + self.eps_num)

        loss = self.loss_fn(mu, pred_flat, nu, truth_flat)

        return loss.mean()


class GeomlossSinkhornPositionLoss(nn.Module):
    """Sinkhorn OT loss over predicted vs. truth positions only.

    Unlike the vertex losses this is not a one-to-one assignment: the truth marginal is
    uniform over existing truth vertices (``truth["exists"] == True``) and the predicted
    marginal is uniform over all predicted slots, so the two point clouds may have
    different sizes.

    Args:
        blur: Entropic regularisation strength (geomloss ``blur`` parameter).
        p: Cost exponent (default 2 for squared Euclidean distance).
        scaling: geomloss multi-scale refinement parameter (range (0, 1)).
        debias: Whether to use the debiased Sinkhorn divergence.
        eps_num: Small constant for numerical safety in marginal normalisation.
    """

    def __init__(
        self,
        blur: float = 0.1,
        p: int = 2,
        scaling: float = 0.5,
        debias: bool = True,
        eps_num: float = 1e-8,
    ):
        super().__init__()
        self.blur = blur
        self.p = p
        self.scaling = scaling
        self.debias = debias
        self.eps_num = eps_num

    def forward(
        self,
        predict: dict[str, torch.Tensor],
        truth: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss_fn = geomloss.SamplesLoss(
            loss="sinkhorn", p=self.p, blur=self.blur, scaling=self.scaling, debias=self.debias
        )

        pred_pos = predict["position"]  # (..., N_pred, 3)
        truth_pos = truth["position"]  # (..., N_truth, 3)
        truth_exists = truth["exists"]  # (..., N_truth) bool

        n_pred = pred_pos.shape[-2]
        n_truth = truth_pos.shape[-2]

        pred_flat = pred_pos.reshape(-1, n_pred, 3)
        truth_flat = truth_pos.reshape(-1, n_truth, 3)
        batch = pred_flat.shape[0]

        # Uniform weight over all predicted slots
        mu = torch.full((batch, n_pred), 1.0 / n_pred, device=pred_pos.device, dtype=pred_pos.dtype)

        # Uniform weight over existing truth vertices only; zero elsewhere
        nu = truth_exists.reshape(batch, n_truth).float()
        nu = nu / (nu.sum(-1, keepdim=True) + self.eps_num)

        return loss_fn(mu, pred_flat, nu, truth_flat).mean()


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
