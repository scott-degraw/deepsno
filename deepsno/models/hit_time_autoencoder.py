import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from deepsno.data.inter_pts_bins import inter_pts_bins
from deepsno.utils.config_parse import instantiate
from deepsno.utils.train import copy_if_tensor


def exp_time_walk(
    q: torch.FloatTensor, a: torch.FloatTensor, b: torch.FloatTensor, c: torch.FloatTensor, d: torch.FloatTensor
) -> torch.FloatTensor:
    return a * torch.exp(-q / b) + c * q + d


class ExpTimeWalk(nn.Module):
    def __init__(
        self,
        n_pmts: int,
        a_init: float = 0.1,
        b_init: float = 0.1,
        c_init: float = -0.01,
        d_init: float = 0.0,
        b_beta: float = 10,
        c_beta: float = 10,
        enable_grad_term: bool = False,
    ):
        super().__init__()

        if b_init <= 0:
            raise ValueError("'b_init' must be positive")
        if c_init > 0:
            raise ValueError("'c_init' must be non-positive")
        self.b_beta = b_beta / b_init
        self.c_beta = -c_beta / c_init
        b_init = np.log(np.exp(self.b_beta * b_init) - 1) / self.b_beta
        c_init = np.log(np.exp(self.c_beta * -c_init) - 1) / self.c_beta

        self.a = nn.Parameter(torch.full((n_pmts,), a_init))
        self._b_base = nn.Parameter(torch.full((n_pmts,), b_init))
        self.enable_grad_term = enable_grad_term
        if self.enable_grad_term:
            self._c_base = nn.Parameter(torch.full((n_pmts,), c_init))
        else:
            self.register_buffer("_c_base", copy_if_tensor(torch.full((n_pmts,), 0.0)))
        self.d = nn.Parameter(torch.full((n_pmts,), d_init))

    @property
    def b(self):
        return F.softplus(self._b_base, beta=self.b_beta, threshold=20.0)

    @property
    def c(self):
        if self.enable_grad_term:
            return -F.softplus(self._c_base, beta=self.c_beta, threshold=20.0)
        else:
            return self._c_base

    def forward(self, pmt_ids: torch.LongTensor, qhs: torch.FloatTensor) -> torch.Tensor:
        return exp_time_walk(q=qhs, a=self.a[pmt_ids], b=self.b[pmt_ids], c=self.c[pmt_ids], d=self.d[pmt_ids])


class CableDelayTimeWalk(nn.Module):
    def __init__(self, n_pmts: int, delay_init: float = 0.0):
        super().__init__()
        self.register_parameter("cable_delays", nn.Parameter(torch.full((n_pmts,), delay_init)))

    def forward(self, pmt_ids: torch.LongTensor) -> torch.Tensor:
        return self.cable_delays[pmt_ids]


def linear_interp(
    ids: torch.LongTensor, x: torch.FloatTensor, xp: torch.FloatTensor, fp: torch.FloatTensor, epsilon: float = 1e-6
) -> torch.FloatTensor:
    # ids, x: (batch_size, context_window)
    # xp, fp: (n_pmts, n_points)

    # assert ids.shape == x.shape

    xp_by_id = xp[ids]
    fp_by_id = fp[ids]
    indices = torch.searchsorted(xp_by_id, x.unsqueeze(-1), side="right").squeeze(-1)
    # The clamp insures that in the extrapolation case we use the two closest points
    right_indices = torch.clamp(indices, min=1, max=xp.shape[1] - 1).unsqueeze(-1)
    left_indices = right_indices - 1
    left_x = torch.gather(xp_by_id, dim=-1, index=left_indices).squeeze()
    left_y = torch.gather(fp_by_id, dim=-1, index=left_indices).squeeze()
    right_x = torch.gather(xp_by_id, dim=-1, index=right_indices).squeeze()
    right_y = torch.gather(fp_by_id, dim=-1, index=right_indices).squeeze()

    slope = (right_y - left_y) / (right_x - left_x + epsilon)

    interp = left_y + slope * (x - left_x)

    return interp


class InterPtsTimeWalk(nn.Module):
    def __init__(self, qhs_hist: str | Path, qhs_scale: float, min_occupancy: int):
        super().__init__()

        with open(qhs_hist, "rb") as f:
            pmt_qhs_hists = pickle.load(f)

        edges = inter_pts_bins(pmt_qhs_hists, min_occupancy=min_occupancy)
        edges /= qhs_scale
        centers = torch.from_numpy(0.5 * (edges[:, 1:-2] + edges[:, 2:-1])).float()
        self.register_buffer("centers", copy_if_tensor(centers))
        self.register_buffer("min_qhs", copy_if_tensor(torch.from_numpy(edges[:, 0]).float()))
        self.register_buffer("max_qhs", copy_if_tensor(torch.from_numpy(edges[:, -1]).float()))

        self.times = nn.Parameter(torch.zeros_like(centers))
        self.high_m = nn.Parameter(torch.zeros(self.centers.shape[0]).float())
        self.high_b = nn.Parameter(torch.zeros(self.centers.shape[0]).float())

    def forward(self, pmt_ids: torch.LongTensor, qhs: torch.FloatTensor) -> torch.FloatTensor:
        # qhs: (batch_size, context_window)
        # min_qhs, max_qhs: (n_pmts,)

        interp = linear_interp(ids=pmt_ids, x=qhs, xp=self.centers, fp=self.times)
        interp_qhs = (qhs >= self.min_qhs[pmt_ids]) | (qhs <= self.max_qhs[pmt_ids])
        pmt_ids = interp_qhs * pmt_ids
        interp = interp_qhs * interp

        straight_line_fit_qhs = qhs > self.max_qhs[pmt_ids]
        interp = interp + straight_line_fit_qhs * self.high_m[pmt_ids] * qhs + self.high_b[pmt_ids]

        return interp


@torch.compile(dynamic=False, fullgraph=True)
class HitTimeAutoEncoder(nn.Module):
    def __init__(
        self,
        position_reconstructor: nn.Module,
        time_walk: nn.Module,
        c_av: float,
        c_av_grad: float,
        c_water: float,
        av_radius: float,
        pmt_valid_file: str | Path,
        all_pass: int = 0x0,
        all_fail: int = 0xFFFFFFFF,
        fix_c: bool = False,
        norm_dict: dict | None = None,
        pos_recon_gets_cal_time: bool = True,
    ):
        super().__init__()
        torch.set_float32_matmul_precision("high")

        self.add_module("position_reconstructor", position_reconstructor)
        self.add_module("time_walk", time_walk)

        if norm_dict is not None:
            self.register_buffer("time_scale", copy_if_tensor(norm_dict["time_scale"]))
            self.register_buffer("position_scale", copy_if_tensor(norm_dict["position_scale"]))

            self.position_reconstructor.add_input_norm(
                time_shift=0,
                time_scale=self.time_scale,
                input_norm=True,
            )
            self.position_reconstructor.add_output_unnorm(
                position_shifts=3 * [0.0],
                position_scales=tuple(self.position_scale.repeat(3)),
                output_unnorm=False,
            )

            if "qhs_scale" in norm_dict:
                self.register_buffer("qhs_scale", copy_if_tensor(norm_dict["qhs_scale"]))

            self.input_norm = True
            self.output_unnorm = False
        else:
            self.input_norm = False
            self.output_unnorm = False

        self.register_parameter("c_av", nn.Parameter(c_av * self.time_scale / self.position_scale))
        self.c_av.requires_grad = not fix_c
        self.register_parameter("c_water", nn.Parameter(c_water * self.time_scale / self.position_scale))
        self.c_water.requires_grad = not fix_c

        self.c_av_gradient = nn.Parameter(c_av_grad / self.time_scale)
        self.c_av_gradient.requires_grad = not fix_c

        self.register_buffer("av_radius", copy_if_tensor(torch.tensor(av_radius) / self.position_scale))

        self.all_pass = all_pass
        self.all_fail = all_fail

        valid = np.loadtxt(pmt_valid_file, dtype=np.uint32).astype(np.bool)
        status = np.where(valid, all_pass, all_fail)
        self.register_buffer("status", copy_if_tensor(status))
        self.register_buffer("pmt_valid", copy_if_tensor(valid))

        self.pos_recon_gets_cal_time = pos_recon_gets_cal_time

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_normalize(positions)

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_unnormalize(positions)

    def time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times / self.time_scale

    def time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.time_scale

    def qhs_normalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return qhs / self.qhs_scale

    def qhs_unnormalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return qhs * self.qhs_scale

    def flight_time(
        self,
        event_positions: torch.FloatTensor,
        pmt_positions: torch.FloatTensor,
        av_offset: torch.FloatTensor,
        epsilon: float = 1e-9,
    ) -> torch.FloatTensor:
        event_positions = event_positions[..., None, :]

        event_2_pmt_vec = pmt_positions - event_positions

        # put event positions in terms of av coordinates
        event_positions = event_positions - av_offset[..., None, :]

        dist_event_2_pmt = torch.linalg.vector_norm(event_2_pmt_vec, dim=-1)
        norm_event_2_pmt_vec = event_2_pmt_vec / dist_event_2_pmt[..., None]

        # Use line sphere intersection calculations (https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection)
        event_pos_projection = torch.sum(norm_event_2_pmt_vec * event_positions, dim=-1)
        discriminant = event_pos_projection.square() - event_positions.square().sum(-1) + self.av_radius.square()

        line_passes_av = discriminant > 0
        event_radius = torch.linalg.vector_norm(event_positions, dim=-1)
        event_inside_av = line_passes_av * (event_radius < self.av_radius)
        event_outside_av = line_passes_av * (event_radius >= self.av_radius)

        # If discriminant is negative then straight line path does not intersect AV
        # In this case dist_av = 0
        sqrt_discriminant = torch.sqrt(line_passes_av * discriminant + epsilon)

        dist_av = torch.zeros(pmt_positions.shape[:-1], device=pmt_positions.device)
        dist_av = (
            dist_av
            + event_inside_av * (-event_pos_projection + sqrt_discriminant)
            + event_outside_av * (2 * sqrt_discriminant)
        )

        dist_water = dist_event_2_pmt - dist_av

        return dist_av / (self.c_av + self.c_av_gradient * dist_av) + dist_water / self.c_water

    def forward(
        self,
        uncal_hit_times: torch.FloatTensor,
        pmt_ids: torch.IntTensor,
        pmt_positions: torch.FloatTensor,
        av_offset: torch.FloatTensor,
        qhs: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        pmt_ids *= self.pmt_valid[pmt_ids]
        self.position_reconstructor.input_norm = False
        self.position_reconstructor.output_unnorm = False

        pmt_mask = pmt_ids == 0
        not_padding_masks = ~pmt_mask
        nhits = torch.sum(not_padding_masks, dim=-1, keepdim=True).float()

        pmt_positions = self.position_normalize(pmt_positions)
        av_offset = self.position_normalize(av_offset)
        uncal_hit_times = self.time_normalize(uncal_hit_times)
        uncal_hit_times[pmt_mask] = 0
        # I believe this is redundant since we subtract the mean below but I still want to include it
        uncal_hit_times = uncal_hit_times - torch.sum(uncal_hit_times, dim=-1, keepdim=True) / nhits

        if qhs is None:
            time_walk = self.time_walk(pmt_ids=pmt_ids)
        else:
            qhs = self.qhs_normalize(qhs)
            time_walk = self.time_walk(pmt_ids=pmt_ids, qhs=qhs)

        cal_hit_times = uncal_hit_times - time_walk
        cal_hit_times[pmt_mask] = 0
        cal_hit_times = cal_hit_times - torch.sum(cal_hit_times, dim=-1, keepdim=True) / nhits

        if self.pos_recon_gets_cal_time:
            predict = self.position_reconstructor(hit_times=cal_hit_times, pmt_ids=pmt_ids)
        else:
            predict = self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)

        # Masked pmt positions have positions of zero
        times_of_flight = self.flight_time(
            event_positions=predict["positions"],
            pmt_positions=pmt_positions,
            av_offset=av_offset,
        )
        time_residuals = not_padding_masks * (cal_hit_times - times_of_flight - predict["times"].unsqueeze(-1))

        out = {
            "time_residuals": time_residuals,
            "pad_masks": ~not_padding_masks,
            "positions": predict["positions"],
            "times": predict["times"],
        }

        if self.output_unnorm:
            out["time_residuals"] = self.time_unnormalize(out["time_residuals"])
            out["positions"] = self.position_unnormalize(out["positions"])
            if "times" in predict:
                out["times"] = self.time_unnormalize(out["times"])

        out["c_av"] = self.c_av
        out["c_water"] = self.c_water

        return out

    @staticmethod
    def time_walk_from_ckpt(ckpt: str | Path) -> dict:
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise ValueError(f"Checkpoint '{ckpt}' does not exist")

        with open(ckpt.parent.parent / "config.yaml") as f:
            config = yaml.safe_load(f)

        ckpt = torch.load(ckpt, weights_only=True, map_location="cpu")

        model = instantiate(config["model"])
        model.load_state_dict(ckpt["model"], strict=True)

        time_walk = model.time_walk

        time_walk_params = {}
        status = ~model.status.detach().numpy().astype(bool)
        time_walk_params["intercept"] = (model.time_scale * time_walk.d).detach().numpy()
        time_walk_params["intercept"] -= np.median(time_walk_params["intercept"][status])
        time_walk_params["gradient"] = (model.time_scale / model.qhs_scale * time_walk.c).detach().numpy()
        time_walk_params["qhs_scale"] = (model.qhs_scale * time_walk.b).detach().numpy()
        time_walk_params["time_scale"] = (model.time_scale * time_walk.a).detach().numpy()
        time_walk_params["status"] = model.status.detach().numpy().astype(np.uint32)

        return time_walk_params
