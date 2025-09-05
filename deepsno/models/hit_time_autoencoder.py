from pathlib import Path
from typing import Tuple

import uproot as ur
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from deepsno.utils.config_parse import instantiate
from deepsno.utils.train import copy_if_tensor, get_best_ckpt



def exp_time_walk(
    q: torch.FloatTensor, a: torch.FloatTensor, b: torch.FloatTensor, c: torch.FloatTensor, d: torch.FloatTensor
) -> torch.FloatTensor:
    return a * torch.exp(-q / b) + c * q + d


class ExpTimeWalk(nn.Module):
    def __init__(self, n_pmts: int, a_init: float = 0.1, b_init: float = 0.1, c_init: float = 0.0, d_init: float = 0.0):
        super().__init__()

        assert b_init > 0, "'b_init' must be positive"
        assert c_init < 0, "'c_init' must be negative"
        b_init = np.log(np.exp(b_init) - 1)
        c_init = np.log(np.exp(-c_init) - 1)

        self.a = nn.Parameter(torch.full((n_pmts,), a_init))
        self.b_base = nn.Parameter(torch.full((n_pmts,), b_init))
        self.c_base = nn.Parameter(torch.full((n_pmts,), c_init))
        self.d = nn.Parameter(torch.full((n_pmts,), d_init))

    @property
    def b(self):
        return F.softplus(self.b_base, beta=1.0, threshold=20.0)

    @property
    def c(self):
        return -F.softplus(self.c_base, beta=1.0, threshold=20.0)

    def forward(self, pmt_ids: torch.LongTensor, qhs: torch.FloatTensor) -> torch.Tensor:
        return exp_time_walk(qhs, self.a[pmt_ids], self.b[pmt_ids], self.c[pmt_ids], self.d[pmt_ids])


class CableDelayTimeWalk(nn.Module):
    def __init__(self, n_pmts: int, delay_init: float = 0.0):
        super().__init__()
        self.register_parameter("cable_delays", nn.Parameter(torch.full((n_pmts,), delay_init)))

    def forward(self, pmt_ids: torch.LongTensor) -> torch.Tensor:
        return self.cable_delays[pmt_ids]


# @torch.compile(dynamic=False, fullgraph=True)
class HitTimeAutoEncoder(nn.Module):
    def __init__(
        self,
        position_reconstructor: nn.Module,
        time_walk: nn.Module,
        c_av: float,
        c_av_grad: float,
        c_water: float,
        av_radius: float,
        dset: str | Path,
        fix_c: bool = False,
        norm_dict: dict | None = None,
    ):
        super().__init__()

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

        with ur.open({dset: "transpose"}) as transpose:
            status = transpose["status"].array(library="np").astype(np.int32)
            self.register_buffer("status", copy_if_tensor(status))

        with ur.open({dset: "metadata"}) as metadata:
            self.register_buffer("run_range", copy_if_tensor(metadata["run_range"].array(library="np")[0]))

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
        #TODO: This may be causing problems with gradients when discriminant is very close to zero  
        sqrt_discriminant = torch.sqrt(nn.functional.relu(discriminant))

        dist_av = torch.zeros(pmt_positions.shape[:-1], device=pmt_positions.device)
        dist_av = dist_av + torch.where(event_inside_av, -event_pos_projection + sqrt_discriminant, 0)
        dist_av = dist_av + torch.where(event_outside_av, 2 * sqrt_discriminant, 0)

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
        self.position_reconstructor.input_norm = self.input_norm

        pmt_positions = self.position_normalize(pmt_positions)
        av_offset = self.position_normalize(av_offset)
        uncal_hit_times = self.time_normalize(uncal_hit_times)

        if qhs is None:
            time_walk = self.time_walk(pmt_ids=pmt_ids)
        else:
            qhs = self.qhs_normalize(qhs)
            time_walk = self.time_walk(pmt_ids=pmt_ids, qhs=qhs)

        cal_hit_times = uncal_hit_times - time_walk

        predict = self.position_reconstructor(hit_times=cal_hit_times, pmt_ids=pmt_ids)

        # Dims: (batch_size, context_window, ...)

        not_padding_masks = pmt_ids != 0

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
        time_walk_params["run_range"] = model.run_range.tolist()

        return time_walk_params


class CableDelayFineTune(HitTimeAutoEncoder):
    def __init__(self, *args, fix_effective_c: bool = False, **kwargs):
        super().__init__(*args, fix_effective_c=fix_effective_c, **kwargs)

        for param in super().parameters():
            param.requires_grad = False

        self.fixed_parameters = ["cable_delays"]
        if not fix_effective_c:
            self.fixed_parameters.append("effective_c")

        self.fine_tune = True
        for param in self.parameters():
            param.requires_grad = True

    def parameters(self, *args, **kwargs):
        if self.fine_tune:
            return [self.get_parameter(param) for param in self.fixed_parameters]
        return super().parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        self.fine_tune = False
        state_dict = super().state_dict(*args, **kwargs)
        self.fine_tune = True
        return state_dict


class PositionRecoFromHitTimeAutoEncoder(HitTimeAutoEncoder):
    # This subclass is used when I want to just look at the predictions from the position reconstructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, uncal_hit_times: torch.FloatTensor, pmt_ids: torch.LongTensor, pmt_positions: torch.FloatTensor
    ) -> dict:
        self.position_reconstructor.input_norm = self.input_norm
        self.position_reconstructor.output_unnorm = self.output_unnorm
        return self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)
