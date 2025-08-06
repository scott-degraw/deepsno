from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from deepsno.utils.config_parse import instantiate
from deepsno.utils.train import copy_if_tensor, get_best_ckpt


class HitTimeAutoEncoderNorm(dict):
    def __init__(self, train_file: str | Path, positions: Tuple = ["x", "y", "z"]):
        super().__init__()

        with h5py.File(train_file) as h5_file:
            self["hit_time_mean"] = h5_file["cal_pmt_events/hit_times"].attrs["mean"].item()
            self["hit_time_rmsd"] = h5_file["cal_pmt_events/hit_times"].attrs["root_mean_square_deviation"].item()
            position_means = [h5_file[f"mc_truth/position/{c}"].attrs["mean"].item() for c in positions]
            self["position_mean"] = np.mean(position_means).item()
            position_rmsds = np.array(
                [h5_file[f"mc_truth/position/{c}"].attrs["root_mean_square_deviation"].item() for c in positions]
            )
            self["position_rmsd"] = np.sqrt(np.mean(np.square(position_rmsds))).item()


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
        return self.a[pmt_ids] * torch.exp(-qhs / self.b[pmt_ids]) + self.c[pmt_ids] * qhs + self.d[pmt_ids]


class CableDelayTimeWalk(nn.Module):
    def __init__(self, n_pmts: int, delay_init: float = 0.0):
        super().__init__()
        self.register_parameter("cable_delays", nn.Parameter(torch.full((n_pmts,), delay_init)))

    def forward(self, pmt_ids: torch.LongTensor) -> torch.Tensor:
        return self.cable_delays[pmt_ids]


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
        dset: str | Path,
        fix_c: bool = False,
        position_reconstructor_state_dict_path: str | Path | None = None,
        norm_dict: dict | None = None,
    ):
        super().__init__()

        self.add_module("position_reconstructor", position_reconstructor)
        self.add_module("time_walk", time_walk)

        if position_reconstructor_state_dict_path is not None:
            state_dict_path = Path(position_reconstructor_state_dict_path)
            if state_dict_path.is_dir():
                state_dict_path = get_best_ckpt(state_dict_path)
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
            self.position_reconstructor.load_state_dict(state_dict["model"], strict=True)

            self.position_reconstructor.input_norm = True
            self.position_reconstructor.output_unnorm = False

            hit_time_mean = self.position_reconstructor.hit_time_mean
            hit_time_rmsd = self.position_reconstructor.hit_time_rmsd
            position_mean = self.position_reconstructor.position_means.mean()

            position_rmsd = self.position_reconstructor.position_rmsds.square().mean().sqrt()

            self.register_buffer("hit_time_mean", copy_if_tensor(hit_time_mean))
            self.register_buffer("hit_time_rmsd", copy_if_tensor(hit_time_rmsd))
            self.register_buffer("position_mean", copy_if_tensor(position_mean))
            self.register_buffer("position_rmsd", copy_if_tensor(position_rmsd))

            self.input_norm = True
            self.output_unnorm = False
        elif norm_dict is not None:
            self.register_buffer("hit_time_mean", copy_if_tensor(norm_dict["hit_time_mean"]))
            self.register_buffer("hit_time_rmsd", copy_if_tensor(norm_dict["hit_time_rmsd"]))
            self.register_buffer("position_mean", copy_if_tensor(norm_dict["position_mean"]))
            self.register_buffer("position_rmsd", copy_if_tensor(norm_dict["position_rmsd"]))

            self.position_reconstructor.add_input_norm(
                hit_time_mean=self.hit_time_mean,
                hit_time_rmsd=self.hit_time_rmsd,
                input_norm=True,
            )
            self.position_reconstructor.add_output_unnorm(
                position_means=self.position_mean.repeat(3),
                position_rmsds=self.position_rmsd.repeat(3),
                output_unnorm=False,
            )

            if "qhs_mean" in norm_dict:
                self.register_buffer("qhs_mean", copy_if_tensor(norm_dict["qhs_mean"]))
                self.register_buffer("qhs_rmsd", copy_if_tensor(norm_dict["qhs_rmsd"]))

            self.input_norm = True
            self.output_unnorm = False
        else:
            self.input_norm = False
            self.output_unnorm = False

        self.register_parameter("c_av", nn.Parameter(c_av * self.hit_time_rmsd / self.position_rmsd))
        self.c_av.requires_grad = not fix_c
        self.register_parameter("c_water", nn.Parameter(c_water * self.hit_time_rmsd / self.position_rmsd))
        self.c_av.requires_grad = not fix_c
        self.c_water.requires_grad = not fix_c

        self.c_av_gradient = nn.Parameter(c_av_grad / self.hit_time_rmsd)
        self.c_av_gradient.requires_grad = not fix_c

        self.register_buffer("av_radius", copy_if_tensor(torch.tensor(av_radius) / self.position_rmsd))

        with h5py.File(dset, "r") as h5_file:
            status = h5_file["pmt_info/status"][:].astype(np.int32)
            self.register_buffer("status", copy_if_tensor(status))
            self.register_buffer("min_run", torch.tensor(h5_file.attrs["min_run"], dtype=torch.int64))
            self.register_buffer("max_run", torch.tensor(h5_file.attrs["max_run"], dtype=torch.int64))

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_normalize(positions)

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_unnormalize(positions)

    def hit_time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.hit_time_mean) / self.hit_time_rmsd

    def hit_time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.hit_time_rmsd + self.hit_time_mean

    def qhs_normalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return (qhs - self.qhs_mean) / self.qhs_rmsd

    def qhs_unnormalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return qhs * self.qhs_rmsd + self.qhs_mean

    def output_unnormalize(self, x: dict) -> dict:
        return {
            "uncal_hit_times": self.hit_time_unnormalize(x["uncal_hit_times"]),
            "positions": self.position_unnormalize(x["positions"]),
        }

    def output_normalize(self, x: dict) -> dict:
        return {
            "uncal_hit_times": self.hit_time_normalize(x["uncal_hit_times"]),
            "positions": self.position_normalize(x["positions"]),
        }

    def flight_time(
        self,
        event_positions: torch.FloatTensor,
        pmt_positions: torch.FloatTensor,
        av_offset: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        event_positions = event_positions[..., None, :]

        event_2_pmt_vec = pmt_positions - event_positions

        # put event positions in terms of av coordinates
        if av_offset is not None:
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
        av_offset: torch.FloatTensor | None = None,
        qhs: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        self.position_reconstructor.input_norm = self.input_norm

        pmt_positions = self.position_normalize(pmt_positions)
        if av_offset is not None:
            av_offset = self.position_normalize(av_offset)

        if qhs is not None:
            cal_hit_times = uncal_hit_times - self.time_walk(pmt_ids=pmt_ids, qhs=qhs)
            predict = self.position_reconstructor(hit_times=cal_hit_times, pmt_ids=pmt_ids)
        else:
            predict = self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)

        predict_positions = predict["positions"]
        if "times" in predict:
            predict_times = predict["times"]

        # Dims: (batch_size, context_window, ...)

        not_padding_masks = pmt_ids != 0

        # Masked pmt positions have positions of zero
        times_of_flight = self.flight_time(
            event_positions=predict_positions,
            pmt_positions=pmt_positions,
            av_offset=av_offset,
        )
        if qhs is None:
            times_of_flight = times_of_flight + self.time_walk(pmt_ids=pmt_ids)
        else:
            qhs = self.qhs_normalize(qhs)
            times_of_flight = times_of_flight + self.time_walk(pmt_ids=pmt_ids, qhs=qhs)

        if "times" in predict:
            times_of_flight = times_of_flight + predict_times.unsqueeze(-1)

        times_of_flight = not_padding_masks * times_of_flight

        out = {"times_of_flight": times_of_flight, "pad_masks": ~not_padding_masks, "positions": predict_positions}

        if "times" in predict:
            out["times"] = predict_times

        if self.output_unnorm:
            out["times_of_flight"] = self.hit_time_unnormalize(out["times_of_flight"])
            out["positions"] = self.position_unnormalize(out["positions"])
            if "times" in predict:
                out["times"] = self.hit_time_unnormalize(out["times"])

        out["c_av"] = self.c_av
        out["c_water"] = self.c_water

        return out

    @staticmethod
    def time_walk_from_ckpt(ckpt: str | Path) -> dict:
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise ValueError(f"Checkpjkoint '{ckpt}' does not exist")

        with open(ckpt.parent.parent / "config.yaml") as f:
            config = yaml.safe_load(f)

        ckpt = torch.load(ckpt, weights_only=True, map_location="cpu")

        model = instantiate(config["model"])
        model.load_state_dict(ckpt["model"], strict=True)

        time_walk = model.time_walk

        time_walk_params = {}
        time_walk_params["intercept"] = (model.hit_time_rmsd * time_walk.d).detach().numpy()
        time_walk_params["intercept"] -= np.median(time_walk_params["intercept"])
        time_walk_params["gradient"] = (model.hit_time_rmsd / model.qhs_rmsd * time_walk.c).detach().numpy()
        time_walk_params["qhs_scale"] = (model.qhs_rmsd * time_walk.b).detach().numpy()
        time_walk_params["time_scale"] = (model.hit_time_rmsd * time_walk.a).detach().numpy()
        time_walk_params["status"] = model.status.detach().numpy().astype(np.uint32)
        time_walk_params["min_run"] = model.min_run.item()
        time_walk_params["max_run"] = model.max_run.item()

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
