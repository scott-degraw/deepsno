from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.utils.train import copy_if_tensor, get_best_ckpt


class HitTimeAutoEncoderNorm(dict):
    def __init__(self, train_file: str | Path, positions: tuple = ["x", "y", "z"]):
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


class HitTimeAutoEncoder(nn.Module):
    def __init__(
        self,
        position_reconstructor: nn.Module,
        n_pmts: int,
        effective_c: float = 100,
        fix_effective_c: bool = False,
        position_reconstructor_state_dict_path: str | Path | None = None,
        norm_dict: dict | None = None,
        positions: tuple = ["x", "y", "z"],
    ):
        super().__init__()

        self.add_module("position_reconstructor", position_reconstructor)

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

            self.input_norm = True
            self.output_unnorm = False
        else:
            self.input_norm = False
            self.output_unnorm = False

        c_eff = effective_c * self.hit_time_rmsd / self.position_rmsd
        self.register_parameter("effective_c", nn.Parameter(c_eff))
        self.effective_c.requires_grad = not fix_effective_c
        self.register_parameter("cable_delays", nn.Parameter(torch.zeros(n_pmts)))

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_normalize(positions)

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.position_unnormalize(positions)

    def hit_time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.hit_time_mean) / self.hit_time_rmsd

    def hit_time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.hit_time_rmsd + self.hit_time_mean

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

    def forward(
        self, uncal_hit_times: torch.FloatTensor, pmt_ids: torch.IntTensor, pmt_positions: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.position_reconstructor.input_norm = self.input_norm

        pmt_positions = self.position_normalize(pmt_positions)

        predict = self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)
        predict_positions = predict["positions"]
        if "times" in predict:
            predict_times = predict["times"]

        # Dims: (batch_size, context_window, ...)

        not_padding_masks = pmt_ids != 0

        # Masked pmt positions have positions of zero
        times_of_flight = torch.linalg.vector_norm(predict_positions[..., None, :] - pmt_positions, dim=-1)

        times_of_flight = times_of_flight / self.effective_c
        times_of_flight = times_of_flight + self.cable_delays[pmt_ids]
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

        return out


class CableDelayFineTune(HitTimeAutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in super().parameters():
            param.requires_grad = False

        self.fine_tune = True
        for param in self.parameters():
            param.requires_grad = True

    def parameters(self, *args, **kwargs):
        if self.fine_tune:
            return [self.get_parameter("effective_c"), self.get_parameter("cable_delays")]
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
