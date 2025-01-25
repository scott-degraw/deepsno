from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.utils.train import copy_if_tensor


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
            self["position_rmsd"] = np.sqrt(np.sum(np.square(position_rmsds))).item()


@torch.compile()
class HitTimeAutoEncoder(nn.Module):
    def __init__(
        self,
        position_reconstructor: nn.Module,
        n_pmts: int,
        norm_dict: dict | None = None,
        positions: tuple = ["x", "y", "z"],
    ):
        super().__init__()

        self.add_module("position_reconstructor", position_reconstructor)

        if norm_dict is not None:
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

        c_eff = 3e2 * self.hit_time_rmsd / self.position_rmsd
        self.register_parameter("effective_c", nn.Parameter(c_eff))
        self.register_parameter("cable_delays", nn.Parameter(torch.zeros(n_pmts)))

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.output_normalize(positions)

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.output_unnormalize(positions)

    def input_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.input_normalize(hit_times)

    def output_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.hit_time_rmsd + self.hit_time_mean

    def output_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.input_normalize(hit_times)

    def forward(
        self, uncal_hit_times: torch.FloatTensor, pmt_ids: torch.IntTensor, pmt_positions: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.position_reconstructor.input_norm = self.input_norm

        pmt_positions = self.position_normalize(pmt_positions)

        predict_positions = self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)

        # Dims: (batch_size, context_window, ...)

        not_padding_masks = pmt_ids != 0

        # Masked pmt positions have positions of zero
        uncal_times = torch.linalg.vector_norm(predict_positions[..., None, :] - pmt_positions, dim=-1)

        uncal_times = uncal_times / self.effective_c
        uncal_times = uncal_times + self.cable_delays[pmt_ids]

        if self.output_unnorm:
            uncal_times = self.output_unnormalize(uncal_times)

        uncal_times = not_padding_masks * uncal_times
        return {"predict": uncal_times, "pad_masks": ~not_padding_masks}


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
    ) -> torch.FloatTensor:
        self.position_reconstructor.input_norm = self.input_norm
        self.position_reconstructor.output_unnorm = self.output_unnorm
        return self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)
