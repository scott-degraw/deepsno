import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import init

from src.utils.utils import copy_if_tensor


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

            self["qhs_mean"] = h5_file["cal_pmt_events/qhs"].attrs["mean"].item()
            self["qhs_rmsd"] = h5_file["cal_pmt_events/qhs"].attrs["root_mean_square_deviation"].item()


class TimeWalkMLP(nn.Module):
    class PerPMTLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            n_pmts: int,
            bias: bool = True,
            dtype: torch.dtype | str = None,
            device: torch.device | str = None,
        ):
            super().__init__()
            factory_kwargs = {"dtype": dtype, "device": device}

            self.in_features = in_features
            self.out_features = out_features
            self.n_pmts = n_pmts
            self.weight = nn.Parameter(torch.empty((n_pmts, out_features, in_features), **factory_kwargs))

            if bias:
                self.bias = nn.Parameter(torch.empty((n_pmts, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)

            self.reset_parameters()

        def reset_parameters(self) -> None:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

        def forward(self, input: torch.FloatTensor, pmt_mask: torch.BoolTensor) -> torch.Tensor:
            # input: (batch..., n_pmts, in_features)
            # pmt_mask: (batch..., n_pmts)
            input = torch.einsum("ijk,...ik->...ij", self.weight, input)
            input = pmt_mask[..., None] * input
            if self.bias is not None:
                input = input + pmt_mask[..., None] * self.bias

            return input

    class MaskedActFn(nn.Module):
        def __init__(self, act_fn: nn.Module):
            super().__init__()
            self.register_module("act_fn", act_fn)

        def forward(self, inputs: torch.Tensor, pmt_mask: torch.BoolTensor) -> torch.Tensor:
            inputs = self.act_fn(inputs)
            return pmt_mask[..., None] * inputs

    def __init__(self, n_pmts: int, act_fn_class: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.n_pmts = n_pmts

        self.layers = nn.ModuleList(
            (
                self.PerPMTLinear(1, 32, n_pmts=n_pmts),
                self.MaskedActFn(act_fn_class()),
                self.PerPMTLinear(32, 32, n_pmts=n_pmts),
                self.MaskedActFn(act_fn_class()),
                self.PerPMTLinear(32, 32, n_pmts=n_pmts),
                self.MaskedActFn(act_fn_class()),
                self.PerPMTLinear(32, 32, n_pmts=n_pmts),
                self.MaskedActFn(act_fn_class()),
                # self.PerPMTLinear(32, 64, n_pmts=n_pmts),
                # self.MaskedActFn(act_fn_class()),
                # self.PerPMTLinear(64, 64, n_pmts=n_pmts),
                # self.MaskedActFn(act_fn_class()),
                # self.PerPMTLinear(64, 32, n_pmts=n_pmts),
                # self.MaskedActFn(act_fn_class()),
                self.PerPMTLinear(32, 1, n_pmts=n_pmts),
            )
        )

    def forward(self, charges: torch.FloatTensor, pmt_ids: torch.LongTensor) -> torch.FloatTensor:
        # charge: (batch..., context_len)
        # pmt_ids: (batch..., context_len)

        factory_kwargs = {"dtype": charges.dtype, "device": charges.device}
        batch_shape = charges.shape[:-1]
        pmt_mask = torch.zeros((*batch_shape, self.n_pmts), dtype=torch.bool, device=factory_kwargs["device"])

        pmt_mask.scatter_(dim=-1, index=pmt_ids, value=True)
        pmt_mask[..., 0] = False  # for the padding pmt with id 0

        charges_per_id = torch.zeros((*batch_shape, self.n_pmts), **factory_kwargs)
        charges_per_id.scatter_(dim=-1, index=pmt_ids, src=charges)
        charges_per_id[..., 0] = 0.0

        charges_per_id.unsqueeze_(-1)
        for layer in self.layers:
            charges_per_id = layer(charges_per_id, pmt_mask)

        charges_per_id.squeeze_()

        times = torch.gather(input=charges_per_id, dim=-1, index=pmt_ids)

        return times


class HitTimeAutoEncoder(nn.Module):
    def __init__(
        self,
        position_reconstructor: nn.Module,
        n_pmts: int,
        norm_dict: dict | None = None,
        positions: tuple = ["x", "y", "z"],
    ):
        super().__init__()

        self.register_parameter("effective_c", nn.Parameter(torch.tensor(1.0)))

        self.add_module("position_reconstructor", position_reconstructor)

        if norm_dict is not None:
            self.register_buffer("hit_time_mean", copy_if_tensor(norm_dict["hit_time_mean"]))
            self.register_buffer("hit_time_rmsd", copy_if_tensor(norm_dict["hit_time_rmsd"]))
            self.register_buffer("position_mean", copy_if_tensor(norm_dict["position_mean"]))
            self.register_buffer("position_rmsd", copy_if_tensor(norm_dict["position_rmsd"]))
            self.register_buffer("qhs_mean", copy_if_tensor(norm_dict["qhs_mean"]))
            self.register_buffer("qhs_rmsd", copy_if_tensor(norm_dict["qhs_rmsd"]))

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

        self.time_walk_mlp = TimeWalkMLP(n_pmts=n_pmts, act_fn_class=nn.ReLU)

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.output_normalize(positions)

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return self.position_reconstructor.output_unnormalize(positions)

    def qhs_normalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return (qhs - self.qhs_mean) / self.qhs_rmsd

    def qhs_unnormalize(self, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return qhs * self.qhs_rmsd + self.qhs_mean

    def hit_time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.hit_time_mean) / self.hit_time_rmsd

    def hit_time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.hit_time_rmsd + self.hit_time_mean

    def input_normalize(self, hit_times: torch.FloatTensor, qhs: torch.FloatTensor) -> torch.FloatTensor:
        return {"hit_times": self.position_reconstructor.input_normalize(hit_times), "qhs": self.qhs_normalize(qhs)}

    def output_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.hit_time_unnormalize(hit_times)

    def output_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.hit_time_normalize(hit_times)

    def forward(
        self,
        uncal_hit_times: torch.FloatTensor,
        qhs: torch.FloatTensor,
        pmt_ids: torch.IntTensor,
        pmt_positions: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pmt_positions = self.position_normalize(pmt_positions)
        if self.input_norm:
            qhs = self.qhs_normalize(qhs)

        predict_positions = self.position_reconstructor(hit_times=uncal_hit_times, pmt_ids=pmt_ids)

        # Dims: (batch_size, context_window, ...)

        not_padding_masks = pmt_ids != 0

        # Masked pmt positions have positions of zero
        uncal_times = torch.linalg.vector_norm(predict_positions[..., None, :] - pmt_positions, dim=-1)
        uncal_times = not_padding_masks * uncal_times

        uncal_times = uncal_times / self.effective_c

        uncal_times = uncal_times + self.time_walk_mlp(charges=qhs, pmt_ids=pmt_ids)

        if self.output_unnorm:
            uncal_times = self.output_unnormalize(uncal_times)

        return {"predict": uncal_times, "pad_masks": ~not_padding_masks}


class PositionRecoFromHitTimeAutoEncoder(HitTimeAutoEncoder):
    # This subclass is used when I want to just look at the predictions from the position reconstructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hit_times: torch.FloatTensor, pmt_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.position_reconstructor(hit_times=hit_times, pmt_ids=pmt_ids)
