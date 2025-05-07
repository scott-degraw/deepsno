from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from deepsno.utils.train import copy_if_tensor


class PositionRecoNorm(dict):
    def __init__(self, train_file: str | Path, positions: tuple = ["x", "y", "z"]):
        super().__init__()

        with h5py.File(train_file) as h5_file:
            self["input_norms"] = {
                "hit_time_mean": h5_file["cal_pmt_events/hit_times"].attrs["mean"].item(),
                "hit_time_rmsd": h5_file["cal_pmt_events/hit_times"].attrs["root_mean_square_deviation"].item(),
            }
            position_means = [h5_file[f"mc_truth/position/{c}"].attrs["mean"].item() for c in positions]
            position_mean = np.mean(position_means).item()
            position_rmsds = np.array(
                [h5_file[f"mc_truth/position/{c}"].attrs["root_mean_square_deviation"].item() for c in positions]
            )
            position_rmsd = np.sqrt(np.mean(np.square(position_rmsds))).item()
            self["output_norms"] = {
                "position_means": 3 * [position_mean],
                "position_rmsds": 3 * [position_rmsd],
            }


class PositionReco(nn.Module):
    def add_input_norm(
        self,
        hit_time_mean: float | torch.FloatTensor,
        hit_time_rmsd: float | torch.FloatTensor,
        input_norm: bool = True,
    ):
        self.register_buffer("hit_time_mean", copy_if_tensor(hit_time_mean))
        self.register_buffer("hit_time_rmsd", copy_if_tensor(hit_time_rmsd))

        self.input_norm = input_norm

    def add_output_unnorm(self, position_means: tuple, position_rmsds: tuple, output_unnorm: bool = True):
        self.register_buffer("position_means", copy_if_tensor(position_means))
        self.register_buffer("position_rmsds", copy_if_tensor(position_rmsds))

        self.output_unnorm = output_unnorm

    def hit_time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.hit_time_mean) / self.hit_time_rmsd

    def hit_time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.hit_time_rmsd + self.hit_time_mean

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return (positions - self.position_means) / self.position_rmsds

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return positions * self.position_rmsds + self.position_means

    def input_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.hit_time_normalize(hit_times)

    def output_unnormalize(self, predict: torch.FloatTensor) -> dict:
        out = {"positions": self.position_unnormalize(predict["positions"])}
        if "times" in predict:
            out["times"] = self.hit_time_unnormalize(predict["times"])
        return out

    def output_normalize(self, predict: torch.FloatTensor) -> dict:
        out = {"positions": self.position_normalize(predict["positions"])}
        if "times" in predict:
            out["times"] = self.hit_time_normalize(predict["times"])
        return out

    def __init__(
        self,
        n_pmts: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        hit_time_embedding_dim: int,
        predict_time: bool = False,
        norm_dict: dict | None = None,
    ):
        super().__init__()
        self.n_pmts = n_pmts
        self.dropout_p = dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # TODO: Go over if we should use nested tensors
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.pmt_id_embeddings = nn.Embedding(n_pmts, d_model)

        self.hit_time_embedder = nn.Sequential(
            nn.Linear(1, hit_time_embedding_dim),
            nn.Tanh(),
            nn.Linear(hit_time_embedding_dim, d_model),
        )

        self.predict_time = predict_time
        if predict_time:

            class PosAndTime(nn.Module):
                # Class that adds a bias to the position prediction but not the time prediction
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(d_model, 4, bias=False)
                    # Mask is to stop gradients flowing into the time bias value
                    self.mask = nn.Buffer(torch.tensor([0.0, 1.0, 1.0, 1.0]))
                    bias = self.mask * nn.Linear(1, 4, bias=True).bias.contiguous()
                    self.bias = nn.Parameter(bias)

                def forward(self, input: torch.Tensor):
                    return self.linear(input) + self.mask * self.bias

            self.position_predictor = PosAndTime()

        else:
            self.position_predictor = nn.Linear(d_model, 3)

        if norm_dict is not None:
            if "input_norms" in norm_dict:
                self.add_input_norm(**norm_dict["input_norms"])
            if "output_norms" in norm_dict:
                self.add_output_unnorm(**norm_dict["output_norms"])
        else:
            self.input_norm = False
            self.output_unnorm = False

    def forward(self, hit_times: torch.FloatTensor, pmt_ids: torch.LongTensor) -> torch.FloatTensor:
        if self.input_norm:
            hit_times = self.input_normalize(hit_times)

        pmt_masks = pmt_ids == 0

        x = self.pmt_id_embeddings(pmt_ids) + self.hit_time_embedder(hit_times.unsqueeze(-1))

        x = self.transformer_encoder(x, src_key_padding_mask=pmt_masks)

        not_padding_masks = ~pmt_masks
        x = torch.sum(x * not_padding_masks.unsqueeze(-1), dim=-2) / not_padding_masks.sum(-1).unsqueeze(-1)

        x = self.position_predictor(x)

        if self.predict_time:
            out = {"positions": x[..., -3:], "times": x[..., 0]}
        else:
            out = {"positions": x}

        if self.output_unnorm:
            out = self.output_unnormalize(out)

        return out
