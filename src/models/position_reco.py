from pathlib import Path

import h5py
import torch
from torch import nn

from src.utils.train import copy_if_tensor


class PositionRecoNorm(dict):
    def __init__(self, train_file: str | Path, positions: tuple = ["x", "y", "z"]):
        super().__init__()

        with h5py.File(train_file) as h5_file:
            self["input_norms"] = {
                "hit_time_mean": float(h5_file["cal_pmt_events/hit_times"].attrs["mean"]),
                "hit_time_rmsd": float(h5_file["cal_pmt_events/hit_times"].attrs["root_mean_square_deviation"]),
            }
            self["output_norms"] = {
                "position_means": [float(h5_file[f"mc_truth/position/{c}"].attrs["mean"]) for c in positions],
                "position_rmsds": [
                    float(h5_file[f"mc_truth/position/{c}"].attrs["root_mean_square_deviation"]) for c in positions
                ],
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

    def input_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.hit_time_mean) / self.hit_time_rmsd

    def output_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return positions * self.position_rmsds + self.position_means

    def output_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return (positions - self.position_means) / self.position_rmsds

    def __init__(
        self,
        n_pmts: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        hit_time_embedding_dim: int,
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

        pmt_id_embeddings = nn.Linear(n_pmts, d_model, bias=False).weight.T.contiguous()
        self.register_parameter("pmt_id_embeddings", nn.Parameter(pmt_id_embeddings))

        self.hit_time_embedder = nn.Sequential(
            nn.Linear(1, hit_time_embedding_dim),
            nn.Tanh(),
            nn.Linear(hit_time_embedding_dim, d_model),
        )
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

        x = self.pmt_id_embeddings[pmt_ids] + self.hit_time_embedder(hit_times.unsqueeze(-1))

        x = self.transformer_encoder(x, src_key_padding_mask=pmt_masks)

        not_padding_masks = ~pmt_masks
        x = torch.sum(x * not_padding_masks.unsqueeze(-1), dim=-2) / not_padding_masks.sum(-1).unsqueeze(-1)

        x = self.position_predictor(x)

        if self.output_unnorm:
            x = self.output_unnormalize(x)

        return x
