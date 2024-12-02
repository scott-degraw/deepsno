import torch
import torch.nn.functional as F
from torch import nn


class PositionReco(nn.Module):
    def __init__(
        self,
        n_pmts: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        hit_time_embedding_dim: int,
    ):
        super().__init__()
        self.n_pmts = n_pmts
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

        self.pmt_embedder = nn.Linear(n_pmts + 1, d_model)
        self.hit_time_embedder = nn.Sequential(
            nn.Linear(1, hit_time_embedding_dim),
            nn.Tanh(),
            nn.Linear(hit_time_embedding_dim, d_model),
        )
        self.position_predictor = nn.Linear(d_model, 3)

    def forward(self, hit_times, pmt_ids):
        pmt_masks = pmt_ids == -1  # -1 indicates the PMT is padded
        not_padding_masks = ~pmt_masks

        pmt_ids += 1  # F.one_hot requires all classes be labelled by 01
        pmt_one_hot = F.one_hot(pmt_ids, num_classes=self.n_pmts + 1).float()

        pmt_embedding = self.pmt_embedder(pmt_one_hot)

        hit_time_embedding = self.hit_time_embedder(hit_times.unsqueeze(-1))

        x = pmt_embedding + hit_time_embedding

        x = self.transformer_encoder(x, src_key_padding_mask=pmt_masks)

        # TODO: Try an einsum here
        x = torch.div(
            torch.sum(x * not_padding_masks.unsqueeze(2), dim=1), torch.sum(not_padding_masks, dim=1).unsqueeze(1)
        )

        x = self.position_predictor(x)

        return x
