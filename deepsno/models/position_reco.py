import torch
from torch import nn

from deepsno.utils.train import copy_if_tensor
from deepsno.models.transformers import SetEncoderVarlenPadded


class PositionReco(nn.Module):
    def add_input_norm(
        self,
        time_shift: float = 0.0,
        time_scale: float = 1.0,
        input_norm: bool = True,
    ):
        self.register_buffer("time_shift", copy_if_tensor(time_shift))
        self.register_buffer("time_scale", copy_if_tensor(time_scale))
        self.input_norm = input_norm

    def add_output_unnorm(
        self,
        position_shifts: tuple[float, float, float],
        position_scales: tuple[float, float, float],
        output_unnorm: bool = True,
    ):
        self.register_buffer("position_shifts", copy_if_tensor(position_shifts))
        self.register_buffer("position_scales", copy_if_tensor(position_scales))

        self.output_unnorm = output_unnorm

    def time_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return (hit_times - self.time_shift) / self.time_scale

    def time_unnormalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return hit_times * self.time_scale + self.time_shift

    def position_normalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return (positions - self.position_shifts) / self.position_scales

    def position_unnormalize(self, positions: torch.FloatTensor) -> torch.FloatTensor:
        return positions * self.position_scales + self.position_shifts

    def input_normalize(self, hit_times: torch.FloatTensor) -> torch.FloatTensor:
        return self.time_normalize(hit_times)

    def output_unnormalize(self, predict: torch.FloatTensor) -> dict:
        out = {"positions": self.position_unnormalize(predict["positions"])}
        if "times" in predict:
            out["times"] = self.time_unnormalize(predict["times"])
        return out

    def output_normalize(self, predict: torch.FloatTensor) -> dict:
        out = {"positions": self.position_normalize(predict["positions"])}
        if "times" in predict:
            out["times"] = self.time_normalize(predict["times"])
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
        
        self.transformer_encoder = SetEncoderVarlenPadded(
            dim_in=d_model,
            dim_hidden=d_model,
            num_heads=nhead,
            num_sabs=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
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

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.transformer_encoder(x, mask=pmt_ids != 0)

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
