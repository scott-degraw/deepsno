from contextlib import nullcontext

import awkward as ak
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from deepsno.utils import stats

print("Implement swiglu")
print("Implement RMSnorm")
print("Implement dummy token")
print("Think about time scaling")
print("Check that activation layers are each being initialized right")
print("Look at inputting encodings at each layer")
print("Consider post norm")
print("Give encoder a final layer norm?")
print("Remove the initial object decoder")


class HitTimeFeatures:
    def __init__(self, n_bins: int, range: tuple[float, float], dtype=np.float32):
        self.bins = np.linspace(*range, n_bins + 1)
        self.dtype = dtype

    def __call__(self, hit_times: ak.Array):
        hist = stats.hist_jagged(hit_times, self.bins, dtype=self.dtype)
        return hist


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        embedding_dim: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        backends: list[SDPBackend] | None = None,
        dtype=None,
    ):
        super().__init__()

        self._qkv_same_embed_dim = q_dim == k_dim == v_dim
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(q_dim, 3 * embedding_dim, bias=bias, dtype=dtype)
        else:
            self.q_proj = nn.Linear(q_dim, embedding_dim, bias=bias, dtype=dtype)
            self.k_proj = nn.Linear(k_dim, embedding_dim, bias=bias, dtype=dtype)
            self.v_proj = nn.Linear(v_dim, embedding_dim, bias=bias, dtype=dtype)

        dim_out = q_dim
        self.out_proj = nn.Linear(embedding_dim, dim_out, bias=bias, dtype=dtype)
        if embedding_dim % nheads != 0:
            raise ValueError("Embedding dim is not divisible by nheads")
        self.head_dim = embedding_dim // nheads
        self.nheads = nheads
        self.bias = bias
        self.dropout = dropout

        self.kernel_backend = nullcontext() if backends is None else sdpa_kernel(backends=backends, set_priority=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query = F.linear(query, q_weight, q_bias)
                key = F.linear(key, k_weight, k_bias)
                value = F.linear(value, v_weight, v_bias)

        else:
            query = self.q_proj(query)
            key = self.q_proj(key)
            value = self.v_proj(value)

        if src_key_padding_mask is not None:
            if not (query.shape == key.shape == value.shape):
                raise NotImplementedError(
                    "When using src_key_padding_mask, query, key, and value must have the same shape"
                )
            attn_mask = ~src_key_padding_mask[..., None, None, :]
        else:
            attn_mask = None

        def sdpa_prepare(x):
            return x.unflatten(-1, [self.nheads, self.head_dim]).transpose(-2, -3)

        query = sdpa_prepare(query)
        key = sdpa_prepare(key)
        value = sdpa_prepare(value)

        dropout = self.dropout if self.training else 0.0

        with self.kernel_backend:
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout)

        attn_output = attn_output.transpose(-3, -2).flatten(-2)

        return self.out_proj(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dim_feedforward: int,
        nheads: int,
        dropout: float = 0.0,
        mha_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        eps: float = 1e-5,
        dtype=None,
    ):
        super().__init__()

        self.activation = activation

        self.linear1 = nn.Linear(model_dim, dim_feedforward, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim, bias=bias, dtype=dtype)

        self.norm1 = nn.LayerNorm(model_dim, bias=bias, eps=eps)
        self.norm2 = nn.LayerNorm(model_dim, bias=bias, eps=eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.multi_head_attn = MultiHeadAttention(
            q_dim=model_dim,
            k_dim=model_dim,
            v_dim=model_dim,
            embedding_dim=model_dim,
            nheads=nheads,
            dropout=mha_dropout,
            bias=bias,
            dtype=dtype,
        )

    def _self_attn(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None, encoding: torch.Tensor | None = None
    ):
        if encoding is not None:
            x = self.multi_head_attn(x + encoding, x + encoding, x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.multi_head_attn(x, x, x, src_key_padding_mask=src_key_padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None, encoding: torch.Tensor | None = None
    ):
        x = x + self._self_attn(self.norm1(x), src_key_padding_mask=src_key_padding_mask, encoding=encoding)
        x = x + self._ff_block(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, class_path: type, kwargs: dict):
        super().__init__()
        self.layers = nn.ModuleList([*(class_path(**kwargs) for _ in range(n_layers))])

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None, encoding: torch.Tensor | None = None
    ):
        x = self.layers[0](x, src_key_padding_mask=src_key_padding_mask, encoding=encoding)
        for layer in self.layers[1:]:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class ObjectDecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dim_feedforward: int,
        nheads: int,
        dropout: float = 0.0,
        mha_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
        eps: float = 1e-5,
        dtype=None,
    ):
        super().__init__()
        self.activation = activation

        self.linear1 = nn.Linear(model_dim, dim_feedforward, bias=bias, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim, bias=bias, dtype=dtype)

        self.norm1 = nn.LayerNorm(model_dim, bias=bias, eps=eps)
        self.norm2 = nn.LayerNorm(model_dim, bias=bias, eps=eps)
        self.norm3 = nn.LayerNorm(model_dim, bias=bias, eps=eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.mha_1 = MultiHeadAttention(
            q_dim=model_dim,
            k_dim=model_dim,
            v_dim=model_dim,
            embedding_dim=model_dim,
            nheads=nheads,
            dropout=mha_dropout,
            bias=bias,
            dtype=dtype,
        )

        self.mha_2 = MultiHeadAttention(
            q_dim=model_dim,
            k_dim=model_dim,
            v_dim=model_dim,
            embedding_dim=model_dim,
            nheads=nheads,
            dropout=mha_dropout,
            bias=bias,
            dtype=dtype,
        )

    def _object_self_attn(self, x: torch.Tensor, encoding: torch.Tensor | None = None):
        if encoding is not None:
            self.mha_1(x + encoding, x + encoding, x)
        else:
            x = self.mha_1(x, x, x)

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        x: torch.Tensor,
        encoder_x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        encoding: torch.Tensor | None = None,
    ):
        x = x + self._object_self_attn(self.norm1(x))
        x = self.norm2(self.dropout2(x))
        encoder_x = self.norm3(encoder_x)
        x = x + self.mha_2(x, encoder_x, encoder_x)

        x = x + self._ff_block(self.norm3(x))

        return x


class ObjectDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        object_head: nn.Module,
        class_path: type,
        n_object_queries: int,
        kwargs: dict,
        position_shift: float = 0.0,
        position_scale: float = 1.0,
        time_shift: float = 0.0,
        time_scale: float = 1.0,
    ):
        super().__init__()

        self.object_head = object_head
        self.layers = nn.ModuleList([*(class_path(**kwargs) for _ in range(n_layers))])

        self.object_query_tokens = nn.Embedding(n_object_queries, embedding_dim=kwargs["model_dim"])

        self.output_unnorm = False
        self.position_shift = position_shift
        self.position_scale = position_scale
        self.time_shift = time_shift
        self.time_scale = time_scale

    def output_normalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = (x["position"] - self.position_shift) / self.position_scale
        x["time"] = (x["time"] - self.time_shift) / self.time_scale
        return x

    def output_unnormalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = x["position"] * self.position_scale + self.position_shift
        x["time"] = x["time"] * self.time_scale + self.time_shift
        return x

    def forward(
        self,
        decoder_x: torch.Tensor,
        encoding: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.object_query_tokens.weight
        for layer in self.layers:
            x = layer(x, encoder_x=decoder_x, encoding=encoding, src_key_padding_mask=src_key_padding_mask)

        x = self.object_head(x)

        if self.output_unnorm:
            x = self.output_unnormalize(x)

        return x


class ObjectFFNHead(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # output dimension: 1 (exists logit) + 3 (position) + 1 (time) = 5
        self.linear = nn.Linear(model_dim, 5)
        # self.linear2 = nn.Linear(model_dim * 2, model_dim)
        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.linear(self.dropout(x))

        exists_logit = x[..., 0]
        position = x[..., 1:4]
        time = x[..., 4]

        return {"exists_logit": exists_logit, "position": position, "time": time}


class MultiHitEncoder(nn.Module):
    def __init__(
        self,
        n_pmts: int,
        model_dim: int,
        encoder: nn.Module,
        waveform_n_bins: int,
        time_scale: float = 1.0,
        time_shift: float = 0.0,
        dtype=None,
    ):
        super().__init__()
        self.n_pmts = n_pmts
        self.pmt_embed = nn.Embedding(n_pmts, embedding_dim=model_dim, dtype=dtype)
        self.hit_time_embed = nn.Linear(waveform_n_bins, model_dim, dtype=dtype)
        self.encoder = encoder
        self.time_scale = time_scale
        self.time_shift = time_shift

    def hit_time_normalize(self, hit_times: torch.Tensor) -> torch.Tensor:
        return (hit_times - self.time_shift) / self.time_scale

    def forward(self, waveforms: torch.Tensor, pmt_ids: torch.Tensor, src_key_padding_mask: torch.Tensor):
        hit_time_embeddings = self.hit_time_embed(waveforms)
        pmt_encoding = self.pmt_embed(pmt_ids)

        x = self.encoder(hit_time_embeddings, src_key_padding_mask=src_key_padding_mask, encoding=pmt_encoding)

        return x


class MultiVertexDecoder(nn.Module):
    def __init__(
        self,
        n_vertices: int,
        model_dim: int,
        bias: bool = True,
        output_unnorm: bool = False,
        position_scale: float = 1.0,
        position_shift: float = 0.0,
        time_scale: float = 1.0,
        time_shift: float = 0.0,
        dtype=None,
    ):
        super().__init__()
        self.projector = nn.Linear(model_dim, n_vertices * 4, dtype=dtype, bias=bias)
        self.output_unnorm = output_unnorm
        self.position_scale = position_scale
        self.position_shift = position_shift
        self.time_scale = time_scale
        self.time_shift = time_shift

    def output_normalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = (x["position"] - self.position_shift) / self.position_scale
        x["time"] = (x["time"] - self.time_shift) / self.time_scale
        return x

    def output_unnormalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = x["position"] * self.position_scale + self.position_shift
        x["time"] = x["time"] * self.time_scale + self.time_shift
        return x

    def forward(self, x: torch.Tensor):
        x = torch.mean(x, axis=-2)

        x = self.projector(x)
        output = {
            "position": x[..., :3],
            "time": x[..., 3],
        }
        if self.output_unnorm:
            output = self.output_unnormalize(output)

        return output


@torch.compile(dynamic=False, fullgraph=True)
class MultiHit(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._output_unnorm = self.decoder.output_unnorm

        torch.set_float32_matmul_precision("high")

    def output_normalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decoder.output_normalize(x)

    def output_unnormalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decoder.output_unnormalize(x)

    @property
    def output_unnorm(self):
        return self._output_unnorm

    @output_unnorm.setter
    def output_unnorm(self, value: bool):
        self._output_unnorm = value
        self.decoder.output_unnorm = value

    def forward(self, waveforms: torch.Tensor, pmt_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(waveforms=waveforms, pmt_ids=pmt_ids, src_key_padding_mask=None)
        return self.decoder(encoded)
