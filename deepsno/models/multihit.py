from contextlib import nullcontext

import numpy as np
import torch
import torch._ops
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from deepsno.models import transformers


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


class ObjectDecoderLayerVarlen(nn.Module):
    """
    Single transformer decoder layer backed by :class:`~deepsno.models.transformers.MABVarlen`.

    Applies two MAB blocks in sequence:

    1. **Self-attention** — queries attend to each other.
    2. **Cross-attention** — queries attend to the (varlen-packed) encoder output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            dim: Feature dimension for both queries and encoder output.
            num_heads: Number of attention heads.
            dim_feedforward: FFN hidden dimension.  Defaults to ``4 * dim``.
            dropout: Dropout probability passed to each MABVarlen block.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.self_attn = transformers.MABVarlen(
            dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.cross_attn = transformers.MABVarlen(
            dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(
        self,
        queries: torch.Tensor,
        encoder_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_enc: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_enc: int,
    ) -> torch.Tensor:
        """
        Args:
            queries: ``(B * n_queries, dim)`` packed query tensor.
            encoder_out: ``(total_hits, dim)`` packed encoder output.
            cu_seqlens_q: Int32 ``(B + 1,)`` uniform cumulative lengths for queries.
            cu_seqlens_enc: Int32 ``(B + 1,)`` variable cumulative lengths for encoder hits.
            max_seqlen_q: Maximum query length (equal to ``n_queries``).
            max_seqlen_enc: Maximum hit sequence length.

        Returns:
            Updated queries of shape ``(B * n_queries, dim)``.
        """
        queries = self.self_attn(
            queries,
            queries,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_q,
        )
        queries = self.cross_attn(
            queries,
            encoder_out,
            cu_seqlens_q,
            cu_seqlens_enc,
            max_seqlen_q,
            max_seqlen_enc,
        )
        return queries


class ObjectDecoderVarlen(nn.Module):
    """
    Object query decoder for varlen encoder outputs.

    Maintains ``n_queries`` learnable query tokens and refines them through
    stacked :class:`ObjectDecoderLayerVarlen` blocks that attend to the packed
    encoder output.  The final embeddings are returned as
    ``(batch_size, n_queries, dim)`` — apply :class:`ObjectFFNHead` externally
    to produce predictions.
    """

    def __init__(
        self,
        n_queries: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        position_shift: float = 0.0,
        position_scale: float = 1.0,
        time_shift: float = 0.0,
        time_scale: float = 1.0,
        energy_scale: float = 1.0,
        energy_shift: float = 0.0,
        time_weight: float = 1.0,
        position_weight: float = 1.0,
        class_weight: float = 1.0,
        dynamic_task_weighting: bool = False,
    ):
        """
        Args:
            n_queries: Number of learnable object query tokens.
            dim: Feature / model dimension.
            num_heads: Number of attention heads per layer.
            num_layers: Number of stacked decoder layers.
            dim_feedforward: FFN hidden dimension.  Defaults to ``4 * dim``.
            dropout: Dropout probability applied after each decoder layer.
            bias: Whether to use bias in linear layers.
            position_shift / position_scale: Output (un)normalisation for position.
            time_shift / time_scale: Output (un)normalisation for time.
            energy_shift / energy_scale: Output (un)normalisation for energy.
            time_weight / position_weight / class_weight: Initial task-loss
                weights, converted to log-sigma² parameters.
            dynamic_task_weighting: If ``True`` sigma² are learnable parameters.
        """
        super().__init__()
        self.n_queries = n_queries
        self.query_tokens = nn.Embedding(n_queries, embedding_dim=dim)

        self.layers = nn.ModuleList(
            [ObjectDecoderLayerVarlen(dim, num_heads, dim_feedforward, dropout, bias) for _ in range(num_layers)]
        )

        self.output_unnorm = False
        self.position_shift = position_shift
        self.position_scale = position_scale
        self.time_shift = time_shift
        self.time_scale = time_scale
        self.energy_shift = energy_shift
        self.energy_scale = energy_scale

        def _w2s(w):
            return -np.log(w)

        grad = dynamic_task_weighting
        self.log_pos_sigma2 = nn.Parameter(torch.tensor([_w2s(position_weight)]), requires_grad=grad)
        self.log_time_sigma2 = nn.Parameter(torch.tensor([_w2s(time_weight)]), requires_grad=grad)
        self.log_class_sigma2 = nn.Parameter(torch.tensor([_w2s(class_weight)]), requires_grad=grad)

    def output_normalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = (x["position"] - self.position_shift) / self.position_scale
        x["time"] = (x["time"] - self.time_shift) / self.time_scale
        x["energy"] = (x["energy"] - self.energy_shift) / self.energy_scale
        return x

    def output_unnormalize(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x["position"] = x["position"] * self.position_scale + self.position_shift
        x["time"] = x["time"] * self.time_scale + self.time_shift
        x["energy"] = x["energy"] * self.energy_scale + self.energy_shift
        return x

    def forward(
        self,
        encoder_out: torch.Tensor,
        cu_seqlens_enc: torch.Tensor,
        max_seqlen_enc: int,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: ``(total_hits, dim)`` packed encoder output.
            cu_seqlens_enc: Int32 ``(B + 1,)`` cumulative hit sequence lengths.
            max_seqlen_enc: Maximum hit sequence length across the batch.

        Returns:
            ``(batch_size, n_queries, dim)`` refined query embeddings.
        """
        batch_size = cu_seqlens_enc.shape[0] - 1
        dim = self.query_tokens.embedding_dim

        # Expand query tokens across the batch: (B * n_queries, dim)
        q = self.query_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, dim)

        # Uniform cu_seqlens for query tokens
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * self.n_queries,
            step=self.n_queries,
            device=encoder_out.device,
            dtype=cu_seqlens_enc.dtype,
        )
        max_seqlen_q = self.n_queries

        for layer in self.layers:
            q = layer(q, encoder_out, cu_seqlens_q, cu_seqlens_enc, max_seqlen_q, max_seqlen_enc)

        return q.view(batch_size, self.n_queries, dim)


class ObjectFFNHead(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.1, bias: bool = True, eps=1e-5):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 6),
        )

    def forward(self, x: torch.Tensor):
        x = self.decoder(x)

        return {
            "exists_logit": x[..., 0],
            "position": x[..., 1:4],
            "time": x[..., 4],
            "energy": F.softplus(x[..., 5]),
        }


class MultiHitPMTEncoderBase(nn.Module):
    """
    Shared base for PMT encoder variants.

    Holds the PMT embedding table, the downstream set encoder, and time
    normalisation parameters.  Subclasses implement :meth:`_embed_hits` to
    produce a ``(B, n_pmts, model_dim)`` per-PMT feature tensor from the
    raw inputs, which is then packed into varlen format and passed through
    the encoder.
    """

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
        """
        Args:
            n_pmts: Number of PMT IDs for the embedding table.
            model_dim: Model hidden dimension.
            encoder: A :class:`~deepsno.models.transformers.SetEncoderVarlen` or
                :class:`~deepsno.models.transformers.InducedSetEncoderVarlen`.
            waveform_n_bins: Number of time bins (used by subclasses).
            time_scale: Normalisation scale for hit times.
            time_shift: Normalisation shift for hit times.
        """
        super().__init__()
        self.n_pmts = n_pmts
        self.pmt_embed = nn.Embedding(n_pmts, embedding_dim=model_dim, dtype=dtype)
        self.encoder = encoder
        self.time_scale = time_scale
        self.time_shift = time_shift

    def hit_time_normalize(self, hit_times: torch.Tensor) -> torch.Tensor:
        return (torch.log(hit_times) - self.time_shift) / self.time_scale

    def _embed_hits(self, pmt_ids: torch.Tensor, hit_times: torch.Tensor) -> torch.Tensor:
        """Return ``(total_hits, model_dim)`` per-hit feature tensor."""
        raise NotImplementedError

    def forward(
        self,
        pmt_ids: torch.Tensor,
        hit_times: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            pmt_ids: ``(total_hits,)`` flat PMT indices from
                :func:`~deepsno.data.multihit.multihit_varlen_collate`.
            hit_times: ``(total_hits,)`` flat hit times.
            cu_seqlens: Int32 ``(B+1,)`` cumulative sequence lengths.
            max_seqlen: Maximum sequence length across the batch.

        Returns:
            ``(encoded, cu_seqlens, max_seqlen)`` where ``encoded`` is
            ``(total_hits, model_dim)`` packed.
        """
        x = self._embed_hits(pmt_ids, hit_times)
        encoded = self.encoder(x, cu_seqlens, max_seqlen)
        return encoded, cu_seqlens, max_seqlen


class MultiHitPMTEncoderUnique(MultiHitPMTEncoderBase):
    """
    Encoder for :class:`~deepsno.data.multihit.MultiHitDatasetUnique` output.

    Expects pre-uniquified PMT IDs with associated per-PMT hit counts.
    Hit-time bin indices are looked up in a learned embedding table, then
    summed across hits per PMT via ``torch.segment_reduce``, and added to
    the PMT embedding.

    Input keys required in batch:
        - ``pmt_ids``: ``(B, n_pmts)`` unique PMT IDs, zero-padded.
        - ``pmt_id_counts``: ``(B, n_pmts)`` number of hits per PMT.
        - ``hit_times``: ``(B, max_context_len)`` binned hit-time indices.
    """

    def __init__(self, *args, waveform_n_bins: int, **kwargs):
        super().__init__(*args, waveform_n_bins=waveform_n_bins, **kwargs)
        model_dim = self.pmt_embed.embedding_dim
        self.hit_time_embed = nn.Embedding(waveform_n_bins, embedding_dim=model_dim, dtype=self.pmt_embed.weight.dtype)

    def _embed_hits(self, pmt_ids: torch.Tensor, hit_times: torch.Tensor, pmt_id_counts: torch.Tensor) -> torch.Tensor:
        # Sum binned hit-time embeddings per PMT, then add PMT embedding
        hit_time_embed = self.hit_time_embed(hit_times)
        hit_time_embed = torch.segment_reduce(hit_time_embed, reduce="sum", lengths=pmt_id_counts, axis=-2)
        x = hit_time_embed + self.pmt_embed(pmt_ids)
        x = (pmt_ids != 0).unsqueeze(-1) * x
        return x

    def forward(
        self,
        pmt_ids: torch.Tensor,
        hit_times: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pmt_id_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = self._embed_hits(pmt_ids, hit_times, pmt_id_counts)
        encoded = self.encoder(x, cu_seqlens, max_seqlen)
        return encoded, cu_seqlens, max_seqlen


class MultiHitPMTEncoderExpanded(MultiHitPMTEncoderBase):
    """
    Encoder for :class:`~deepsno.data.multihit.MultiHitDatasetExpanded` output.

    Expects one entry per hit (expanded/repeated form).  Hit-time bin indices
    are embedded, then hits are grouped by PMT via
    ``torch.unique_consecutive`` + ``torch.segment_reduce`` (mean), and the
    result is added to the PMT embedding.

    Input keys required in batch:
        - ``pmt_ids``: ``(B, max_context_len)`` PMT IDs, one per hit,
          sorted by PMT ID and zero-padded.
        - ``hit_times``: ``(B, max_context_len)`` binned hit-time indices.
    """

    def __init__(self, *args, waveform_n_bins: int, time_embed_dropout: float = 0.1, **kwargs):
        super().__init__(*args, waveform_n_bins=waveform_n_bins, **kwargs)
        model_dim = self.pmt_embed.embedding_dim

        self.hit_time_embed = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.Tanh(),
            nn.Dropout(time_embed_dropout),
            nn.Linear(model_dim, model_dim),
        )

    def _embed_hits(self, pmt_ids: torch.Tensor, hit_times: torch.Tensor) -> torch.Tensor:
        hit_times = self.hit_time_normalize(hit_times)
        x = self.hit_time_embed(hit_times.unsqueeze(-1))
        x = x + self.pmt_embed(pmt_ids)
        return x


# Backward-compatible alias
MultiHitPMTEncoder = MultiHitPMTEncoderUnique


@torch.compile(dynamic=False, fullgraph=True)
class MultiHit(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, head: nn.Module):
        """
        Args:
            encoder: :class:`MultiHitEncoder` — embeds and packs the hit data.
            decoder: :class:`ObjectDecoderVarlen` — refines object queries.
            head: :class:`ObjectFFNHead` (or compatible) — maps query embeddings
                to prediction dicts.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
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

    def forward(
        self,
        pmt_ids: torch.Tensor,
        hit_times: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded, cu_seqlens, max_seqlen = self.encoder(pmt_ids, hit_times, cu_seqlens, max_seqlen)
            queries = self.decoder(encoded, cu_seqlens, max_seqlen)  # (B, n_queries, dim)

        output = self.head(queries.float())

        if self.decoder.output_unnorm:
            output = self.decoder.output_unnormalize(output)

        return {
            **output,
            "log_sigma2": {
                "position": self.decoder.log_pos_sigma2,
                "time": self.decoder.log_time_sigma2,
                "exists": self.decoder.log_class_sigma2,
            },
        }
