import torch
import torch._ops
from torch import nn
from torch.nn import functional as F

from deepsno.models import transformers
from deepsno.models.transformers import VarlenTensor


class ObjectDecoderLayerVarlen(nn.Module):
    """
    Single transformer decoder layer backed by :class:`~deepsno.models.transformers.ISABVarlen`.

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
            dim,
            num_heads,
            bias=bias,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.cross_attn = transformers.MABVarlen(
            dim,
            num_heads,
            bias=bias,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, q: VarlenTensor, enc: VarlenTensor) -> VarlenTensor:
        """
        Args:
            q: Query :class:`~deepsno.models.transformers.VarlenTensor` —
               ``(B * n_queries, dim)`` packed, uniform cu_seqlens.
            enc: Encoder :class:`~deepsno.models.transformers.VarlenTensor` —
                 ``(total_hits, dim)`` packed, variable cu_seqlens.

        Returns:
            :class:`~deepsno.models.transformers.VarlenTensor` with the same structure as ``q``.
        """
        q = self.self_attn(q, q)
        return self.cross_attn(q, enc)


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
        deep_supervision: bool = False,
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
            deep_supervision: If True, forward returns ``(final_q, [intermediate_q_0, ...,
                intermediate_q_{N-2}])`` so callers can apply auxiliary losses on each layer.
        """
        super().__init__()
        self.n_queries = n_queries
        self.query_tokens = nn.Embedding(n_queries, embedding_dim=dim)

        self.layers = nn.ModuleList(
            [
                ObjectDecoderLayerVarlen(
                    dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
                )
                for _ in range(num_layers)
            ]
        )

        self.deep_supervision = deep_supervision
        self.norm = nn.LayerNorm(dim)
        self.output_unnorm = False
        self.position_shift = position_shift
        self.position_scale = position_scale
        self.time_shift = time_shift
        self.time_scale = time_scale
        self.energy_shift = energy_shift
        self.energy_scale = energy_scale

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

    def forward(self, enc_vt: VarlenTensor):
        """
        Args:
            enc_vt: :class:`~deepsno.models.transformers.VarlenTensor` of packed
                encoder output, shape ``(total_hits, dim)``.

        Returns:
            When ``deep_supervision=False`` (default): ``(batch_size, n_queries, dim)`` tensor.
            When ``deep_supervision=True``: tuple of ``(final_q, [intermediate_q_0, ...,
            intermediate_q_{N-2}])`` where each element is ``(batch_size, n_queries, dim)``.
        """
        batch_size = enc_vt.cu_seqlens.shape[0] - 1
        dim = self.query_tokens.embedding_dim

        q = self.query_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, dim)
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * self.n_queries,
            step=self.n_queries,
            dtype=enc_vt.cu_seqlens.dtype,
            device=enc_vt.cu_seqlens.device,
        )
        q_vt = VarlenTensor(q, cu_seqlens_q, self.n_queries)

        if self.deep_supervision:
            intermediates = []
            for i, layer in enumerate(self.layers):
                q_vt = layer(q_vt, enc_vt)
                if i < len(self.layers) - 1:
                    intermediates.append(self.norm(q_vt.data.view(batch_size, self.n_queries, dim)))
            return self.norm(q_vt.data.view(batch_size, self.n_queries, dim)), intermediates

        for layer in self.layers:
            q_vt = layer(q_vt, enc_vt)
        return self.norm(q_vt.data.view(batch_size, self.n_queries, dim))


class ObjectFFNHead(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.1, bias: bool = True, eps=1e-5):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 7),
        )

    def forward(self, x: torch.Tensor):
        x = self.decoder(x)

        return {
            "exists_logit": x[..., 0],
            "position": x[..., 1:4],
            "time": x[..., 4],
            "energy": F.softplus(x[..., 5]),
            "log_weight": x[..., 6],
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
        self.base_pmt_ids = torch.arange(n_pmts, dtype=torch.long)
        self.pmt_embed = nn.Embedding(n_pmts, embedding_dim=model_dim, dtype=dtype)
        self.encoder = encoder
        self.time_scale = time_scale
        self.time_shift = time_shift

    def hit_time_normalize(self, hit_times: torch.Tensor) -> torch.Tensor:
        return (torch.log(hit_times) - self.time_shift) / self.time_scale

    def _embed_hits(self, pmt_ids: torch.Tensor, hit_times: torch.Tensor) -> torch.Tensor:
        """Return ``(total_hits, model_dim)`` per-hit feature tensor."""
        raise NotImplementedError

    def forward(self, hits: VarlenTensor, hit_times: torch.Tensor) -> VarlenTensor:
        """
        Args:
            hits: :class:`~deepsno.models.transformers.VarlenTensor` whose
                ``data`` field holds ``(total_hits,)`` flat PMT indices.
            hit_times: ``(total_hits,)`` flat hit times.

        Returns:
            :class:`~deepsno.models.transformers.VarlenTensor` with
            ``(total_hits, model_dim)`` packed data.
        """
        x = self._embed_hits(hits.data, hit_times)
        x = self.encoder(hits._replace(data=x))
        return x


class MultiHitPMTEncoderUnique(nn.Module):
    """
    PMT encoder for the :class:`~deepsno.data.multihit.UniqueHitInputMaker` output.

    Receives varlen flat hit times (concatenated across the batch, sorted by PMT id
    within each event) and a fixed-size ``pmt_lengths`` tensor.  A single
    ``segment_reduce`` aggregates hits per PMT, yielding a fixed-size
    ``(B, n_pmts, dim)`` representation that is then processed by a fixed-size
    :class:`~deepsno.models.transformers.InducedSetEncoder`.

    Input keys (from :class:`~deepsno.data.multihit.MultiHitUniqueCollate`):
        - ``hit_times``:   ``(total_hits,)`` varlen flat hit times.
        - ``pmt_lengths``: ``(B, n_pmts)`` hit count per PMT per event.
    """

    def __init__(
        self,
        n_pmts: int,
        model_dim: int,
        encoder: nn.Module,
        hit_time_embed: nn.Module,
        time_scale: float = 1.0,
        time_shift: float = 0.0,
        dtype=None,
    ):
        super().__init__()
        self.n_pmts = n_pmts
        self.pmt_embed = nn.Embedding(n_pmts, model_dim, dtype=dtype)
        self.hit_time_embed = hit_time_embed
        self.encoder = encoder
        self.time_scale = time_scale
        self.time_shift = time_shift

    def forward(self, hit_times: torch.Tensor, pmt_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hit_times:   ``(total_hits,)`` flat hit times concatenated across the batch.
            pmt_lengths: ``(B, n_pmts)`` hit count per PMT per event.

        Returns:
            ``(B, n_pmts, model_dim)`` encoded PMT representation.
        """
        B = pmt_lengths.shape[0]
        hit_times_norm = (torch.log(hit_times) - self.time_shift) / self.time_scale
        hit_embed = self.hit_time_embed(hit_times_norm.unsqueeze(-1))  # (total_hits, dim)

        x = torch.segment_reduce(hit_embed, reduce="sum", lengths=pmt_lengths.view(-1), axis=0)  # (B*n_pmts, dim)
        x = x / pmt_lengths.view(-1, 1).clamp(min=1)  # mean pooling, avoid div by zero
        x = x.view(B, self.n_pmts, -1)  # (B, n_pmts, dim)
        x = x + self.pmt_embed.weight.to(dtype=x.dtype)  # keep bfloat16 residual stream
        return self.encoder(x)  # (B, n_pmts, dim)


class MultiHitPMTEncoderExpanded(MultiHitPMTEncoderBase):
    """
    Encoder for per-hit varlen input produced by
    :class:`~deepsno.data.multihit.MultiHitExpandedVarlenCollate`.

    Each hit's time embedding is added to its PMT embedding, then the packed
    sequence is passed through the set encoder.

    Input keys:
        - ``hits``:      :class:`~deepsno.models.transformers.VarlenTensor` whose
          ``data`` field holds ``(total_hits,)`` flat PMT IDs.
        - ``hit_times``: ``(total_hits,)`` flat hit times.
    """

    def __init__(self, *args, hit_time_embed: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_time_embed = hit_time_embed

    def _embed_hits(self, pmt_ids: torch.Tensor, hit_times: torch.Tensor) -> torch.Tensor:
        hit_times = self.hit_time_normalize(hit_times)
        x = self.hit_time_embed(hit_times.unsqueeze(-1))
        x = x + self.pmt_embed(pmt_ids)
        return x


# Backward-compatible alias
MultiHitPMTEncoder = MultiHitPMTEncoderUnique


class ObjectDecoderLayer(nn.Module):
    """Single fixed-size decoder layer: self-attention among queries, then cross-attention to encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.self_attn = transformers.MAB(
            dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
        )
        self.cross_attn = transformers.MAB(
            dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
        )

    def forward(self, q: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:   ``(B, n_queries, dim)`` query tensor.
            enc: ``(B, N, dim)`` encoder output.

        Returns:
            ``(B, n_queries, dim)`` updated queries.
        """
        q = self.self_attn(q, q)
        return self.cross_attn(q, enc)


class ObjectDecoder(nn.Module):
    """Fixed-size object query decoder for dense ``(B, N, dim)`` encoder outputs.

    Maintains ``n_queries`` learnable query tokens and refines them through stacked
    :class:`ObjectDecoderLayer` blocks.  Drop-in replacement for
    :class:`ObjectDecoderVarlen` when the encoder output is a dense tensor rather
    than a :class:`~deepsno.models.transformers.VarlenTensor`.
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
        energy_shift: float = 0.0,
        energy_scale: float = 1.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.query_tokens = nn.Embedding(n_queries, dim)
        self.layers = nn.ModuleList(
            [
                ObjectDecoderLayer(
                    dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
                )
                for _ in range(num_layers)
            ]
        )

        self.deep_supervision = deep_supervision
        self.norm = nn.LayerNorm(dim)
        self.output_unnorm = False
        self.position_shift = position_shift
        self.position_scale = position_scale
        self.time_shift = time_shift
        self.time_scale = time_scale
        self.energy_shift = energy_shift
        self.energy_scale = energy_scale

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

    def forward(self, enc: torch.Tensor):
        """
        Args:
            enc: ``(B, N, dim)`` encoder output.

        Returns:
            When ``deep_supervision=False`` (default): ``(B, n_queries, dim)`` tensor.
            When ``deep_supervision=True``: tuple of ``(final_q, [intermediate_q_0, ...,
            intermediate_q_{N-2}])`` where each element is ``(B, n_queries, dim)``.
        """
        B = enc.shape[0]
        q = self.query_tokens.weight.to(dtype=enc.dtype).unsqueeze(0).expand(B, -1, -1)

        if self.deep_supervision:
            intermediates = []
            for i, layer in enumerate(self.layers):
                q = layer(q, enc)
                if i < len(self.layers) - 1:
                    intermediates.append(self.norm(q))
            return self.norm(q), intermediates

        for layer in self.layers:
            q = layer(q, enc)
        return self.norm(q)


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
        self.register_forward_pre_hook(self._sync_batch_size, with_kwargs=True)

    def _sync_batch_size(self, module, args, kwargs):
        hits = kwargs.get("hits")
        if isinstance(hits, VarlenTensor):
            torch._dynamo.mark_static(hits.cu_seqlens, 0)

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

    @torch.compile(dynamic=True, fullgraph=True)
    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc_vt = self.encoder(**inputs)
            decoder_out = self.decoder(enc_vt)

        if self.decoder.deep_supervision:
            queries, intermediates = decoder_out
            output = self.head(queries.float())
            output["aux_outputs"] = [self.head(q.float()) for q in intermediates]
        else:
            output = self.head(decoder_out.float())

        if self.decoder.output_unnorm:
            output = self.decoder.output_unnormalize(output)
            if self.decoder.deep_supervision:
                output["aux_outputs"] = [self.decoder.output_unnormalize(aux) for aux in output["aux_outputs"]]

        return output
