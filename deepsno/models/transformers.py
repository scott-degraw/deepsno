from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None


class VarlenTensor(NamedTuple):
    """Bundles a packed flat tensor with its varlen bookkeeping.

    Fields:
        data:       ``(total_elements, dim)`` packed feature tensor.
        cu_seqlens: Int32 ``(B + 1,)`` cumulative sequence lengths.
        max_seqlen: Maximum sequence length across the batch.
    """

    data: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


def padded_to_varlen(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> "VarlenTensor":
    """Convert padded rectangular tensors from a dataloader to the flat varlen
    format expected by :class:`SetTransformerVarlen` (and its sub-modules).

    Args:
        x: Float tensor of shape ``(batch_size, max_len, dim)`` containing
           padded set elements.
        mask: Boolean tensor of shape ``(batch_size, max_len)``.  A ``True``
              entry indicates a *valid* (non-padding) element.

    Returns:
        :class:`VarlenTensor` with ``data``, ``cu_seqlens``, and ``max_seqlen``.

    Example::

        vt = padded_to_varlen(x_pad, mask)
        out = model(vt)
    """
    # seqlens: (batch_size,) – number of valid tokens per sample
    seqlens = mask.sum(dim=1)  # (B,)
    # Use the statically-known padded length as a conservative upper bound for
    # max_seqlen.  This avoids a data-dependent .item() call, which would cause
    # a graph break under torch.compile.
    max_seqlen = mask.shape[1]

    # cu_seqlens: [0, n_0, n_0+n_1, ..., total] shape (B+1,)
    # F.pad avoids an in-place index-put (IndexPutBackward) that would break
    # the torch.compile graph.
    cu_seqlens = F.pad(seqlens.cumsum(dim=0).to(torch.int32), (1, 0))

    # Pack only the valid elements (mask == True) row-by-row
    x_flat = x[mask]  # (total, D)

    return VarlenTensor(x_flat, cu_seqlens, max_seqlen)


def varlen_to_padded(
    x_flat: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Inverse of :func:`padded_to_varlen`.

    Scatters a flat packed tensor back into a zero-padded ``(B, L, D)`` tensor
    using the original validity mask.

    Args:
        x_flat: ``(total_elements, dim)`` packed tensor — e.g. the output of a
            varlen encoder.
        mask: ``(batch_size, max_len)`` boolean mask — the **same** mask passed
            to :func:`padded_to_varlen`.  ``True`` = valid position.  The
            number of ``True`` entries must equal ``total_elements``.
        fill_value: Value written into padding positions (default ``0.0``).

    Returns:
        ``(batch_size, max_len, dim)`` tensor with valid positions restored from
        ``x_flat`` and padding positions set to ``fill_value``.

    Example::

        vt = padded_to_varlen(x_pad, mask)
        encoded_vt = encoder(vt)
        encoded_pad = varlen_to_padded(encoded_vt.data, mask)
    """
    B, L = mask.shape
    D = x_flat.shape[-1]
    out = x_flat.new_full((B, L, D), fill_value)
    # masked_scatter is the functional equivalent of `out[mask] = x_flat`.
    # The in-place boolean index-put triggers IndexPutBackward, which breaks
    # the torch.compile graph; masked_scatter does not.
    return out.masked_scatter(mask.unsqueeze(-1).expand(B, L, D), x_flat)


class MABVarlen(nn.Module):
    """
    Multihead Attention Block tailored for variable-length inputs (varlen).
    Utilizes flash_attn_varlen_func for memory-efficient attention over packed sequences.
    """

    def __init__(
        self, dim: int, num_heads: int, bias: bool = True, dim_feedforward: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = dim // num_heads
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * dim

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim, bias=bias),
        )

    def forward(self, q: "VarlenTensor", k: "VarlenTensor") -> "VarlenTensor":
        """
        Args:
            q: Query :class:`VarlenTensor` — ``(total_q, dim)`` packed data.
            k: Key/value :class:`VarlenTensor` — ``(total_k, dim)`` packed data.

        Returns:
            :class:`VarlenTensor` with the same sequence structure as ``q``.
        """
        q_n = self.norm1(q.data)
        k_src = q_n if k is q else k.data
        q_proj = self.q_proj(q_n).view(-1, self.num_heads, self.head_dim)
        k_proj = self.k_proj(k_src).view(-1, self.num_heads, self.head_dim)
        v_proj = self.v_proj(k_src).view(-1, self.num_heads, self.head_dim)

        if varlen_attn is None:
            raise NotImplementedError("torch.nn.attention.varlen.varlen_attn is not available in your PyTorch version.")

        attn_out = varlen_attn(
            q_proj,
            k_proj,
            v_proj,
            cu_seq_q=q.cu_seqlens,
            cu_seq_k=k.cu_seqlens,
            max_q=q.max_seqlen,
            max_k=k.max_seqlen,
        )

        attn_out = attn_out.view(-1, self.dim)
        out = q.data + self.dropout(self.out_proj(attn_out))
        out = out + self.dropout(self.ffn(self.norm2(out)))
        return VarlenTensor(out, q.cu_seqlens, q.max_seqlen)


class SABVarlen(nn.Module):
    """
    Self-Attention Block for varlen inputs.
    """

    def __init__(
        self, dim: int, num_heads: int, bias: bool = True, dim_feedforward: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        self.mab = MABVarlen(dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, vt: "VarlenTensor") -> "VarlenTensor":
        return self.mab(vt, vt)


class PMAVarlen(nn.Module):
    """
    Pooling by Multihead Attention tailored for varlen inputs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int = 1,
        bias: bool = True,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MABVarlen(dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, vt: "VarlenTensor") -> torch.Tensor:
        """
        Args:
            vt: :class:`VarlenTensor` of shape ``(total_elements, dim)``.

        Returns:
            Tensor of shape ``(batch_size * num_seeds, dim)``.
        """
        batch_size = vt.cu_seqlens.shape[0] - 1
        s = self.S.expand(batch_size, -1, -1).reshape(-1, self.S.shape[-1])
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * self.num_seeds, step=self.num_seeds, device=vt.data.device, dtype=vt.cu_seqlens.dtype
        )
        return self.mab(VarlenTensor(s, cu_seqlens_q, self.num_seeds), vt).data


class ISABVarlen(nn.Module):
    """
    Induced Self-Attention Block for varlen inputs.

    Reduces the quadratic cost of self-attention from O(n²) to O(mn) by routing
    through ``num_inducing`` learnable inducing points ``I``:

    .. code-block:: text

        H   = MAB(I, X)   # (num_inducing per example, dim)
        out = MAB(X, H)   # (n per example, dim)

    Both MAB calls operate on packed varlen sequences so no padding is needed.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inducing: int,
        bias: bool = True,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads.
            num_inducing: Number of inducing points ``m``.  Smaller values give
                more compression; ``m >= n`` recovers standard self-attention.
            dim_feedforward: FFN hidden dimension (both MAB blocks).  Defaults
                to ``4 * dim`` if not set.
            dropout: Dropout probability applied after attention output and FFN
                in each MAB block.  Default: ``0.0`` (disabled).
        """
        super().__init__()
        self.num_inducing = num_inducing
        self.I = nn.Parameter(torch.empty(1, num_inducing, dim))
        nn.init.xavier_uniform_(self.I.view(num_inducing, dim))  # treat as (m, dim) so fan_in=dim, not m*dim
        self.mab1 = MABVarlen(dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout)
        self.mab2 = MABVarlen(dim, num_heads, bias=bias, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, vt: "VarlenTensor") -> "VarlenTensor":
        """
        Args:
            vt: :class:`VarlenTensor` of shape ``(total_elements, dim)``.

        Returns:
            :class:`VarlenTensor` of the same shape as the input.
        """
        batch_size = vt.cu_seqlens.shape[0] - 1
        ind_data = self.I.expand(batch_size, -1, -1).reshape(-1, self.I.shape[-1])
        cu_seqlens_ind = torch.arange(
            0, (batch_size + 1) * self.num_inducing, step=self.num_inducing,
            dtype=vt.cu_seqlens.dtype, device=vt.cu_seqlens.device,
        )
        ind_vt = VarlenTensor(ind_data, cu_seqlens_ind, self.num_inducing)

        H_vt = self.mab1(ind_vt, vt)
        return self.mab2(vt, H_vt)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class SetEncoderVarlen(nn.Module):
    """
    Set encoder using stacked :class:`SABVarlen` blocks.

    Projects the raw input features to ``dim_hidden`` then applies ``num_sabs``
    Self-Attention Blocks.  Output is a packed tensor with the **same shape**
    as the (projected) input — one vector per set element — suitable for
    passing to :class:`SetPoolingVarlen` or any downstream module.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_heads: int,
        num_sabs: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim_in: Raw input feature dimension.
            dim_hidden: Hidden (output) dimension after the input projection.
            num_heads: Number of attention heads in each SAB.
            num_sabs: Number of stacked :class:`SABVarlen` blocks.
            dim_feedforward: FFN hidden dimension in each SAB.  Defaults to
                ``4 * dim_hidden`` if not set.
            dropout: Dropout probability for each SAB.  Default: ``0.0``.
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden)
        self.sabs = nn.ModuleList(
            [
                SABVarlen(dim_hidden, num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(num_sabs)
            ]
        )
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, vt: "VarlenTensor") -> "VarlenTensor":
        """
        Args:
            vt: :class:`VarlenTensor` with ``(total_elements, dim_in)`` data.

        Returns:
            :class:`VarlenTensor` with ``(total_elements, dim_hidden)`` data.
        """
        vt = VarlenTensor(self.proj(vt.data), vt.cu_seqlens, vt.max_seqlen)
        for sab in self.sabs:
            vt = sab(vt)
        return VarlenTensor(self.norm(vt.data), vt.cu_seqlens, vt.max_seqlen)


class SetEncoderVarlenPadded(nn.Module):
    """
    Convenience wrapper around :class:`SetEncoderVarlen` for callers that work
    with the standard padded ``(batch_size, max_len, dim_in)`` tensors produced
    by most DataLoaders.

    Internally calls :func:`padded_to_varlen` to build ``cu_seqlens`` and
    ``max_seqlen``, runs the encoder, then scatters the result back to a
    zero-padded ``(batch_size, max_len, dim_hidden)`` tensor via
    :func:`varlen_to_padded`.  The interface is therefore identical to a
    standard ``nn.Module`` that accepts ``(x, mask)`` — no varlen bookkeeping
    is required from the caller.

    Args:
        dim_in: Raw input feature dimension.
        dim_hidden: Hidden (output) dimension after the input projection.
        num_heads: Number of attention heads in each SAB.
        num_sabs: Number of stacked :class:`SABVarlen` blocks (default: 2).
        dim_feedforward: FFN hidden dimension in each SAB.  Defaults to
            ``4 * dim_hidden`` if not provided.
        dropout: Dropout probability for each SAB.  Default: ``0.0``.
        fill_value: Value written into padding positions of the output tensor
            (default ``0.0``).

    Example::

        encoder = SetEncoderVarlenPadded(dim_in=16, dim_hidden=64, num_heads=4)
        # x:    (B, L, 16) padded feature tensor
        # mask: (B, L)     bool tensor, True = valid element
        out = encoder(x, mask)   # (B, L, 64), padding positions are 0
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_heads: int,
        num_sabs: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        fill_value: float = 0.0,
    ):
        super().__init__()
        self.fill_value = fill_value
        self.encoder = SetEncoderVarlen(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_sabs=num_sabs,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(batch_size, max_len, dim_in)`` padded feature tensor.
            mask: ``(batch_size, max_len)`` boolean tensor where ``True``
                marks valid (non-padding) positions.

        Returns:
            ``(batch_size, max_len, dim_hidden)`` encoded tensor.  Padding
            positions are filled with ``self.fill_value`` (default ``0.0``).
        """
        enc_vt = self.encoder(padded_to_varlen(x, mask))
        return varlen_to_padded(enc_vt.data, mask, fill_value=self.fill_value)


class InducedSetEncoderVarlen(nn.Module):
    """
    Set encoder using stacked :class:`ISABVarlen` blocks.

    Identical interface to :class:`SetEncoderVarlen` but uses induced
    self-attention, reducing per-block complexity from O(n²) to O(mn).
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_heads: int,
        num_inducing: int = 32,
        num_isabs: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim_in: Raw input feature dimension.
            dim_hidden: Hidden (output) dimension after the input projection.
            num_heads: Number of attention heads in each ISAB.
            num_inducing: Number of inducing points ``m`` per ISAB.
            num_isabs: Number of stacked :class:`ISABVarlen` blocks.
            dim_feedforward: FFN hidden dimension in each ISAB.  Defaults to
                ``4 * dim_hidden`` if not set.
            dropout: Dropout probability for each ISAB.  Default: ``0.0``.
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_hidden)
        self.isabs = nn.ModuleList(
            [
                ISABVarlen(
                    dim=dim_hidden,
                    num_heads=num_heads,
                    num_inducing=num_inducing,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_isabs)
            ]
        )
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, vt: "VarlenTensor") -> "VarlenTensor":
        """
        Args:
            vt: :class:`VarlenTensor` with ``(total_elements, dim_in)`` data.

        Returns:
            :class:`VarlenTensor` with ``(total_elements, dim_hidden)`` data.
        """
        vt = VarlenTensor(self.proj(vt.data), vt.cu_seqlens, vt.max_seqlen)
        for isab in self.isabs:
            vt = isab(vt)
        return VarlenTensor(self.norm(vt.data), vt.cu_seqlens, vt.max_seqlen)


# ---------------------------------------------------------------------------
# Fixed-size (B, N, dim) attention blocks
# ---------------------------------------------------------------------------


class MAB(nn.Module):
    """Multihead Attention Block for fixed-size ``(B, Q, dim)`` / ``(B, K, dim)`` inputs.

    Uses ``F.scaled_dot_product_attention`` (auto-dispatches to Flash Attention or
    memory-efficient attention) with pre-norm residuals.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self._dropout = dropout
        dim_feedforward = dim_feedforward or 4 * dim

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim, bias=bias),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: ``(B, Q, dim)`` query tensor.
            k: ``(B, K, dim)`` key/value tensor.

        Returns:
            ``(B, Q, dim)`` updated query tensor.
        """
        B, Q, C = q.shape

        q_n = self.norm1(q)
        # For self-attention (k is q), project k/v from the already-normed q_n so that
        # both sides of the attention are consistent (pre-norm applied uniformly).
        k_src = q_n if k is q else k
        q_proj = self.q_proj(q_n).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(k_src).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(k_src).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q_proj,
            k_proj,
            v_proj,
            dropout_p=self._dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, Q, C)
        q = q + self.drop(self.out_proj(attn_out))
        q = q + self.drop(self.ffn(self.norm2(q)))
        return q


class ISAB(nn.Module):
    """Fixed-size Induced Self-Attention Block.

    Reduces ``O(N²)`` self-attention to ``O(Nm)`` via ``m`` learnable inducing points::

        H   = MAB(I, X)   # (B, m, dim)
        out = MAB(X, H)   # (B, N, dim)

    Input and output are both dense ``(B, N, dim)`` tensors.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inducing: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads.
            num_inducing: Number of inducing points ``m``.
            dim_feedforward: FFN hidden dimension.  Defaults to ``4 * dim``.
            dropout: Dropout probability.
        """
        super().__init__()
        self.I = nn.Embedding(num_inducing, dim)
        self.mab1 = MAB(dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias)
        self.mab2 = MAB(dim=dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, dim)`` input set.

        Returns:
            ``(B, N, dim)`` updated set.
        """
        ind = self.I.weight.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, m, dim)
        H = self.mab1(ind, x)
        return self.mab2(x, H)


class InducedSetEncoder(nn.Module):
    """Fixed-size induced set encoder: stacks :class:`ISAB` blocks on ``(B, N, dim)`` inputs.

    Unlike :class:`InducedSetEncoderVarlen`, all sequences in the batch must have
    the same length (e.g. after aggregating hits to a fixed PMT grid via
    ``segment_reduce``).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inducing: int,
        num_isabs: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads per ISAB.
            num_inducing: Number of inducing points ``m``.
            num_isabs: Number of stacked :class:`ISAB` blocks.
            dim_feedforward: FFN hidden dimension.  Defaults to ``4 * dim``.
            dropout: Dropout probability.
        """
        super().__init__()
        self.isabs = nn.ModuleList(
            [
                ISAB(
                    dim=dim,
                    num_heads=num_heads,
                    num_inducing=num_inducing,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_isabs)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N, dim)`` input set.

        Returns:
            ``(B, N, dim)`` encoded set.
        """
        for isab in self.isabs:
            x = isab(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class SetPoolingVarlen(nn.Module):
    """
    Pooling head for varlen set encoders.

    Applies :class:`PMAVarlen` to aggregate per-element representations into a
    fixed-size per-set vector, then passes it through an MLP decoder.

    Can be paired with either :class:`SetEncoderVarlen` or
    :class:`InducedSetEncoderVarlen` — or any encoder that produces a packed
    ``(total_elements, dim_hidden)`` output.
    """

    def __init__(
        self,
        dim_hidden: int,
        dim_out: int,
        num_heads: int,
        num_seeds: int = 1,
        dim_feedforward: int | None = None,
    ):
        """
        Args:
            dim_hidden: Feature dimension coming from the encoder.
            dim_out: Final output dimension.
            num_heads: Number of attention heads in the PMA.
            num_seeds: Number of seed vectors; controls the aggregation width.
            dim_feedforward: FFN hidden dimension in the PMA block.  Defaults
                to ``4 * dim_hidden`` if not set.
        """
        super().__init__()
        self.pma = PMAVarlen(dim=dim_hidden, num_heads=num_heads, num_seeds=num_seeds, dim_feedforward=dim_feedforward)
        self.dec = nn.Sequential(
            nn.Linear(num_seeds * dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, vt: "VarlenTensor") -> torch.Tensor:
        """
        Args:
            vt: :class:`VarlenTensor` with ``(total_elements, dim_hidden)`` data.

        Returns:
            ``(batch_size, dim_out)`` per-set predictions.
        """
        batch_size = vt.cu_seqlens.shape[0] - 1
        x = self.pma(vt).view(batch_size, -1)
        return self.dec(x)


# ---------------------------------------------------------------------------
# Full models (encoder + pooling composed)
# ---------------------------------------------------------------------------


class SetTransformerVarlen(nn.Module):
    """
    Set Transformer for varlen inputs.

    Composes :class:`SetEncoderVarlen` and :class:`SetPoolingVarlen`.
    Access the sub-modules directly for fine-grained control::

        model.encoder   # SetEncoderVarlen
        model.pooling   # SetPoolingVarlen
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_heads: int,
        num_sabs: int = 2,
        num_seeds: int = 1,
        dim_feedforward: int | None = None,
    ):
        """
        Args:
            dim_in: Input feature dimension.
            dim_hidden: Hidden representation dimension.
            dim_out: Output dimension.
            num_heads: Number of attention heads.
            num_sabs: Number of Self-Attention Blocks.
            num_seeds: Number of PMA seed vectors.
            dim_feedforward: FFN hidden dimension for every attention block.
                Defaults to ``4 * dim_hidden`` if not set.
        """
        super().__init__()
        self.encoder = SetEncoderVarlen(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_sabs=num_sabs,
            dim_feedforward=dim_feedforward,
        )
        self.pooling = SetPoolingVarlen(
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_heads=num_heads,
            num_seeds=num_seeds,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, vt: "VarlenTensor") -> torch.Tensor:
        """
        Args:
            vt: :class:`VarlenTensor` with ``(total_elements, dim_in)`` data.

        Returns:
            ``(batch_size, dim_out)`` per-set predictions.
        """
        return self.pooling(self.encoder(vt))


class InducedSetTransformerVarlen(nn.Module):
    """
    Induced Set Transformer for varlen inputs.

    Composes :class:`InducedSetEncoderVarlen` and :class:`SetPoolingVarlen`.
    Access the sub-modules directly for fine-grained control::

        model.encoder   # InducedSetEncoderVarlen
        model.pooling   # SetPoolingVarlen
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_heads: int,
        num_inducing: int = 32,
        num_isabs: int = 2,
        num_seeds: int = 1,
        dim_feedforward: int | None = None,
    ):
        """
        Args:
            dim_in: Input feature dimension.
            dim_hidden: Hidden representation dimension.
            dim_out: Output dimension.
            num_heads: Number of attention heads.
            num_inducing: Number of inducing points ``m`` per ISAB.
            num_isabs: Number of Induced Self-Attention Blocks.
            num_seeds: Number of PMA seed vectors.
            dim_feedforward: FFN hidden dimension for every attention block.
                Defaults to ``4 * dim_hidden`` if not set.
        """
        super().__init__()
        self.encoder = InducedSetEncoderVarlen(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_isabs=num_isabs,
            dim_feedforward=dim_feedforward,
        )
        self.pooling = SetPoolingVarlen(
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_heads=num_heads,
            num_seeds=num_seeds,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, vt: "VarlenTensor") -> torch.Tensor:
        """
        Args:
            vt: :class:`VarlenTensor` with ``(total_elements, dim_in)`` data.

        Returns:
            ``(batch_size, dim_out)`` per-set predictions.
        """
        return self.pooling(self.encoder(vt))
