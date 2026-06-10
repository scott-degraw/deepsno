import pandas as pd
import torch
from torch import nn

from deepsno.models.transformers import VarlenTensor


class PointNetVarlen(nn.Module):
    """
    PointNet-style per-point encoder for varlen packed sequences.

    Applies a shared local MLP to each point, global max-pools per event
    (varlen-aware via ``torch.segment_reduce``), concatenates the global feature
    back to each local feature, then applies an output MLP.

    Interface is identical to :class:`~deepsno.models.transformers.SetEncoderVarlen`:
    accepts packed ``(total_hits, dim_in)`` and returns ``(total_hits, dim_hidden)``.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_local_layers: int = 3,
        num_output_layers: int = 3,
        dropout: float = 0.0,
    ):
        """
        Args:
            dim_in: Input feature dimension (e.g. 4 for (x, y, z, t)).
            dim_hidden: Hidden and output dimension.
            num_local_layers: Depth of the per-point local MLP.
            num_output_layers: Depth of the output MLP (after global concat).
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()

        def _mlp(in_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(n_layers):
                layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.local_mlp = _mlp(dim_in, dim_hidden, num_local_layers)
        self.output_mlp = _mlp(dim_hidden * 2, dim_hidden, num_output_layers)

    def forward(self, vt: VarlenTensor) -> torch.Tensor:
        """
        Args:
            vt: :class:`~deepsno.models.transformers.VarlenTensor` with
                ``(total_hits, dim_in)`` packed point cloud.

        Returns:
            ``(total_hits, dim_hidden)`` per-point feature tensor.
        """
        lengths = (vt.cu_seqlens[1:] - vt.cu_seqlens[:-1]).to(torch.int32)
        x = vt.data

        local_feat = self.local_mlp(x)

        # Per-event global max pool: (B, dim_hidden)
        global_feat = torch.segment_reduce(local_feat, reduce="max", lengths=lengths, axis=0)

        # Expand back to per-hit: (total_hits, dim_hidden)
        global_feat_expanded = global_feat.repeat_interleave(lengths, dim=0)

        combined = torch.cat([local_feat, global_feat_expanded], dim=-1)
        return self.output_mlp(combined)


class MultiHitPMTEncoderPointNet(nn.Module):
    """
    PMT hit encoder that treats each hit as a 4D point (x, y, z, t) and
    encodes the resulting point cloud with a :class:`PointNetVarlen`.

    PMT positions are loaded from a CSV file (``pmt_info.csv``) and stored as a
    fixed buffer indexed by PMT ID.  Hit times are log-normalised with the same
    convention as :class:`~deepsno.models.multihit.MultiHitPMTEncoderBase`.

    Drop-in encoder for :class:`~deepsno.models.multihit.MultiHit`: exposes the
    same ``forward`` signature and return type.
    """

    def __init__(
        self,
        pmt_positions_path: str,
        encoder: nn.Module,
        time_scale: float = 1.0,
        time_shift: float = 0.0,
        position_scale: float = 1.0,
    ):
        """
        Args:
            pmt_positions_path: Path to ``pmt_info.csv`` with columns
                ``position_x``, ``position_y``, ``position_z`` indexed by ``pmt_id``.
            encoder: A :class:`PointNetVarlen` instance mapping
                ``(total_hits, 4)`` → ``(total_hits, model_dim)``.
            time_scale: Denominator for log-time normalisation.
            time_shift: Value subtracted from ``log(hit_time)`` before dividing.
            position_scale: All three spatial coordinates are divided by this
                value (e.g. detector radius in mm).
        """
        super().__init__()

        df = pd.read_csv(pmt_positions_path, index_col="pmt_id")
        positions = torch.tensor(
            df[["position_x", "position_y", "position_z"]].values,
            dtype=torch.float32,
        )
        self.register_buffer("pmt_positions", positions)

        self.encoder = encoder
        self.time_scale = time_scale
        self.time_shift = time_shift
        self.position_scale = position_scale

    def _normalize_time(self, hit_times: torch.Tensor) -> torch.Tensor:
        return (torch.log(hit_times) - self.time_shift) / self.time_scale

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
        positions = self.pmt_positions[hits.data] / self.position_scale  # (total_hits, 3)
        times = self._normalize_time(hit_times).unsqueeze(-1)  # (total_hits, 1)
        x = torch.cat([positions, times], dim=-1)  # (total_hits, 4)

        return hits._replace(data=self.encoder(hits._replace(data=x)))
