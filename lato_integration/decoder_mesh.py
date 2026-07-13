"""
Enhanced Mesh Decoder with cross-attention, occupancy pruning, and edge prediction.

Inherits from TRELLIS's SLatMeshDecoder and adds the most impactful LATO
enhancements: cross-attention to original latent, occupancy-guided pruning
at each subdivision level, and a ConnectionHead for direct edge prediction.
"""

from typing import List, Literal, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.structured_latent_vae.decoder_mesh import (
    SLatMeshDecoder as _SLatMeshDecoder,
    SparseSubdivideBlock3d as _SparseSubdivideBlock3d,
)
from trellis.models.structured_latent_vae.decoder_mesh import (
    ElasticSLatMeshDecoder as _ElasticSLatMeshDecoder,
)
from trellis.modules import sparse as sp
from trellis.modules.sparse.linear import SparseLinear
from trellis.modules.sparse.nonlinearity import SparseGELU
from trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from trellis.representations import MeshExtractResult
from trellis.models.sparse_elastic_mixin import SparseTransformerElasticMixin

from .base import SparseTransformerCrossBase
from .vertex_encoder import ConnectionHead


class SparsePredictionHead(nn.Module):
    """
    Small MLP head for occupancy prediction on sparse tensors.

    Adapted from LATO: lato/models/lato_vae/lato_vae.py lines 16-27.

    Args:
        channels: Input feature channels.
        out_channels: Output channels (typically 1 for occupancy).
        mlp_ratio: Hidden layer multiplier.
    """

    def __init__(self, channels: int, out_channels: int = 1, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            SparseLinear(channels, hidden_channels),
            SparseGELU(approximate="tanh"),
            SparseLinear(hidden_channels, out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)


class EnhancedSparseSubdivideBlock3d(_SparseSubdivideBlock3d):
    """
    Enhanced 3D subdivision block with occupancy-based pruning.

    Extends TRELLIS's SparseSubdivideBlock3d with LATO-style occupancy
    prediction for pruning non-surface voxels at each resolution level.

    Args:
        (same as SparseSubdivideBlock3d)
        use_pruning: Whether to add occupancy pruning.
    """

    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
        use_pruning: bool = True,
    ):
        super().__init__(
            channels=channels,
            resolution=resolution,
            out_channels=out_channels,
            num_groups=num_groups,
        )
        self.use_pruning = use_pruning
        if use_pruning:
            self.pruning_head = SparsePredictionHead(
                self.out_channels, out_channels=1
            )

    def forward(
        self,
        x: sp.SparseTensor,
        pruning: bool = False,
        training: bool = True,
        threshold: float = 0.5,
        force_no_prune: bool = False,
    ) -> Tuple[sp.SparseTensor, Optional[sp.SparseTensor]]:
        """
        Apply subdivision with optional occupancy pruning.

        Args:
            x: Input sparse tensor.
            pruning: Whether to apply pruning.
            training: Whether in training mode (affects pruning behavior).
            threshold: Occupancy threshold for pruning during inference.
            force_no_prune: Force no pruning (for final layer visualization).

        Returns:
            Tuple of (output sparse tensor, optional occupancy probabilities).
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)

        if not pruning or not self.use_pruning:
            return h, None

        occ_prob = self.pruning_head(h)

        if not training and not force_no_prune:
            scores = torch.sigmoid(occ_prob.feats).squeeze(-1)
            # LATO-style: ensure at least one child per parent survives
            if scores.numel() % 8 == 0:
                grouped_scores = scores.view(-1, 8)
                grouped_mask = grouped_scores >= threshold
                empty_parent_mask = grouped_mask.sum(dim=1) == 0
                if empty_parent_mask.any():
                    _, top_indices = torch.topk(
                        grouped_scores[empty_parent_mask], k=1, dim=1
                    )
                    parent_rows = torch.nonzero(empty_parent_mask, as_tuple=True)[0].unsqueeze(1)
                    grouped_mask[parent_rows, top_indices] = True
                occ_mask = grouped_mask.view(-1)
            else:
                occ_mask = scores >= threshold

            h = sp.SparseTensor(
                feats=h.feats[occ_mask],
                coords=h.coords[occ_mask],
            )

        return h, occ_prob


class EnhancedSLatMeshDecoder(_SLatMeshDecoder):
    """
    Enhanced SLat Mesh Decoder with cross-attention, pruning, and edge prediction.

    The most substantial enhancement over the original SLatMeshDecoder:
    1. Cross-attention to the original latent for better feature propagation
    2. Occupancy-guided pruning at each subdivision level (LATO-style)
    3. ConnectionHead for direct edge/topology prediction

    Args:
        (same as SLatMeshDecoder)
        use_cross_attn: Whether to enable cross-attention to latent.
        cross_attn_num_blocks: Number of cross-attention blocks.
        use_pruning: Whether to add occupancy pruning at subdivision levels.
        use_edge_pred: Whether to add ConnectionHead for edge prediction.
    """

    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        use_cross_attn: bool = True,
        cross_attn_num_blocks: int = 2,
        use_pruning: bool = True,
        use_edge_pred: bool = True,
    ):
        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
        )
        self.use_cross_attn = use_cross_attn
        self.use_pruning = use_pruning
        self.use_edge_pred = use_edge_pred
        self._latent_channels = latent_channels

        # Replace upsample blocks with enhanced versions that have pruning
        if use_pruning:
            self.upsample = nn.ModuleList([
                EnhancedSparseSubdivideBlock3d(
                    channels=model_channels,
                    resolution=resolution,
                    out_channels=model_channels // 4,
                    num_groups=32,
                    use_pruning=True,
                ),
                EnhancedSparseSubdivideBlock3d(
                    channels=model_channels // 4,
                    resolution=resolution * 2,
                    out_channels=model_channels // 8,
                    num_groups=32,
                    use_pruning=True,
                ),
            ])

        if use_cross_attn:
            # Project latent to context dimension for cross-attention
            self.latent_proj = sp.SparseLinear(latent_channels, model_channels)

            # Cross-attention block(s): after the main transformer body,
            # attend back to the original latent
            self.cross_attn = SparseTransformerCrossBase(
                in_channels=model_channels,
                model_channels=model_channels,
                context_channels=model_channels,
                num_blocks=cross_attn_num_blocks,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                pe_mode=pe_mode,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,
            )

        if use_edge_pred:
            # ConnectionHead for edge prediction between vertex pairs
            # Input is 2 * out_channels (concatenated pair features)
            self.connection_head = ConnectionHead(
                channels=self.out_channels,
                out_channels=1,
                mlp_ratio=2.0,
            )

        if use_fp16:
            if use_cross_attn:
                self.cross_attn.convert_to_fp16()
                self.latent_proj.apply(
                    lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
                )

    def forward(
        self,
        x: sp.SparseTensor,
        original_latent: Optional[sp.SparseTensor] = None,
        return_pruning: bool = False,
    ) -> List[MeshExtractResult]:
        """
        Forward pass with cross-attention and occupancy pruning.

        Args:
            x: Input latent sparse tensor.
            original_latent: The original VAE latent to cross-attend to.
            return_pruning: If True, also return pruning occupancy predictions.

        Returns:
            List of MeshExtractResult, or (results, pruning_data) if return_pruning=True.
        """
        h = super(_SLatMeshDecoder, self).forward(x)  # SparseTransformerBase.forward

        # Apply cross-attention to original latent
        if self.use_cross_attn and original_latent is not None:
            ctx = self.latent_proj(original_latent)
            h = self.cross_attn(h, ctx)

        pruning_data = []
        for block in self.upsample:
            if self.use_pruning:
                h, occ_prob = block(
                    h,
                    pruning=True,
                    training=self.training,
                    threshold=0.2,  # LATO default inference threshold
                )
                if return_pruning:
                    pruning_data.append(occ_prob)
            else:
                h = block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)
        results = self.to_representation(h)

        if return_pruning:
            return results, pruning_data
        return results

    def predict_edges(
        self,
        vertex_feats: torch.Tensor,
        edge_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict edge probabilities between candidate vertex pairs.

        Following LATO's approach: for each candidate pair (u, v),
        run ConnectionHead([feat_u | feat_v]) + ConnectionHead([feat_v | feat_u]),
        then apply sigmoid.

        Args:
            vertex_feats: Vertex features [N, C].
            edge_candidates: Candidate edge pairs [E, 2] (indices into vertex_feats).

        Returns:
            Edge probabilities [E].
        """
        if not self.use_edge_pred:
            return torch.ones(edge_candidates.shape[0], device=vertex_feats.device)

        u_feats = vertex_feats[edge_candidates[:, 0]]
        v_feats = vertex_feats[edge_candidates[:, 1]]

        # Bidirectional prediction (LATO approach)
        logits_uv = self.connection_head(torch.cat([u_feats, v_feats], dim=-1))
        logits_vu = self.connection_head(torch.cat([v_feats, u_feats], dim=-1))

        edge_probs = torch.sigmoid(logits_uv + logits_vu).squeeze(-1)
        return edge_probs

    def convert_to_fp16(self) -> None:
        """Convert the torso of the model to float16."""
        super().convert_to_fp16()
        if self.use_cross_attn:
            self.cross_attn.convert_to_fp16()
            self.latent_proj.apply(
                lambda m: m.convert_to_fp16() if hasattr(m, "convert_to_fp16") else None
            )

    def convert_to_fp32(self) -> None:
        """Convert the torso of the model to float32."""
        super().convert_to_fp32()
        if self.use_cross_attn:
            self.cross_attn.convert_to_fp32()
            self.latent_proj.apply(
                lambda m: m.convert_to_fp32() if hasattr(m, "convert_to_fp32") else None
            )


class EnhancedElasticSLatMeshDecoder(SparseTransformerElasticMixin, EnhancedSLatMeshDecoder):
    """
    Enhanced SLat Mesh Decoder with elastic memory management, cross-attention,
    and occupancy pruning.
    """
    pass
