"""
SparseTransformerCrossBase adapted from LATO.

Provides a multi-block sparse transformer with cross-attention support,
using TRELLIS's existing SparseTransformerCrossBlock and modules.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn

from trellis.modules import sparse as sp
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.sparse.transformer import SparseTransformerCrossBlock
from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32


def block_attn_config(self):
    """
    Return the attention configuration of the model.
    Adapted from LATO's block_attn_config generator.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerCrossBase(nn.Module):
    """
    Sparse Transformer with cross-attention, without output layers.

    Adapted from LATO: lato/models/lato_vae/base.py

    Serves as the base class for decoders that need to cross-attend
    to a conditioning context (e.g., the original VAE latent).

    Args:
        in_channels: Input feature channels.
        model_channels: Internal model channels.
        context_channels: Channels of the cross-attention context.
        num_blocks: Number of transformer blocks.
        num_heads: Number of attention heads (derived from num_head_channels if None).
        num_head_channels: Channels per head (default 64).
        mlp_ratio: MLP hidden dimension ratio.
        attn_mode: Attention mode ('full', 'swin', 'shift_window', etc.).
        window_size: Window size for windowed attention.
        pe_mode: Positional encoding mode ('ape' or 'rope').
        use_fp16: Whether to use FP16 internally.
        use_checkpoint: Whether to use activation checkpointing.
        qk_rms_norm: Whether to use RMS norm on Q/K.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        context_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape":
            self.pos_embedder_x = AbsolutePositionEmbedder(model_channels)
            self.pos_embedder_ctx = AbsolutePositionEmbedder(context_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerCrossBlock(
                model_channels,
                context_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """Convert the torso of the model to float16."""
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert the torso of the model to float32."""
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        """Initialize transformer layers with Xavier uniform."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, context: sp.SparseTensor) -> sp.SparseTensor:
        """
        Forward pass with self-attention on x and cross-attention to context.

        Args:
            x: Input sparse tensor [N x in_channels].
            context: Context sparse tensor for cross-attention [M x context_channels].

        Returns:
            Output sparse tensor [N x model_channels].
        """
        h = self.input_layer(x)
        if self.pe_mode == "ape" and len(self.blocks) != 0:
            h = h + self.pos_embedder_x(x.coords[:, 1:])
            context = context + self.pos_embedder_ctx(context.coords[:, 1:])
        for block in self.blocks:
            h = block(h, context)
        return h
