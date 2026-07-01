from typing import Tuple

from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from models.uniencoder.rope_3d import apply_rope
from models.uniencoder.utils import DropPath, SwiGLU


class MHSA3D(nn.Module):
    """Multi-head self-attention for 3D token sequences, with 3D RoPE support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.num_register_tokens = num_register_tokens

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.post_attn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
        spatial_shape: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        qkv = rearrange(
            self.qkv(x), "b l (three h d) -> three b h l d", three=3, h=self.num_heads, d=self.head_dim
        )
        q, k, v = qkv.unbind(0)

        if rope_embed is not None:
            assert spatial_shape is not None, "spatial_shape must be provided when using RoPE3D."
            q, k = apply_rope(q, k, rope_embed, self.num_register_tokens)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=False,
        )
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)").contiguous()

        attn_out = self.post_attn_norm(attn_out)
        attn_out = self.proj_drop(self.proj(attn_out))
        return attn_out


class EVABlock(nn.Module):
    """
    EVA-02 style transformer block: pre-norm MHSA (with RoPE3D) + pre-norm SwiGLU MLP,
    each with LayerScale and DropPath on the residual branch.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dropout: float,
        attn_dropout: float,
        proj_dropout: float,
        drop_path: float,
        layer_scale_init_value: float | None,
        num_register_tokens: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHSA3D(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            num_register_tokens=num_register_tokens,
        )
        self.drop_path1 = DropPath(drop_path)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim=dim, dropout=mlp_dropout)
        self.mlp.init_weights()
        self.gamma1: nn.Parameter | None = None
        self.gamma2: nn.Parameter | None = None
        if layer_scale_init_value is not None:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
        spatial_shape: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x), rope_embed=rope_embed, spatial_shape=spatial_shape)
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x + self.drop_path1(attn_out)

        mlp_out = self.mlp(self.ln2(x))
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        x = x + self.drop_path2(mlp_out)
        return x
