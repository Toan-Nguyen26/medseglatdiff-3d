from typing import Tuple

import torch
from torch import nn


class Tokenizer3D(nn.Module):
    """Single strided 3D conv: turns each non-overlapping patch into one token."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: Tuple[int, int, int] = (8, 8, 8)) -> None:
        super().__init__()
        k = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=k, stride=k, padding=0, bias=True)
        self.patch = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, embed_dim, D//patch, H//patch, W//patch)
        """
        return self.proj(x)
