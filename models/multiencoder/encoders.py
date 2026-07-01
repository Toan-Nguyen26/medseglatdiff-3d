"""
Four modality-specific CNN encoders with element-wise sum fusion.

Each encoder takes a single MRI modality (B, 1, D, H, W) and produces a
feature map at 1/8 spatial resolution (B, embed_dim, D/8, H/8, W/8) via
three stride-2 downsampling levels. Missing modalities are already zeroed
out in the input by apply_modality_mask() before this module is called —
their encoder outputs ~0 and contribute nothing to the sum.

The fused output (B, embed_dim, D/8, H/8, W/8) is the conditioning signal
for the diffusion U-Net, replacing the Uni-Encoder in the simplified pipeline.
At the default 96^3 crop size, output spatial shape is (12, 12, 12), matching
the mask VAE latent grid.
"""

import torch
from torch import nn

from models.mask_vae.blocks import Downsample3D, ResBlock3D


class ModalityEncoder(nn.Module):
    """
    Single-modality 3D CNN encoder.

    (B, 1, D, H, W) -> (B, embed_dim, D/8, H/8, W/8)

    Three ResBlock+Downsample levels give 8x spatial reduction.
    base_channels controls the intermediate channel width; embed_dim is the
    final output channel count (= the diffusion conditioning dimension).
    """

    def __init__(self, embed_dim: int = 256, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Conv3d(1, c, kernel_size=3, padding=1)
        # level 1: c -> 2c, spatial /2
        self.res1 = ResBlock3D(c, c * 2)
        self.down1 = Downsample3D(c * 2)
        # level 2: 2c -> 4c, spatial /2
        self.res2 = ResBlock3D(c * 2, c * 4)
        self.down2 = Downsample3D(c * 4)
        # level 3: 4c -> embed_dim, spatial /2
        self.res3 = ResBlock3D(c * 4, embed_dim)
        self.down3 = Downsample3D(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(self.res1(x))
        x = self.down2(self.res2(x))
        x = self.down3(self.res3(x))
        return x


class MultiModalEncoder(nn.Module):
    """
    Four independent modality encoders fused by element-wise sum.

    Input:  (B, 4, D, H, W) — channel order FLAIR/T1ce/T1/T2, missing
            modalities already zeroed by apply_modality_mask().
    Output: (B, embed_dim, D/8, H/8, W/8)

    Why sum not concat:
    - Missing modality -> that encoder outputs ~zero -> contributes nothing.
    - Sum is permutation-invariant and handles any subset of modalities without
      special-casing or variable-length inputs.
    - All 15 non-empty modality combinations are seen during training via
      sample_modality_mask(), so the encoders learn to produce useful features
      regardless of which modalities are present.
    """

    SPATIAL_DOWNSAMPLE_FACTOR: int = 8  # 3 stride-2 levels

    def __init__(
        self,
        embed_dim: int = 256,
        base_channels: int = 32,
        num_modalities: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.encoders = nn.ModuleList(
            [ModalityEncoder(embed_dim, base_channels) for _ in range(num_modalities)]
        )

    def output_spatial_shape(self, crop_size: int) -> tuple[int, int, int]:
        s = crop_size // self.SPATIAL_DOWNSAMPLE_FACTOR
        return (s, s, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_modalities, D, H, W) — missing modalities zeroed out
        Returns:
            (B, embed_dim, D/8, H/8, W/8)
        """
        features = [enc(x[:, i : i + 1]) for i, enc in enumerate(self.encoders)]
        return sum(features)  # type: ignore[return-value]
