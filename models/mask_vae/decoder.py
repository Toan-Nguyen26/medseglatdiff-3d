import torch
from torch import nn

from models.mask_vae.blocks import ResBlock3D, Upsample3D
from models.mask_vae.config import MaskVAEConfig


class MaskDecoder3D(nn.Module):
    """Latent grid -> reconstructed mask logits, (B, num_classes, D, H, W).

    Output is raw logits (no softmax) — the WCE reconstruction loss applies
    cross-entropy directly, matching the small/sparse-structure emphasis used
    for the mask VAE in MedSegLatDiff.
    """

    def __init__(self, config: MaskVAEConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * m for m in config.channel_multipliers]
        reversed_channels = list(reversed(channels))

        self.from_latent = nn.Conv3d(config.latent_channels, reversed_channels[0], kernel_size=1)

        # Upsample after every level, mirroring the encoder's downsample-every-level
        # so the round trip is spatially symmetric (2**len(channel_multipliers) total).
        levels: list[nn.Module] = []
        in_ch = reversed_channels[0]
        for out_ch in reversed_channels:
            blocks = [
                ResBlock3D(in_ch if i == 0 else out_ch, out_ch, groups=config.group_norm_groups)
                for i in range(config.num_res_blocks_per_level)
            ]
            blocks.append(Upsample3D(out_ch))
            levels.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.levels = nn.ModuleList(levels)

        final_ch = reversed_channels[-1]
        self.norm_out = nn.GroupNorm(min(config.group_norm_groups, final_ch), final_ch)
        self.act_out = nn.SiLU()
        self.to_logits = nn.Conv3d(final_ch, config.num_classes, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, D/f, H/f, W/f)
        Returns:
            logits: (B, num_classes, D, H, W)
        """
        x = self.from_latent(z)
        for level in self.levels:
            x = level(x)
        x = self.act_out(self.norm_out(x))
        return self.to_logits(x)
