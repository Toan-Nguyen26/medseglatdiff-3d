import torch
from torch import nn

from models.mask_vae.blocks import Downsample3D, ResBlock3D
from models.mask_vae.config import MaskVAEConfig


class MaskEncoder3D(nn.Module):
    """
    One-hot mask volume -> (mu, logvar) over the latent grid.
    len(channel_multipliers) downsampling levels, each halving spatial
    resolution -> total downsample = 2 ** num_levels.
    """

    def __init__(self, config: MaskVAEConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * m for m in config.channel_multipliers]

        self.stem = nn.Conv3d(config.num_classes, channels[0], kernel_size=3, padding=1)

        # Downsample after every level (including the last) so that
        # len(channel_multipliers) levels => 2**len(channel_multipliers) total
        # downsampling, matching MaskVAEConfig.downsample_factor.
        levels: list[nn.Module] = []
        in_ch = channels[0]
        for out_ch in channels:
            blocks = [
                ResBlock3D(in_ch if i == 0 else out_ch, out_ch, groups=config.group_norm_groups)
                for i in range(config.num_res_blocks_per_level)
            ]
            blocks.append(Downsample3D(out_ch))
            levels.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.levels = nn.ModuleList(levels)

        final_ch = channels[-1]
        self.norm_out = nn.GroupNorm(min(config.group_norm_groups, final_ch), final_ch)
        self.act_out = nn.SiLU()
        # 2x latent_channels: mean and log-variance, standard VAE reparameterization
        self.to_moments = nn.Conv3d(final_ch, 2 * config.latent_channels, kernel_size=1)

    def forward(self, mask_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mask_onehot: (B, num_classes, D, H, W)
        Returns:
            mu, logvar: each (B, latent_channels, D/f, H/f, W/f)
        """
        x = self.stem(mask_onehot)
        for level in self.levels:
            x = level(x)
        x = self.act_out(self.norm_out(x))
        moments = self.to_moments(x)
        mu, logvar = moments.chunk(2, dim=1)
        return mu, logvar
