import torch
from torch import nn

from models.diffusion.blocks import TimestepResBlock3D
from models.diffusion.config import DiffusionConfig, PixelDiffusionConfig
from models.diffusion.time_embedding import TimeEmbedding
from models.mask_vae.blocks import Downsample3D, Upsample3D


class UNet3D(nn.Module):
    """
    Noise predictor ε_θ(z_cond, t) operating in the (already-compressed) mask
    latent space.

    Conditioning: the Uni-Encoder's feature grid is projected (1x1x1 conv) down
    to `cond_proj_channels` and concatenated channel-wise with the noisy mask
    latent at the input. The condition is NOT noised and NOT a function of t —
    it's injected once at the input, same as MedSegLatDiff's `z_cond = z_S,t ⊕
    z̄_X` (Eq. 11). Timestep conditioning happens separately, via additive
    embedding inside every TimestepResBlock3D.

    Internal U-Net hierarchy downsamples len(channel_multipliers)-1 times
    (standard U-Net: no downsample after the bottleneck level) — this is on
    top of the spatial compression the mask VAE and Uni-Encoder already did,
    not a replacement for it.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config

        self.cond_proj = nn.Conv3d(config.condition_channels, config.cond_proj_channels, kernel_size=1)
        self.time_embedding = TimeEmbedding(config.time_emb_dim)

        in_ch = config.latent_channels + config.cond_proj_channels
        channels = [config.base_channels * m for m in config.channel_multipliers]

        self.conv_in = nn.Conv3d(in_ch, channels[0], kernel_size=3, padding=1)

        # --- down path ---
        down_blocks: list[nn.ModuleList] = []
        downsamples: list[nn.Module] = []
        cur_ch = channels[0]
        for level_idx, out_ch in enumerate(channels):
            blocks = nn.ModuleList(
                [
                    TimestepResBlock3D(
                        cur_ch if i == 0 else out_ch, out_ch, config.time_emb_dim, groups=config.group_norm_groups
                    )
                    for i in range(config.num_res_blocks_per_level)
                ]
            )
            down_blocks.append(blocks)
            cur_ch = out_ch
            is_last_level = level_idx == len(channels) - 1
            downsamples.append(Downsample3D(cur_ch) if not is_last_level else nn.Identity())
        self.down_blocks = nn.ModuleList(down_blocks)
        self.downsamples = nn.ModuleList(downsamples)

        # --- bottleneck ---
        self.bottleneck = nn.ModuleList(
            [
                TimestepResBlock3D(cur_ch, cur_ch, config.time_emb_dim, groups=config.group_norm_groups)
                for _ in range(2)
            ]
        )

        # --- up path (mirrors down path, with skip connections) ---
        reversed_channels = list(reversed(channels))
        up_blocks: list[nn.ModuleList] = []
        upsamples: list[nn.Module] = []
        for level_idx, out_ch in enumerate(reversed_channels):
            # Skip connection is concatenated once, before the first block of
            # the level; later blocks at this level take out_ch -> out_ch.
            blocks = nn.ModuleList(
                [
                    TimestepResBlock3D(
                        cur_ch + out_ch if i == 0 else out_ch,
                        out_ch,
                        config.time_emb_dim,
                        groups=config.group_norm_groups,
                    )
                    for i in range(config.num_res_blocks_per_level)
                ]
            )
            up_blocks.append(blocks)
            cur_ch = out_ch
            is_last_level = level_idx == len(reversed_channels) - 1
            upsamples.append(Upsample3D(cur_ch) if not is_last_level else nn.Identity())
        self.up_blocks = nn.ModuleList(up_blocks)
        self.upsamples = nn.ModuleList(upsamples)

        self.norm_out = nn.GroupNorm(min(config.group_norm_groups, cur_ch), cur_ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv3d(cur_ch, config.latent_channels, kernel_size=3, padding=1)
        # Zero-init the final conv: at the start of training the model predicts
        # zero noise, which is the standard DDPM stabilization trick.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_t: noisy mask latent, (B, latent_channels, *spatial)
            t: timestep indices, (B,)
            condition: Uni-Encoder feature grid, (B, condition_channels, *spatial) — same spatial shape as z_t
        Returns:
            predicted noise, (B, latent_channels, *spatial)
        """
        assert z_t.shape[2:] == condition.shape[2:], (
            f"mask latent spatial shape {z_t.shape[2:]} != condition spatial shape {condition.shape[2:]}. "
            "The mask VAE's downsample factor and the Uni-Encoder's patch_shape must produce matching grids."
        )

        time_emb = self.time_embedding(t)
        cond_feat = self.cond_proj(condition)
        x = torch.cat([z_t, cond_feat], dim=1)
        x = self.conv_in(x)

        skips: list[torch.Tensor] = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                x = block(x, time_emb)
            skips.append(x)
            x = downsample(x)

        for block in self.bottleneck:
            x = block(x, time_emb)

        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, time_emb)
            x = upsample(x)

        x = self.act_out(self.norm_out(x))
        return self.conv_out(x)


class PixelUNet3D(nn.Module):
    """
    Pixel-space noise predictor for concat-conditioned diffusion.

    Input  = cat(volume_masked, noisy_mask)  — (B, image_ch+mask_ch, 96, 96, 96)
    Output = predicted noise for mask only   — (B, mask_ch, 96, 96, 96)

    No separate conditioning tensor — the image is baked into the input.
    Timestep conditioning via additive embedding inside every ResBlock.
    """

    def __init__(self, config: PixelDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        self.time_embedding = TimeEmbedding(config.time_emb_dim)

        channels = [config.base_channels * m for m in config.channel_multipliers]
        self.conv_in = nn.Conv3d(config.in_channels, channels[0], kernel_size=3, padding=1)

        # --- down path ---
        down_blocks: list[nn.ModuleList] = []
        downsamples: list[nn.Module] = []
        cur_ch = channels[0]
        for level_idx, out_ch in enumerate(channels):
            blocks = nn.ModuleList([
                TimestepResBlock3D(
                    cur_ch if i == 0 else out_ch, out_ch,
                    config.time_emb_dim, groups=config.group_norm_groups,
                )
                for i in range(config.num_res_blocks_per_level)
            ])
            down_blocks.append(blocks)
            cur_ch = out_ch
            is_last = level_idx == len(channels) - 1
            downsamples.append(Downsample3D(cur_ch) if not is_last else nn.Identity())
        self.down_blocks = nn.ModuleList(down_blocks)
        self.downsamples = nn.ModuleList(downsamples)

        # --- bottleneck ---
        self.bottleneck = nn.ModuleList([
            TimestepResBlock3D(cur_ch, cur_ch, config.time_emb_dim, groups=config.group_norm_groups)
            for _ in range(2)
        ])

        # --- up path ---
        reversed_channels = list(reversed(channels))
        up_blocks: list[nn.ModuleList] = []
        upsamples: list[nn.Module] = []
        for level_idx, out_ch in enumerate(reversed_channels):
            blocks = nn.ModuleList([
                TimestepResBlock3D(
                    cur_ch + out_ch if i == 0 else out_ch, out_ch,
                    config.time_emb_dim, groups=config.group_norm_groups,
                )
                for i in range(config.num_res_blocks_per_level)
            ])
            up_blocks.append(blocks)
            cur_ch = out_ch
            is_last = level_idx == len(reversed_channels) - 1
            upsamples.append(Upsample3D(cur_ch) if not is_last else nn.Identity())
        self.up_blocks = nn.ModuleList(up_blocks)
        self.upsamples = nn.ModuleList(upsamples)

        self.norm_out = nn.GroupNorm(min(config.group_norm_groups, cur_ch), cur_ch)
        self.act_out  = nn.SiLU()
        self.conv_out = nn.Conv3d(cur_ch, config.mask_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: cat(volume_masked, noisy_mask), (B, image_ch+mask_ch, D, H, W)
        t: timestep indices, (B,)
        Returns predicted noise, (B, mask_ch, D, H, W)
        """
        time_emb = self.time_embedding(t)
        x = self.conv_in(x)

        skips: list[torch.Tensor] = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                x = block(x, time_emb)
            skips.append(x)
            x = downsample(x)

        for block in self.bottleneck:
            x = block(x, time_emb)

        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            x = torch.cat([x, skips.pop()], dim=1)
            for block in blocks:
                x = block(x, time_emb)
            x = upsample(x)

        return self.conv_out(self.act_out(self.norm_out(x)))
