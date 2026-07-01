from dataclasses import dataclass
from typing import Tuple


@dataclass
class DiffusionConfig:
    """
    Config for the latent diffusion U-Net. Operates entirely on the mask
    latent grid — noise is added to/predicted from `latent_channels`-channel
    tensors, conditioned on the (never-noised) Uni-Encoder feature grid.

    latent_channels must match MaskVAEConfig.latent_channels.
    condition_channels must match the Uni-Encoder scale's embed_dim
    (e.g. 600 for Tiny, 384 for Nano — see models/uniencoder/scales.py).
    Both the mask latent and the Uni-Encoder grid must already share the same
    spatial shape (12^3 at the 96^3/patch-8 settings used elsewhere) — this
    module does no spatial resampling.
    """
    latent_channels: int = 8
    condition_channels: int = 600
    cond_proj_channels: int = 64

    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks_per_level: int = 2
    group_norm_groups: int = 8
    time_emb_dim: int = 256

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
