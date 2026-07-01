from dataclasses import dataclass
from typing import Tuple


@dataclass
class MaskVAEConfig:
    """
    3D VAE for the BraTS segmentation mask. Input/output is the mask itself
    (one-hot across class channels), not the MRI image.

    Downsampling = 2 ** len(channel_multipliers), which must match the
    Uni-Encoder's patch_shape downsampling (8x, i.e. 3 levels) so the mask
    latent and the Uni-Encoder's conditioning feature grid have the same
    spatial resolution for channel-wise concatenation in the diffusion U-Net.
    See docs/decisions.md.
    """
    num_classes: int = 4  # background + necrotic core + edema + enhancing tumor
    crop_shape: Tuple[int, int, int] = (96, 96, 96)
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks_per_level: int = 2
    latent_channels: int = 8
    group_norm_groups: int = 8

    @property
    def downsample_factor(self) -> int:
        return 2 ** len(self.channel_multipliers)

    @property
    def latent_spatial_shape(self) -> Tuple[int, int, int]:
        f = self.downsample_factor
        return tuple(s // f for s in self.crop_shape)  # type: ignore[return-value]
