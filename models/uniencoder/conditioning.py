"""
Factory + checkpoint loading for the Uni-Encoder used as the diffusion model's
image conditioning backbone (Stage 3/4) — i.e. TokUniEncoder, no decoder.

Output of `UniEncoderConditioning.forward` is a continuous feature grid
(B, embed_dim, *tokens_spatial_shape), which the diffusion U-Net's conditioning
path should project (1x1x1 conv) down to its own channel width before
concatenating with the noisy mask latent. See docs/decisions.md for why the
mask VAE's downsampling factor must match patch_shape so the two grids align
spatially without resampling.
"""

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from models.uniencoder.backbone import TokUniEncoder
from models.uniencoder.scales import get_scale_config


class UniEncoderConditioning(nn.Module):
    def __init__(
        self,
        scale: str = "Tiny",
        *,
        in_channels: int = 4,
        crop_size: int = 96,
        patch_size: int = 8,
        num_register_tokens: int = 4,
    ) -> None:
        super().__init__()
        config = get_scale_config(
            scale,
            in_channels=in_channels,
            crop_shape=(crop_size, crop_size, crop_size),
            patch_shape=(patch_size, patch_size, patch_size),
            num_register_tokens=num_register_tokens,
        )
        self.embed_dim = config.embed_dim
        self.tokens_spatial_shape = config.tokens_spatial_shape
        self.encoder = TokUniEncoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, D, H, W) -> (B, embed_dim, *tokens_spatial_shape)"""
        return self.encoder(x)

    def load_stage1_checkpoint(self, checkpoint_path: str | Path, *, strict: bool = False) -> tuple[list[str], list[str]]:
        """
        Load weights from a Stage-1 `UniEncoderPretrain` checkpoint. Only keys
        belonging to the inner backbone (Uni-Encoder proper, not the
        LightDecoder discarded after pretraining) are kept.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict: Mapping[str, torch.Tensor] = checkpoint.get("model_state_dict", checkpoint)

        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            # Stage-1 wrapper nests the ViT as `backbone.*`; ViT itself nests
            # the decoder as `decoder.*`, which we deliberately drop here.
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone.") :]
            if clean_key.startswith("decoder."):
                continue
            if clean_key in self.encoder.state_dict():
                filtered[clean_key] = value

        result = self.encoder.load_state_dict(filtered, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
