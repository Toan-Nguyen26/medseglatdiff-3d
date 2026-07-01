"""
Stage-1 masked self-supervised pretraining wrapper around the Uni-Encoder.

Wraps the plain ViT (Uni-Encoder + LightDecoder) with:
  - a learnable "prior" volume, used to fill in modality/patch-masked regions
    of the input instead of zeros (so the reconstruction target stays
    meaningful even when an entire modality is dropped),
  - the actual modality+patch masking logic from mask_tools.py.

After this stage converges, only the inner `backbone`'s TokUniEncoder-equivalent
weights matter — see backbone.TokUniEncoder, which is what gets reused as the
diffusion conditioning encoder in Stage 3/4.
"""

from typing import Optional, Tuple, cast

import torch
from torch import nn

from models.uniencoder.backbone import ViT
from models.uniencoder.config import BackboneConfig
from models.uniencoder.mask_tools import apply_mask, tokenizer as patch_tokenizer
from models.uniencoder.scales import get_scale_config

Location3D = Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class UniEncoderPretrain(nn.Module):
    """Masked-pretraining wrapper: applies modality+patch masking, then runs ViT
    (Uni-Encoder + LightDecoder) to reconstruct the full, unmasked input volume."""

    def __init__(
        self,
        config: BackboneConfig,
        *,
        pre_train: bool = True,
        crop_size: int = 96,
        patch_size: int = 8,
        patch_mask_ratio: float = 0.75,
        original_shape: Tuple[int, int, int] = (176, 205, 154),
        num_mask_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = ViT(config)
        self.pre_train = pre_train
        self.num_modals = config.in_channels
        self.patch_size = patch_size
        self.mask_ratio = patch_mask_ratio
        self.num_mask_modalities = num_mask_modalities
        self.crop_size = crop_size

        # Learnable prior used to fill masked regions, instead of zeros — lets
        # the model reconstruct a plausible signal even where a whole modality
        # is missing, rather than trivially predicting "zero".
        self.learnable_prior = nn.Parameter(
            torch.randn((1, config.in_channels, *original_shape))
        )
        self.register_buffer(
            "dummy_input",
            patch_tokenizer(torch.ones((1, config.in_channels, crop_size, crop_size, crop_size)), patch_size=patch_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, location: Optional[Location3D] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert location is not None, "location (crop offsets into the original volume) is required."
        batch_size = x.shape[0]
        dummy_input = cast(torch.Tensor, self.dummy_input)

        mask = apply_mask(
            batch_size=batch_size,
            patch_size=self.patch_size,
            patch_mask_ratio=self.mask_ratio,
            raw_input=dummy_input,
            num_mask_modalities=self.num_mask_modalities,
            use_patch_mask=True,
            crop_size=self.crop_size,
        )
        (d0, _d1), (h0, _h1), (w0, _w1) = location
        depth, height, width = x.shape[2:]

        prior = self.learnable_prior.expand(batch_size, -1, -1, -1, -1)
        d_idx = d0[:, None] + torch.arange(depth, device=x.device, dtype=d0.dtype)[None, :]
        h_idx = h0[:, None] + torch.arange(height, device=x.device, dtype=h0.dtype)[None, :]
        w_idx = w0[:, None] + torch.arange(width, device=x.device, dtype=w0.dtype)[None, :]

        cropped_prior = prior.gather(
            2, d_idx[:, None, :, None, None].expand(-1, prior.shape[1], -1, prior.shape[3], prior.shape[4])
        )
        cropped_prior = cropped_prior.gather(
            3, h_idx[:, None, None, :, None].expand(-1, cropped_prior.shape[1], cropped_prior.shape[2], -1, cropped_prior.shape[4])
        )
        cropped_prior = cropped_prior.gather(
            4, w_idx[:, None, None, None, :].expand(-1, cropped_prior.shape[1], cropped_prior.shape[2], cropped_prior.shape[3], -1)
        )

        masked_x = x * mask + cropped_prior * (1 - mask)
        return self.backbone(masked_x), cropped_prior


def build_uniencoder_pretrain(
    scale: str = "Tiny",
    *,
    in_channels: int = 4,
    crop_size: int = 96,
    patch_size: int = 8,
    num_register_tokens: int = 4,
    patch_mask_ratio: float = 0.75,
    num_mask_modalities: int = 3,
    original_shape: Tuple[int, int, int] = (176, 205, 154),
) -> UniEncoderPretrain:
    """Factory for Stage-1 pretraining. Defaults to Tiny per docs/decisions.md
    (compute-constrained, targeting Kaggle free-tier feasibility)."""
    config = get_scale_config(
        scale,
        in_channels=in_channels,
        crop_shape=(crop_size, crop_size, crop_size),
        patch_shape=(patch_size, patch_size, patch_size),
        num_register_tokens=num_register_tokens,
    )
    return UniEncoderPretrain(
        config,
        crop_size=crop_size,
        patch_size=patch_size,
        patch_mask_ratio=patch_mask_ratio,
        original_shape=original_shape,
        num_mask_modalities=num_mask_modalities,
    )
