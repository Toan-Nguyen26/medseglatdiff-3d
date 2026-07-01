"""
Scale presets for the Uni-Encoder, matching the variants in Hooorace-S/UniME.
Defaulting our own pipeline to Tiny/Nano for compute reasons (see docs/decisions.md).
"""

from typing import TypedDict, cast

from models.uniencoder.config import BackboneConfig


class ScaleSpec(TypedDict):
    embed_dim: int
    depth: int
    num_heads: int


_SCALE_SPECS: dict[str, ScaleSpec] = {
    "Original": {"embed_dim": 864, "depth": 16, "num_heads": 12},
    "Base": {"embed_dim": 864, "depth": 16, "num_heads": 12},
    "Small": {"embed_dim": 792, "depth": 12, "num_heads": 12},
    "Tiny": {"embed_dim": 600, "depth": 12, "num_heads": 10},
    "Nano": {"embed_dim": 384, "depth": 12, "num_heads": 8},
}


def get_scale_config(
    size: str,
    *,
    in_channels: int = 4,
    crop_shape: tuple[int, int, int] = (96, 96, 96),
    patch_shape: tuple[int, int, int] = (8, 8, 8),
    num_register_tokens: int = 4,
) -> BackboneConfig:
    key = size.title()
    if key not in _SCALE_SPECS:
        raise ValueError(f"Unknown Uni-Encoder scale '{size}'. Supported: {sorted(_SCALE_SPECS.keys())}.")
    spec = cast(ScaleSpec, _SCALE_SPECS[key])
    tokens_spatial_shape = tuple(c // p for c, p in zip(crop_shape, patch_shape))
    return BackboneConfig(
        in_channels=in_channels,
        out_channels=in_channels,
        crop_shape=crop_shape,
        patch_shape=patch_shape,
        tokens_spatial_shape=cast(tuple[int, int, int], tokens_spatial_shape),
        embed_dim=spec["embed_dim"],
        depth=spec["depth"],
        num_heads=spec["num_heads"],
        drop_path_rate=0.2,
        layer_scale_init_value=0.1,
        num_register_tokens=num_register_tokens,
        patch_drop_rate=0.0,
    )
