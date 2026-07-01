from dataclasses import dataclass
from typing import Tuple


@dataclass
class BackboneConfig:
    """
    Configuration for the Uni-Encoder backbone.

    in_channels: number of input modality channels (4 for BraTS: FLAIR/T1ce/T1/T2)
    crop_shape: shape of the input crop, (D, H, W)
    patch_shape: shape of each tokenizer patch, (D, H, W)
    tokens_spatial_shape: crop_shape // patch_shape per axis
    embed_dim / depth / num_heads: transformer sizing
    num_register_tokens: extra non-spatial tokens (improves stability under masked pretraining)
    patch_drop_rate: fraction of tokens dropped at the sequence level (separate from the
        modality/patch masking used during Stage-1 pretraining itself)
    """
    in_channels: int = 4
    out_channels: int = 4
    crop_shape: Tuple[int, int, int] = (96, 96, 96)
    patch_shape: Tuple[int, int, int] = (8, 8, 8)
    embed_dim: int = 600
    depth: int = 12
    num_heads: int = 10
    drop_path_rate: float = 0.2
    patch_drop_rate: float = 0.0
    return_mask: bool = False
    mlp_dropout: float = 0.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    layer_scale_init_value: float = 0.1
    rope_base: float = 10000.0
    use_learned_pos_embed: bool = True
    tokens_spatial_shape: Tuple[int, int, int] = (12, 12, 12)
    use_rotary_pos_embed: bool = True
    num_register_tokens: int = 4
