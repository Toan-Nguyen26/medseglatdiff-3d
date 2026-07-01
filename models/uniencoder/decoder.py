import math

import torch
from torch import nn

from models.uniencoder.utils import LayerNormNd


class LightDecoder(nn.Module):
    """
    Lightweight auxiliary decoder used only during Stage-1 masked self-supervised
    pretraining, to reconstruct the full input volume from the Uni-Encoder's token
    grid. Discarded after pretraining — never used downstream as the diffusion
    conditioning path.
    """

    def __init__(
        self, in_channels: int, out_channels: int, patch_shape: tuple[int, int, int] = (8, 8, 8)
    ) -> None:
        super().__init__()
        num_stages = self._compute_num_stages(patch_shape)
        strides = self._compute_strides(patch_shape, num_stages)
        channels = self._compute_channel_schedule(in_channels, out_channels, num_stages)
        self.decoder = self._build_decoder_layers(channels, strides)

    @staticmethod
    def _compute_num_stages(patch: tuple[int, int, int]) -> int:
        return int(round(math.log2(max(patch))))

    @staticmethod
    def _compute_strides(patch_shape: tuple[int, int, int], num_stages: int) -> list[tuple[int, int, int]]:
        return [
            (
                2 if (patch_shape[0] // (2**stage)) % 2 == 0 else 1,
                2 if (patch_shape[1] // (2**stage)) % 2 == 0 else 1,
                2 if (patch_shape[2] // (2**stage)) % 2 == 0 else 1,
            )
            for stage in reversed(range(num_stages))
        ]

    @staticmethod
    def _round_to_multiple_of_8(value: float) -> int:
        return max(8, round(value / 8) * 8)

    def _compute_channel_schedule(self, in_channels: int, out_channels: int, num_stages: int) -> list[int]:
        dim_reduction_factor = (in_channels / (2 * out_channels)) ** (1 / num_stages)
        channels = [
            self._round_to_multiple_of_8(in_channels / (dim_reduction_factor ** (i + 1)))
            for i in range(num_stages)
        ]
        channels = [in_channels] + channels
        channels[-1] = out_channels
        return channels

    @staticmethod
    def _build_decoder_layers(channels: list[int], strides: list[tuple[int, int, int]]) -> nn.Sequential:
        layers: list[nn.Module] = []
        num_layers = len(channels) - 1

        for idx in range(num_layers - 1):
            layers.extend(
                [
                    nn.ConvTranspose3d(
                        in_channels=channels[idx],
                        out_channels=channels[idx + 1],
                        kernel_size=strides[idx],
                        stride=strides[idx],
                        bias=True,
                    ),
                    LayerNormNd(channels[idx + 1]),
                    nn.GELU(),
                ]
            )

        layers.append(
            nn.ConvTranspose3d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=strides[-1],
                stride=strides[-1],
                bias=True,
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
