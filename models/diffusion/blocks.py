import torch
from torch import nn


class TimestepResBlock3D(nn.Module):
    """
    GroupNorm-SiLU-Conv x2 residual block with a timestep embedding injected
    additively after the first conv — the standard DDPM U-Net building block.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, *, groups: int = 8) -> None:
        super().__init__()
        groups_in = min(groups, in_channels)
        groups_out = min(groups, out_channels)

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(time_emb))[:, :, None, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)
