import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Standard transformer/DDPM sinusoidal embedding.

    Args:
        timesteps: (B,) integer or float timestep indices
        dim: embedding dimension (must be even)
    Returns:
        (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding -> 2-layer MLP, the standard DDPM time-conditioning path."""

    def __init__(self, time_emb_dim: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embed = sinusoidal_timestep_embedding(timesteps, self.time_emb_dim)
        return self.mlp(embed)
