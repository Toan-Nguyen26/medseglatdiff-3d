"""
Standard DDPM Gaussian diffusion schedule (Ho et al., 2020), which is what
MedSegLatDiff's Eqs. 2-5 describe in prose (linear beta schedule, simplified
MSE training objective, fixed-variance reverse sampling). Implemented against
the well-established canonical formulas rather than transcribed literally
from the paper's OCR-extracted equations, since PDF text extraction garbled
several sub/superscripts in that section.
"""

from dataclasses import dataclass

import torch
from torch import nn

from models.diffusion.config import DiffusionConfig
from models.diffusion.unet3d import UNet3D


def _extract(coeffs: torch.Tensor, t: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    """Gather per-sample coefficients at timestep t, reshaped to broadcast over (B, C, *spatial)."""
    out = coeffs.gather(0, t).float()
    return out.view(t.shape[0], *([1] * (len(broadcast_shape) - 1)))


@dataclass
class DiffusionStepOutput:
    x_prev: torch.Tensor  # denoised sample at t-1


class GaussianDiffusionSchedule(nn.Module):
    """
    Holds the noise schedule and implements:
      - q_sample: forward process, clean latent -> noisy latent at timestep t
      - p_sample: one reverse (denoising) step
      - sample_loop: full reverse process, pure noise -> clean latent
    Operates purely on tensors — has no opinion about what the latent
    represents (mask latent here, but it's generic).
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.num_timesteps = num_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            x0: clean latent, (B, C, *spatial)
            t: timestep indices, (B,)
            noise: optional pre-sampled noise (same shape as x0); sampled fresh if None
        Returns:
            (x_t, noise) — noise is returned since it's the training target.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_omab = _extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_omab * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, predicted_noise: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        One reverse step: x_{t-1} from x_t and the model's predicted noise,
        using fixed variance = beta_t (the "simple" DDPM sampling variant).
        """
        beta_t = _extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_omab_t = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_omab_t) * predicted_noise)

        noise = torch.randn_like(x_t)
        # no noise added at the final step (t == 0)
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample_loop(
        self,
        unet: UNet3D,
        condition: torch.Tensor,
        latent_shape: torch.Size,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Full reverse process: pure Gaussian noise -> denoised latent, conditioned
        on `condition` (the Uni-Encoder feature grid) at every step.

        Args:
            unet: trained noise predictor
            condition: (B, condition_channels, *spatial) — kept fixed across all steps
            latent_shape: (B, latent_channels, *spatial) — shape of the mask latent to generate
        Returns:
            z_0: denoised mask latent, ready for MaskVAE3D.decode(...)
        """
        unet.eval()
        batch_size = latent_shape[0]
        x_t = torch.randn(latent_shape, device=device, generator=generator)

        for step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            predicted_noise = unet(x_t, t, condition)
            x_t = self.p_sample(predicted_noise, x_t, t)

        return x_t


def diffusion_loss(
    unet: UNet3D,
    schedule: GaussianDiffusionSchedule,
    z0: torch.Tensor,
    condition: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L_simple: MSE between true and predicted noise (Eq. 4). Samples a random
    timestep per batch element if `t` isn't given.
    """
    batch_size = z0.shape[0]
    if t is None:
        t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=z0.device, dtype=torch.long)

    x_t, noise = schedule.q_sample(z0, t)
    predicted_noise = unet(x_t, t, condition)
    return torch.nn.functional.mse_loss(predicted_noise, noise)
