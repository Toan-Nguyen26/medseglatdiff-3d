from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.mask_vae.config import MaskVAEConfig
from models.mask_vae.decoder import MaskDecoder3D
from models.mask_vae.encoder import MaskEncoder3D


@dataclass
class MaskVAEOutput:
    logits: torch.Tensor  # (B, num_classes, D, H, W), reconstructed mask logits
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor  # sampled latent actually fed to the decoder


class MaskVAE3D(nn.Module):
    """
    Continuous 3D VAE for BraTS segmentation masks. Encodes a one-hot mask
    volume to a latent grid (matching the Uni-Encoder's spatial resolution),
    and decodes back to class logits.

    This is the *mask* side of the pipeline — kept and trained independently
    of the Uni-Encoder (see docs/decisions.md, "sequential training" decision).
    """

    def __init__(self, config: MaskVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MaskEncoder3D(config)
        self.decoder = MaskDecoder3D(config)

    def encode(self, mask_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(mask_onehot)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, mask_onehot: torch.Tensor, *, sample: bool = True) -> MaskVAEOutput:
        mu, logvar = self.encode(mask_onehot)
        z = self.reparameterize(mu, logvar) if sample else mu
        logits = self.decode(z)
        return MaskVAEOutput(logits=logits, mu=mu, logvar=logvar, z=z)


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    target_onehot: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    WCE reconstruction loss (replaces MSE), per MedSegLatDiff's finding that
    this better preserves small/sparse structures (e.g. enhancing tumor core)
    instead of treating them as noise.

    Args:
        logits: (B, num_classes, D, H, W) raw decoder output
        target_onehot: (B, num_classes, D, H, W) ground-truth one-hot mask
        class_weights: (num_classes,) — upweight small/rare classes
    """
    target_indices = target_onehot.argmax(dim=1)  # (B, D, H, W)
    return F.cross_entropy(logits, target_indices, weight=class_weights)


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL term against a unit Gaussian prior, averaged over batch."""
    kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).flatten(1).sum(dim=1)
    return kl_per_sample.mean()


def mask_vae_loss(
    output: MaskVAEOutput,
    target_onehot: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    kl_weight: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Total loss = WCE reconstruction + kl_weight * KL.
    kl_weight is intentionally tiny by default — reconstruction fidelity matters
    far more than latent regularity for this use case (the diffusion model
    learns the latent distribution directly in Stage 3/4, it doesn't sample
    z ~ N(0, I) from this VAE's prior the way a generative VAE would).
    """
    recon_loss = weighted_cross_entropy_loss(output.logits, target_onehot, class_weights=class_weights)
    kl_loss = kl_divergence_loss(output.mu, output.logvar)
    total = recon_loss + kl_weight * kl_loss
    return total, {"recon_loss": recon_loss.detach(), "kl_loss": kl_loss.detach(), "total_loss": total.detach()}
