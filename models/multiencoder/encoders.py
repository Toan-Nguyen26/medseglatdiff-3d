"""
MONAI-based 3D image encoder and VAE for diffusion conditioning.

Two classes are provided:

  ImageEncoder  — deterministic AutoEncoder. encode() gives a fixed conditioning
                  feature grid. Fast, no sampling overhead.

  ImageVAE      — variational version. encode() returns (mu, logvar); the
                  reparameterised z ~ N(mu, exp(logvar)) is used for conditioning.
                  At inference, z = mu (deterministic, same as ImageEncoder).
                  During training the KL term forces sigma to reflect how much
                  information was actually available, so missing modalities produce
                  larger sigma — this is the uncertainty signal we propagate to the
                  diffusion U-Net when sampling N plausible masks.

Both compress (B, 4, D, H, W) → (B, embed_dim, D/8, H/8, W/8).
At the default 96^3 crop and embed_dim=256 the bottleneck is (B, 256, 12, 12, 12).

Reconstruction loss is computed ONLY on present (non-zero) modality channels so
the encoder is never penalised for not knowing about modalities it never saw.
"""

import torch
import torch.nn.functional as F
from torch import nn

from monai.networks.nets import AutoEncoder


# ---------------------------------------------------------------------------
# Deterministic encoder (original)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    SPATIAL_DOWNSAMPLE_FACTOR: int = 8

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        num_res_units: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._ae = AutoEncoder(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=(32, 64, 128, embed_dim),
            strides=(2, 2, 2, 1),
            num_res_units=num_res_units,
            norm="INSTANCE",
            act="RELU",
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, D, H, W) → (B, embed_dim, D/8, H/8, W/8)"""
        return self._ae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, embed_dim, D/8, H/8, W/8) → (B, 4, D, H, W)"""
        return self._ae.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def output_spatial_shape(self, crop_size: int) -> tuple[int, int, int]:
        s = crop_size // self.SPATIAL_DOWNSAMPLE_FACTOR
        return (s, s, s)


# ---------------------------------------------------------------------------
# Variational encoder (new)
# ---------------------------------------------------------------------------

class ImageVAE(nn.Module):
    """
    Variational image encoder for diffusion conditioning with uncertainty.

    Architecture:
        Shared backbone (MONAI AutoEncoder encoder path)
            → mu_head (1×1×1 conv)   → mu     (B, embed_dim, 12, 12, 12)
            → logvar_head (1×1×1 conv)→ logvar (B, embed_dim, 12, 12, 12)
        z = mu + eps * exp(0.5 * logvar)   [training]
        z = mu                              [inference]
        Decoder (MONAI AutoEncoder decoder path)
            → recon (B, 4, 96, 96, 96)

    Uncertainty signal:
        With all 4 modalities: backbone has rich input → small logvar
        With 1 modality only:  backbone has sparse input → large logvar
        N samples of z ~ N(mu, exp(logvar)) give N different conditioning
        signals → N different segmentation masks → uncertainty estimation.
    """

    SPATIAL_DOWNSAMPLE_FACTOR: int = 8

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        num_res_units: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        # bottleneck dim = last channel; all-but-last stages use stride 2,
        # last stage uses stride 1 (channel projection only, no spatial reduction)
        self.embed_dim = channels[-1]
        strides = (2,) * (len(channels) - 1) + (1,)

        self._backbone = AutoEncoder(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="INSTANCE",
            act="RELU",
        )

        # VAE heads on top of the bottleneck
        self.mu_head     = nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.logvar_head = nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (B, 4, D, H, W) → (mu, logvar), each (B, embed_dim, D/8, H/8, W/8)
        """
        h = self._backbone.encode(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample z ~ N(mu, exp(logvar)) during training.
        Returns mu directly at inference (no stochasticity needed for forward pass).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, embed_dim, D/8, H/8, W/8) → (B, 4, D, H, W)"""
        return self._backbone.decode(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (recon, mu, logvar).
        Use mu (or a sample from reparameterize) as the diffusion conditioning signal.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def output_spatial_shape(self, crop_size: int) -> tuple[int, int, int]:
        s = crop_size // self.SPATIAL_DOWNSAMPLE_FACTOR
        return (s, s, s)

    def sample_conditioning(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw n_samples conditioning vectors from z ~ N(mu, exp(logvar)).
        Used at inference to generate N plausible segmentation masks.

        Returns: (n_samples, B, embed_dim, D/8, H/8, W/8)
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            samples = [mu + torch.randn_like(std) * std for _ in range(n_samples)]
        return torch.stack(samples)


# ---------------------------------------------------------------------------
# Mask VAE
# ---------------------------------------------------------------------------

class MaskVAE(nn.Module):
    """
    Variational autoencoder for binary segmentation masks (WT / TC / ET).

    Compresses (B, 3, D, H, W) binary masks into a small latent space
    (B, latent_channels, D/8, H/8, W/8).  KL regularisation forces
    z ~ N(0, 1) — perfectly matched to DDPM Gaussian noise, so no logit
    transform is needed.

    At training time the diffusion U-Net denoises z.
    At inference the decoder converts z_0 back to full-resolution masks.

    Decoder outputs raw logits — call .sigmoid() for probabilities.
    """

    SPATIAL_DOWNSAMPLE_FACTOR: int = 8

    def __init__(
        self,
        num_classes: int = 3,
        latent_channels: int = 4,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        num_res_units: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes     = num_classes
        self.latent_channels = latent_channels
        self.channels        = channels
        embed_dim = channels[-1]
        strides   = (2,) * (len(channels) - 1) + (1,)

        self._backbone = AutoEncoder(
            spatial_dims=3,
            in_channels=num_classes,
            out_channels=num_classes,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="INSTANCE",
            act="RELU",
        )

        # project backbone bottleneck → latent_channels (and back)
        self.mu_head     = nn.Conv3d(embed_dim, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv3d(embed_dim, latent_channels, kernel_size=1)
        self.pre_decode  = nn.Conv3d(latent_channels, embed_dim, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, D, H, W) → (mu, logvar), each (B, latent_channels, D/8, H/8, W/8)"""
        h = self._backbone.encode(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_channels, D/8, H/8, W/8) → (B, 3, D, H, W) logits"""
        return self._backbone.decode(self.pre_decode(z))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon_logits, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def output_spatial_shape(self, crop_size: int) -> tuple[int, int, int]:
        s = crop_size // self.SPATIAL_DOWNSAMPLE_FACTOR
        return (s, s, s)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(
    recon: torch.Tensor,
    volume_masked: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss computed ONLY on present (non-zero) modality channels.
    Missing channels (all-zero in volume_masked) contribute zero loss —
    the encoder is never penalised for not knowing what it never saw.

    Args:
        recon:         (B, 4, D, H, W) — decoder output
        volume_masked: (B, 4, D, H, W) — input with absent channels zeroed
    """
    # present[b, c] = 1 if channel c has any signal in sample b
    present = (volume_masked.abs().sum(dim=(2, 3, 4), keepdim=True) > 0).float()
    return F.mse_loss(recon * present, volume_masked * present)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: KL( N(mu, exp(logvar)) || N(0, 1) ).
    Closed-form solution: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))

    This term forces the encoder to produce meaningful sigma:
      - large sigma when little information is available (missing modalities)
      - small sigma when all modalities are present and the representation is confident
    """
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()


def vae_loss(
    recon: torch.Tensor,
    volume_masked: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined VAE loss = reconstruction_loss + beta * kl_loss.

    beta controls how strongly the KL term regularises the latent space.
    Small beta (1e-4 default) lets the model focus on reconstruction first;
    increase toward 1e-3 or 1e-2 if sigma collapses to near-zero.

    Returns (total, recon_loss, kl) for separate logging.
    """
    r_loss = reconstruction_loss(recon, volume_masked)
    k_loss = kl_loss(mu, logvar)
    return r_loss + beta * k_loss, r_loss, k_loss


def mask_vae_recon_loss(
    recon_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dice + BCE for binary mask reconstruction.

    Works for both mask formats:
      - 3-channel [WT, TC, ET]  (region_based)
      - 4-channel [BG, NCR, ED, ET]  (subregion_based)

    pos_weight: (num_channels,) tensor to upweight rare classes in BCE.
    For subregion mode, pass e.g. tensor([0.1, 10.0, 3.0, 5.0]) to heavily
    upweight NCR (channel 1) relative to background (channel 0).
    """
    if pos_weight is not None:
        # broadcast pos_weight over (B, C, D, H, W)
        pw = pos_weight.to(recon_logits.device)
        pw = pw.view(1, -1, *([1] * (recon_logits.ndim - 2)))
        bce = F.binary_cross_entropy_with_logits(recon_logits, gt_masks, pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(recon_logits, gt_masks)

    probs = recon_logits.sigmoid()
    eps   = 1e-5
    dims  = list(range(2, recon_logits.ndim))
    tp    = (probs * gt_masks).sum(dim=dims)
    fp    = (probs * (1 - gt_masks)).sum(dim=dims)
    fn    = ((1 - probs) * gt_masks).sum(dim=dims)
    dice  = (1 - (2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()
    return bce + dice


def mask_vae_loss(
    recon_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined MaskVAE loss = mask_vae_recon_loss + beta * kl_loss.
    Returns (total, recon_loss, kl).
    """
    r_loss = mask_vae_recon_loss(recon_logits, gt_masks, pos_weight=pos_weight)
    k_loss = kl_loss(mu, logvar)
    return r_loss + beta * k_loss, r_loss, k_loss
