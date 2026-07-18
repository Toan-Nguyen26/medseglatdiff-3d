"""
Joint training: MONAI image encoder + diffusion U-Net, mask VAE frozen.

Pipeline (two stages total):

    Stage 1: train_mask_vae.py      (mask VAE, independent)
    Stage 2: train_joint.py  <--    (image encoder + U-Net jointly, mask VAE frozen)

The MONAI AutoEncoder encoder and the diffusion U-Net train together from
scratch via a combined loss:

    total = diffusion_loss + recon_weight * reconstruction_loss

diffusion_loss: standard DDPM MSE on the mask latent.
reconstruction_loss: L1 between decoded image and masked input, giving the
    encoder a direct image-level supervision signal without a separate
    pretraining stage. The decoder is discarded at inference.

Missing-modality robustness: every training step randomly draws one of the 15
non-empty modality combinations via sample_modality_mask(), so the encoder
learns to produce useful conditioning from any available modality subset.

Run from repo root:
    python3 -m training.train_joint \\
        --data_root /path/to/brats_processed \\
        --mask_vae_checkpoint checkpoints/mask_vae/final.pth
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask
from models.diffusion.config import DiffusionConfig
from models.diffusion.schedule import GaussianDiffusionSchedule, diffusion_loss
from models.diffusion.unet3d import UNet3D
from models.mask_vae.config import MaskVAEConfig
from models.mask_vae.vae import MaskVAE3D
from models.multiencoder.encoders import ImageEncoder, reconstruction_loss
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "joint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="train.txt")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=4)

    # Mask VAE (frozen — must match Stage 1 training config)
    parser.add_argument("--mask_vae_checkpoint", type=str, required=True)
    parser.add_argument("--mask_vae_base_channels", type=int, default=64,
                        help="Must match Stage 1 mask VAE training value")
    parser.add_argument("--latent_channels", type=int, default=8)

    # Image encoder
    parser.add_argument("--encoder_embed_dim", type=int, default=256,
                        help="Bottleneck channels; also the diffusion condition_channels")
    parser.add_argument("--encoder_num_res_units", type=int, default=2)

    # Diffusion U-Net
    parser.add_argument("--cond_proj_channels", type=int, default=64)
    parser.add_argument("--unet_base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=1000)

    # Loss
    parser.add_argument("--recon_weight", type=float, default=0.05,
                        help="Weight for the auxiliary image reconstruction loss. "
                             "Diffusion loss dominates; this just gives the encoder "
                             "a direct image-level signal.")

    # Optimiser
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)

    # Infra
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint saved by this script to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[MaskVAE3D, ImageEncoder, UNet3D, GaussianDiffusionSchedule]:
    # --- Mask VAE (frozen) ---
    mask_vae_config = MaskVAEConfig(
        num_classes=args.num_classes,
        crop_shape=(args.crop_size, args.crop_size, args.crop_size),
        base_channels=args.mask_vae_base_channels,
        latent_channels=args.latent_channels,
    )
    mask_vae = MaskVAE3D(mask_vae_config).to(device)
    vae_ckpt = torch.load(args.mask_vae_checkpoint, map_location=device, weights_only=True)
    mask_vae.load_state_dict(vae_ckpt.get("model_state_dict", vae_ckpt))
    mask_vae.eval()
    for p in mask_vae.parameters():
        p.requires_grad = False

    # --- MONAI image encoder (trained jointly) ---
    image_encoder = ImageEncoder(
        in_channels=4,
        embed_dim=args.encoder_embed_dim,
        num_res_units=args.encoder_num_res_units,
    ).to(device)

    # Spatial alignment: encoder bottleneck must match mask VAE latent grid
    encoder_spatial = image_encoder.output_spatial_shape(args.crop_size)
    vae_spatial = mask_vae_config.latent_spatial_shape
    if encoder_spatial != vae_spatial:
        raise ValueError(
            f"Spatial mismatch: encoder output {encoder_spatial} != "
            f"mask VAE latent {vae_spatial}. "
            f"encoder downsample={image_encoder.SPATIAL_DOWNSAMPLE_FACTOR}x, "
            f"VAE downsample={mask_vae_config.downsample_factor}x — they must match."
        )

    # --- Diffusion U-Net (trained jointly) ---
    diffusion_config = DiffusionConfig(
        latent_channels=args.latent_channels,
        condition_channels=args.encoder_embed_dim,
        cond_proj_channels=args.cond_proj_channels,
        base_channels=args.unet_base_channels,
        num_timesteps=args.num_timesteps,
    )
    unet = UNet3D(diffusion_config).to(device)
    schedule = GaussianDiffusionSchedule(
        diffusion_config.num_timesteps,
        diffusion_config.beta_start,
        diffusion_config.beta_end,
    ).to(device)

    return mask_vae, image_encoder, unet, schedule


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    mask_vae, image_encoder, unet, schedule = build_models(args, device)

    # Encoder (both encode and decode paths) + U-Net are trainable
    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters()) + list(unet.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        image_encoder.load_state_dict(ckpt["encoder_state_dict"])
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(args.data_root, args.split_file),
        crop_size=args.crop_size,
        num_classes=args.num_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_id = new_run_id(STAGE)
    logger = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    if args.resume is not None:
        logger.note(f"Resumed from {args.resume} at step {start_step}.")

    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    image_encoder.train()
    unet.train()
    step = start_step
    data_iter = iter(loader)

    while step < args.num_steps:
        try:
            volume, mask_onehot = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            volume, mask_onehot = next(data_iter)

        volume = volume.to(device)           # (B, 4, D, H, W)
        mask_onehot = mask_onehot.to(device) # (B, 4, D, H, W)

        # Random missing-modality mask — core of missing-modality robustness
        modality_mask = sample_modality_mask(volume.shape[0]).to(device)
        volume_masked = apply_modality_mask(volume, modality_mask)

        # Encode image → conditioning feature + optionally decode for recon loss
        z_img = image_encoder.encode(volume_masked)   # (B, embed_dim, 12, 12, 12)

        # Auxiliary reconstruction loss — gives encoder direct image supervision
        recon = image_encoder.decode(z_img)           # (B, 4, D, H, W)
        recon_loss = reconstruction_loss(recon, volume_masked)

        # Encode mask → clean latent target (VAE frozen, use mean only)
        with torch.no_grad():
            mu, _logvar = mask_vae.encode(mask_onehot)
        z0 = mu                                        # (B, latent_channels, 12, 12, 12)

        diff_loss = diffusion_loss(unet, schedule, z0, z_img)

        loss = diff_loss + args.recon_weight * recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            logger.log_metrics(
                step=step,
                train_loss=loss.item(),
                diff_loss=diff_loss.item(),
                recon_loss=recon_loss.item(),
            )
            print(
                f"[{STAGE}] step={step}  "
                f"loss={loss.item():.4f}  "
                f"diff={diff_loss.item():.4f}  "
                f"recon={recon_loss.item():.4f}"
            )

        if step % args.ckpt_every == 0 and step > start_step:
            ckpt_path = checkpoint_dir / f"step_{step}.pth"
            torch.save(
                {
                    "encoder_state_dict": image_encoder.state_dict(),
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            logger.save_checkpoint_meta(
                ckpt_path,
                step=step,
                epoch=step // max(1, len(loader)),
                parent_checkpoint=args.resume,
                seed=args.seed,
                extra={"mask_vae_checkpoint": args.mask_vae_checkpoint},
            )

        step += 1

    final_ckpt_path = checkpoint_dir / "final.pth"
    torch.save(
        {
            "encoder_state_dict": image_encoder.state_dict(),
            "unet_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        final_ckpt_path,
    )
    logger.save_checkpoint_meta(
        final_ckpt_path,
        step=step,
        epoch=step // max(1, len(loader)),
        parent_checkpoint=args.resume,
        seed=args.seed,
    )
    logger.append_to_experiments_index(
        f"MONAI encoder joint, {step} steps, embed_dim={args.encoder_embed_dim}, "
        f"recon_weight={args.recon_weight}, final_loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
