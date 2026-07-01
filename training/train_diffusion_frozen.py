"""
Stage 3: train the diffusion U-Net with the Uni-Encoder AND mask VAE both
frozen. This is the third of four sequential training stages (see
docs/decisions.md, "Sequential training, not joint-from-scratch"):

    1. pretrain_uniencoder.py  (independent)
    2. train_mask_vae.py       (independent)
    3. train_diffusion_frozen.py   <- this script
    4. finetune_joint_llrd.py  (unfreezes the Uni-Encoder with layer-wise LR decay)

Both upstream checkpoints (Uni-Encoder Stage-1, mask VAE) are required inputs
here, not trained from scratch alongside the U-Net — joint training from
random init would let diffusion-loss gradients corrupt their pretrained
weights before the U-Net has learned anything useful to condition on.

Mask latent target: we encode the mask with the VAE and use the encoder MEAN
(mu), not a reparameterized sample, as the clean diffusion target z0. The
paper's own VQ-VAE produces a deterministic quantized latent (no sampling
noise at this stage at all); using mu is the closest continuous-VAE analogue
to that, and avoids stacking VAE-sampling noise on top of diffusion noise
during training.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask
from models.diffusion.config import DiffusionConfig
from models.diffusion.schedule import GaussianDiffusionSchedule, diffusion_loss
from models.diffusion.unet3d import UNet3D
from models.mask_vae.config import MaskVAEConfig
from models.mask_vae.vae import MaskVAE3D
from models.uniencoder.conditioning import UniEncoderConditioning
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "stage3_diffusion_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing vol/, seg/, *.txt splits")
    parser.add_argument("--split_file", type=str, default="train.txt")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--scale", type=str, default="Tiny", choices=["Original", "Base", "Small", "Tiny", "Nano"])
    parser.add_argument("--uniencoder_checkpoint", type=str, required=True, help="Stage-1 pretrained checkpoint")
    parser.add_argument("--mask_vae_checkpoint", type=str, required=True, help="Stage-2 trained mask VAE checkpoint")

    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--mask_vae_base_channels", type=int, default=64, help="Must match the value used in Stage 2 mask VAE training")
    parser.add_argument("--cond_proj_channels", type=int, default=64)
    parser.add_argument("--unet_base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[UniEncoderConditioning, MaskVAE3D, UNet3D, GaussianDiffusionSchedule]:
    uniencoder = UniEncoderConditioning(
        args.scale, in_channels=4, crop_size=args.crop_size, patch_size=args.patch_size,
    ).to(device)
    missing, unexpected = uniencoder.load_stage1_checkpoint(args.uniencoder_checkpoint)
    if missing or unexpected:
        raise RuntimeError(
            f"Uni-Encoder checkpoint did not load cleanly. missing={missing} unexpected={unexpected}. "
            "Refusing to silently train against a partially-loaded encoder."
        )
    uniencoder.eval()
    for p in uniencoder.parameters():
        p.requires_grad = False

    mask_vae_config = MaskVAEConfig(
        num_classes=args.num_classes,
        crop_shape=(args.crop_size, args.crop_size, args.crop_size),
        base_channels=args.mask_vae_base_channels,
        latent_channels=args.latent_channels,
    )
    mask_vae = MaskVAE3D(mask_vae_config).to(device)
    vae_checkpoint = torch.load(args.mask_vae_checkpoint, map_location=device, weights_only=True)
    mask_vae.load_state_dict(vae_checkpoint.get("model_state_dict", vae_checkpoint))
    mask_vae.eval()
    for p in mask_vae.parameters():
        p.requires_grad = False

    if uniencoder.tokens_spatial_shape != mask_vae_config.latent_spatial_shape:
        raise ValueError(
            f"Spatial mismatch: Uni-Encoder grid {uniencoder.tokens_spatial_shape} != "
            f"mask VAE latent grid {mask_vae_config.latent_spatial_shape}. "
            "Downsample factors must match (see docs/decisions.md)."
        )

    diffusion_config = DiffusionConfig(
        latent_channels=args.latent_channels,
        condition_channels=uniencoder.embed_dim,
        cond_proj_channels=args.cond_proj_channels,
        base_channels=args.unet_base_channels,
        num_timesteps=args.num_timesteps,
    )
    unet = UNet3D(diffusion_config).to(device)
    schedule = GaussianDiffusionSchedule(
        diffusion_config.num_timesteps, diffusion_config.beta_start, diffusion_config.beta_end
    ).to(device)

    return uniencoder, mask_vae, unet, schedule


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    uniencoder, mask_vae, unet, schedule = build_models(args, device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        unet.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(args.data_root, args.split_file),
        crop_size=args.crop_size,
        num_classes=args.num_classes,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    run_id = new_run_id(STAGE)
    logger = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    if args.resume is not None:
        logger.note(f"Resumed from {args.resume} at step {start_step}.")

    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unet.train()
    step = start_step
    data_iter = iter(loader)
    while step < args.num_steps:
        try:
            volume, mask_onehot = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            volume, mask_onehot = next(data_iter)

        volume = volume.to(device)
        mask_onehot = mask_onehot.to(device)

        modality_mask = sample_modality_mask(volume.shape[0]).to(device)
        volume = apply_modality_mask(volume, modality_mask)

        with torch.no_grad():
            condition = uniencoder(volume)
            mu, _logvar = mask_vae.encode(mask_onehot)
            z0 = mu

        loss = diffusion_loss(unet, schedule, z0, condition)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            logger.log_metrics(step=step, train_loss=loss.item())
            print(f"[{STAGE}] step={step} loss={loss.item():.4f}")

        if step % args.ckpt_every == 0 and step > start_step:
            ckpt_path = checkpoint_dir / f"step_{step}.pth"
            torch.save(
                {
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            logger.save_checkpoint_meta(
                ckpt_path, step=step, epoch=step // max(1, len(loader)),
                parent_checkpoint=args.resume, seed=args.seed,
                extra={"uniencoder_checkpoint": args.uniencoder_checkpoint, "mask_vae_checkpoint": args.mask_vae_checkpoint},
            )

        step += 1

    final_ckpt_path = checkpoint_dir / "final.pth"
    torch.save(
        {"model_state_dict": unet.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step},
        final_ckpt_path,
    )
    logger.save_checkpoint_meta(final_ckpt_path, step=step, epoch=step // max(1, len(loader)), parent_checkpoint=args.resume, seed=args.seed)
    logger.append_to_experiments_index(f"{args.scale} diffusion U-Net, {step} steps, final_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
