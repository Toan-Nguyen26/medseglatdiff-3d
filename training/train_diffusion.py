"""
Pixel-space conditioned diffusion for BraTS segmentation.

No VAE, no encoder. The MRI volume and noisy mask are concatenated and passed
directly into a single 3D U-Net that predicts the noise.

  Training (DDPM):
    1. Crop 96³ patch (70% ROI-centred, 30% random)
    2. Simulate missing modality: zero out absent channels
    3. Scale GT mask {0,1} → {-1,+1}
    4. Sample t, add noise: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    5. x_input = cat(volume_masked, x_t)   — (B, 7, 96, 96, 96)
    6. UNet(x_input, t) → ε_pred
    7. Loss = MSE(ε_pred, ε)

  Inference (DDIM):
    x_T ~ N(0,I)
    for t = T..0:
        x_input = cat(volume_masked, x_t)
        x_{t-1} = DDIM_step(UNet(x_input, t), t, x_t)
    pred = (x_0 + 1) / 2 > 0.5   ← map {-1,+1} back to binary

Run from repo root:
    python3 -m training.train_diffusion \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_20pct \\
        --device mps
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

from data.brats_dataset import (
    BraTSDataset, apply_modality_mask, sample_modality_mask, seg_to_regions,
)
from models.diffusion.config import PixelDiffusionConfig
from models.diffusion.unet3d import PixelUNet3D
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "diffusion"
REGION_NAMES = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data_root",     required=True)
    p.add_argument("--splits_dir",    default=None)
    p.add_argument("--split_file",    default="train.txt")
    p.add_argument("--val_split_file",default="val.txt")
    p.add_argument("--crop_size",     type=int, default=96)

    # ROI crop
    p.add_argument("--roi_crop_ratio",  type=float, default=0.7,
                   help="Fraction of training crops centred on tumor ROI (rest random).")
    p.add_argument("--roi_max_offset",  type=int,   default=20,
                   help="Max voxel offset from tumour centroid for ROI crops.")

    # Model
    p.add_argument("--base_channels",  type=int, default=32,
                   help="UNet base channels. 32 for MPS smoke test, 64 for H200.")
    p.add_argument("--channel_mults",  type=str, default="1,2,4",
                   help="Comma-separated channel multipliers for UNet depth.")

    # Diffusion schedule
    p.add_argument("--num_timesteps",       type=int, default=1000)
    p.add_argument("--num_inference_steps", type=int, default=10,
                   help="DDIM steps during validation (10 is fast; use 50 for final eval).")

    # Training
    p.add_argument("--num_epochs",   type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)

    # Logging
    p.add_argument("--log_every",      type=int, default=50)
    p.add_argument("--val_every",      type=int, default=200)
    p.add_argument("--num_val_cases",  type=int, default=3)
    p.add_argument("--ckpt_every",     type=int, default=1000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume",         type=str, default=None)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


@torch.no_grad()
def val_dice(
    unet: PixelUNet3D,
    val_cases: list[tuple[torch.Tensor, np.ndarray]],
    num_inference_steps: int,
    num_timesteps: int,
    device: torch.device,
) -> dict[str, float]:
    unet.eval()
    inf_scheduler = DDIMScheduler(num_train_timesteps=num_timesteps)
    inf_scheduler.set_timesteps(num_inference_steps)

    scores: dict[str, list[float]] = {r: [] for r in REGION_NAMES}

    for vol, gt_regions_np in val_cases:
        vol = vol.to(device)                          # (1, 4, 96, 96, 96)
        spatial = vol.shape[2:]

        # Pick one random missing-modality scenario per val case
        mod_mask = sample_modality_mask(1).to(device)
        vol_in = apply_modality_mask(vol, mod_mask)

        # DDIM denoising
        x = torch.randn(1, 3, *spatial, device=device)
        for t in inf_scheduler.timesteps:
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            x_in = torch.cat([vol_in, x], dim=1)     # (1, 7, 96, 96, 96)
            pred_noise = unet(x_in, t_batch)
            x = inf_scheduler.step(pred_noise, t, x)[0].to(device).float()

        # x is in [-1, +1] → map to [0, 1] → threshold
        pred_bin = ((x[0] + 1.0) / 2.0 > 0.5).cpu().numpy()  # (3, D, H, W)

        for i, r in enumerate(REGION_NAMES):
            scores[r].append(_dice(pred_bin[i].astype(bool),
                                   gt_regions_np[i].astype(bool)))

    metrics = {r: float(np.mean(scores[r])) for r in REGION_NAMES}
    metrics["mean"] = float(np.mean(list(metrics.values())))
    unet.train()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device     = torch.device(args.device)
    splits_dir = args.splits_dir or args.data_root

    channel_mults = tuple(int(x) for x in args.channel_mults.split(","))
    config = PixelDiffusionConfig(
        base_channels=args.base_channels,
        channel_multipliers=channel_mults,
        num_timesteps=args.num_timesteps,
    )
    unet = PixelUNet3D(config).to(device)
    print(f"PixelUNet3D  in={config.in_channels}ch  out={config.mask_channels}ch  "
          f"base={config.base_channels}  mults={channel_mults}")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.split_file),
        crop_size=args.crop_size,
        region_based=True,
        roi_crop_ratio=args.roi_crop_ratio,
        roi_max_offset=args.roi_max_offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Training cases : {len(dataset)}  |  ROI ratio={args.roi_crop_ratio}")

    # Fixed val cases (centre crop, no ROI, no modality masking yet)
    val_cases: list[tuple[torch.Tensor, np.ndarray]] = []
    val_path = os.path.join(splits_dir, args.val_split_file)
    if os.path.exists(val_path):
        val_ds = BraTSDataset(
            root=args.data_root,
            split_file=val_path,
            crop_size=args.crop_size,
            region_based=True,
            random_crop=False,
        )
        for i in range(min(args.num_val_cases, len(val_ds))):
            vol, gt = val_ds[i]
            val_cases.append((vol.unsqueeze(0), gt.numpy()))
    else:
        print(f"  [warn] val split not found at {val_path}")

    run_id         = new_run_id(STAGE)
    logger         = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(loader)
    start_epoch     = start_step // max(1, steps_per_epoch)
    total_steps     = args.num_epochs * steps_per_epoch
    best_mean_dice  = 0.0

    print(f"Total steps    : {total_steps}")

    unet.train()
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step,
                desc=f"[{STAGE}]", unit="step")

    for epoch in range(start_epoch, args.num_epochs):
        tqdm.write(f"\n[{STAGE}] Epoch {epoch + 1}/{args.num_epochs}")

        for _, gt_regions in loader:
            volume    = _.to(device)               # (B, 4, 96, 96, 96)
            gt_regions = gt_regions.to(device)     # (B, 3, 96, 96, 96) binary {0,1}

            # Simulate missing modalities
            mod_mask       = sample_modality_mask(volume.shape[0]).to(device)
            volume_masked  = apply_modality_mask(volume, mod_mask)

            # Scale GT {0,1} → {-1,+1} so signal std ≈ noise std
            x_0 = gt_regions * 2.0 - 1.0

            # DDPM forward process
            epsilon = torch.randn_like(x_0)
            t = torch.randint(0, args.num_timesteps, (volume.shape[0],), device=device)
            x_t = noise_scheduler.add_noise(x_0, epsilon, t)

            # Concat image + noisy mask → predict noise
            x_input    = torch.cat([volume_masked, x_t], dim=1)   # (B, 7, 96, 96, 96)
            pred_noise = unet(x_input, t)
            loss       = F.mse_loss(pred_noise, epsilon)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(epoch=f"{epoch+1}/{args.num_epochs}",
                             loss=f"{loss.item():.4f}")

            if step % args.log_every == 0:
                logger.log_metrics(step=step, csv="train",
                                   epoch=epoch+1, loss=loss.item())
                tqdm.write(f"[{STAGE}] epoch={epoch+1}  step={step}  "
                           f"loss={loss.item():.4f}")

            if step % args.val_every == 0 and step > start_step and val_cases:
                metrics = val_dice(unet, val_cases,
                                   args.num_inference_steps,
                                   args.num_timesteps, device)
                logger.log_metrics(step=step, csv="val", **metrics)

                is_best = metrics["mean"] > best_mean_dice
                if is_best:
                    best_mean_dice = metrics["mean"]
                    torch.save({
                        "unet_state_dict":      unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step, "epoch": epoch,
                        "config": vars(args),
                        "best_mean_dice": best_mean_dice,
                    }, checkpoint_dir / "best.pth")

                tqdm.write(
                    f"  [val]  Dice — "
                    + "  ".join(f"{r}={metrics[r]:.3f}" for r in REGION_NAMES)
                    + f"  mean={metrics['mean']:.3f}"
                    + ("  ← best" if is_best else "")
                )

            if step % args.ckpt_every == 0 and step > start_step:
                torch.save({
                    "unet_state_dict":      unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "epoch": epoch,
                    "config": vars(args),
                }, checkpoint_dir / f"step_{step}.pth")

            step += 1
            pbar.update(1)

    pbar.close()
    torch.save({
        "unet_state_dict":      unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "epoch": args.num_epochs,
        "config": vars(args),
    }, checkpoint_dir / "final.pth")
    print(f"\nDone. Checkpoint → {checkpoint_dir}/final.pth")
    logger.append_to_experiments_index(
        f"pixel diffusion, {args.num_epochs} epochs ({step} steps), "
        f"base_ch={args.base_channels}, final_loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
