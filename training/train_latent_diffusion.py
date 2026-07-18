"""
Stage 3: Train the latent diffusion UNet.

Pipeline:
  - Mask latents (z_mask) are pre-cached on disk — run scripts/cache_latents.py first
  - ImageVAE is frozen and encodes masked MRI volumes on-the-fly (conditioning)
  - MaskVAE is frozen and decodes z_0 → mask predictions at val time
  - UNet3D learns to denoise z_mask conditioned on z_img

Operating in the 32³ latent space vs 128³ pixel space makes each step ~100×
cheaper than pixel-space diffusion.

Smoke-test (local / MPS):
    python3 -m training.train_latent_diffusion \\
        --latent_dir     data/brats_latents \\
        --image_data     data/brats_roi128_2023 \\
        --image_vae_ckpt checkpoints/image_vae_.../best.pth \\
        --mask_vae_ckpt  checkpoints/mask_vae_.../best.pth \\
        --splits_dir     splits/brats_roi128_smoke \\
        --base_channels  32 \\
        --channel_mults  1,2 \\
        --num_steps      200 \\
        --batch_size     2 \\
        --num_workers    0 \\
        --val_every      50 \\
        --device         mps

A100 full run (called automatically from run_a100.sh):
    python3 -m training.train_latent_diffusion \\
        --latent_dir     data/brats_latents \\
        --image_data     data/brats_roi128_2023 \\
        --image_vae_ckpt checkpoints/image_vae_.../best.pth \\
        --mask_vae_ckpt  checkpoints/mask_vae_.../best.pth \\
        --splits_dir     splits/brats_roi128_full \\
        --num_steps      200000 \\
        --batch_size     32 \\
        --num_workers    8 \\
        --device         cuda
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.brats_dataset import (
    apply_modality_mask,
    sample_modality_mask,
    seg_to_regions,
    subregions_to_regions,
)
from models.diffusion.config import DiffusionConfig
from models.diffusion.schedule import GaussianDiffusionSchedule, diffusion_loss
from models.diffusion.unet3d import UNet3D
from models.multiencoder.encoders import ImageVAE, MaskVAE
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "latent_diffusion"
REGION_NAMES = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Paths
    p.add_argument("--latent_dir",      required=True,
                   help="Folder with z_mask/ from cache_latents.py")
    p.add_argument("--image_data",      required=True,
                   help="ROI-cropped data folder with vol/ and seg/ (128³ .npy)")
    p.add_argument("--image_vae_ckpt",  required=True)
    p.add_argument("--mask_vae_ckpt",   required=True)
    p.add_argument("--splits_dir",      default=None,
                   help="Folder with train.txt / val.txt. Defaults to --latent_dir.")
    p.add_argument("--split_file",      default="train.txt")
    p.add_argument("--val_split_file",  default="val.txt")
    p.add_argument("--checkpoint_dir",  default="checkpoints")

    # UNet config
    p.add_argument("--latent_channels",    type=int, default=4)
    p.add_argument("--cond_proj_channels", type=int, default=64)
    p.add_argument("--base_channels",      type=int, default=64,
                   help="32 for local smoke test, 64 for A100.")
    p.add_argument("--channel_mults",      default="1,2,4")
    p.add_argument("--num_res_blocks",     type=int, default=2)
    p.add_argument("--num_timesteps",      type=int, default=1000)

    # Training
    p.add_argument("--num_steps",    type=int,   default=200_000)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)

    # Val / logging
    p.add_argument("--val_every",          type=int, default=1000)
    p.add_argument("--log_every",          type=int, default=100)
    p.add_argument("--ckpt_every",         type=int, default=5000)
    p.add_argument("--num_val_cases",      type=int, default=4)
    p.add_argument("--num_inference_steps",type=int, default=50,
                   help="DDIM steps for val sampling (50 is enough for Dice tracking).")

    p.add_argument("--resume",  default=None)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_names(path: str | Path) -> list[str]:
    with open(path) as f:
        return sorted(line.strip() for line in f if line.strip())


class LatentDiffusionDataset(Dataset):
    """Loads cached mask latents + raw volumes for on-the-fly ImageVAE encoding."""

    def __init__(self, z_mask_dir: Path, vol_dir: Path, names: list[str]) -> None:
        self.z_mask_dir = z_mask_dir
        self.vol_dir    = vol_dir
        self.names      = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name   = self.names[idx]
        z_mask = np.load(self.z_mask_dir / f"{name}_z_mask.npy")          # (4, 32, 32, 32)
        vol    = np.load(self.vol_dir    / f"{name}_vol.npy")              # (128,128,128,4)
        vol    = vol.transpose(3, 0, 1, 2).astype(np.float32)             # (4,128,128,128)
        return torch.from_numpy(z_mask.astype(np.float32)), torch.from_numpy(vol)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_image_vae(ckpt_path: str, device: torch.device) -> ImageVAE:
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg      = ckpt["config"]
    channels = tuple(int(c) for c in cfg["encoder_channels"].split(","))
    vae = ImageVAE(
        in_channels=4,
        channels=channels,
        num_res_units=cfg.get("encoder_num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_mask_vae(ckpt_path: str, device: torch.device) -> tuple[MaskVAE, bool]:
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg         = ckpt["config"]
    channels    = tuple(int(c) for c in cfg["mask_vae_channels"].split(","))
    subregion   = bool(cfg.get("subregion_mode", False))
    num_classes = 4 if subregion else 3
    vae = MaskVAE(
        num_classes=num_classes,
        latent_channels=cfg.get("latent_channels", 4),
        channels=channels,
        num_res_units=cfg.get("num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, subregion


# ---------------------------------------------------------------------------
# DDIM fast sampler (deterministic, eta=0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    unet:       UNet3D,
    schedule:   GaussianDiffusionSchedule,
    condition:  torch.Tensor,
    latent_shape: tuple[int, ...],
    *,
    device:     torch.device,
    n_steps:    int = 50,
) -> torch.Tensor:
    """Deterministic DDIM reverse process with `n_steps` uniformly-spaced timesteps."""
    T   = schedule.num_timesteps
    gap = max(T // n_steps, 1)
    # e.g. [999, 979, ..., 19, 0]  (inclusive of 0)
    ts  = list(reversed(range(0, T, gap)))

    x_t = torch.randn(latent_shape, device=device)
    ab  = schedule.alpha_bars  # (T,)

    for i, t_curr in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1

        t_batch = torch.full((latent_shape[0],), t_curr, device=device, dtype=torch.long)
        eps_pred = unet(x_t, t_batch, condition)

        ab_t    = ab[t_curr].float()
        # Predict clean x0
        x0_pred = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()

        if t_prev >= 0:
            ab_prev = ab[t_prev].float()
            x_t = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred
        else:
            x_t = x0_pred

    return x_t


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred * gt).sum()
    return float((2 * tp + eps) / (pred.sum() + gt.sum() + eps))


def run_val(
    unet:       UNet3D,
    image_vae:  ImageVAE,
    mask_vae:   MaskVAE,
    schedule:   GaussianDiffusionSchedule,
    val_names:  list[str],
    vol_dir:    Path,
    seg_dir:    Path,
    *,
    device:     torch.device,
    subregion:  bool,
    n_inf_steps: int,
    n_cases:    int,
) -> dict[str, float]:
    unet.eval()
    use_amp = (device.type == "cuda")

    all_dice: list[list[float]] = []

    for name in val_names[:n_cases]:
        vol = np.load(vol_dir / f"{name}_vol.npy")       # (H,W,D,4)
        seg = np.load(seg_dir / f"{name}_seg.npy")       # (H,W,D)

        vol_t = torch.from_numpy(
            vol.transpose(3, 0, 1, 2).astype(np.float32)
        ).unsqueeze(0).to(device)                        # (1,4,128,128,128)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                mu_img, _ = image_vae.encode(vol_t)      # (1, embed_dim, 32, 32, 32)

        z0 = ddim_sample(
            unet, schedule, mu_img,
            latent_shape=(1, mask_vae.latent_channels, *mu_img.shape[2:]),
            device=device,
            n_steps=n_inf_steps,
        )

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                logits = mask_vae.decode(z0.to(mask_vae.pre_decode.weight.dtype
                                               if hasattr(mask_vae, "pre_decode") else z0.dtype))
        pred = logits.sigmoid().squeeze(0).cpu().numpy()  # (C, H, W, D)

        if subregion:
            pred_regions = subregions_to_regions(pred)    # (3, H, W, D)
        else:
            pred_regions = (pred > 0.5).astype(np.float32)

        gt_regions = seg_to_regions(seg)                  # (3, H, W, D) [WT, TC, ET]

        case_dice = [dice_score(pred_regions[c], gt_regions[c]) for c in range(3)]
        all_dice.append(case_dice)

    mean_dice = np.mean(all_dice, axis=0)
    return {
        "wt_dice": float(mean_dice[0]),
        "tc_dice": float(mean_dice[1]),
        "et_dice": float(mean_dice[2]),
        "mean_dice": float(mean_dice.mean()),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_vis(
    unet:       UNet3D,
    image_vae:  ImageVAE,
    mask_vae:   MaskVAE,
    schedule:   GaussianDiffusionSchedule,
    val_names:  list[str],
    vol_dir:    Path,
    seg_dir:    Path,
    save_dir:   Path,
    step:       int,
    *,
    device:     torch.device,
    subregion:  bool,
    n_inf_steps: int,
    n_cases:    int = 4,
) -> None:
    unet.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (device.type == "cuda")
    COLOURS = {0: (0,0,0), 1: (0,0,200), 2: (0,200,0), 4: (200,0,0)}

    def _seg_rgb(arr3ch: np.ndarray) -> np.ndarray:
        # arr3ch: (3, S, S) — [WT, TC, ET] at one slice
        seg = np.zeros(arr3ch.shape[1:], dtype=np.uint8)
        seg[arr3ch[0] > 0.5] = 2   # ED
        seg[arr3ch[1] > 0.5] = 1   # NCR
        seg[arr3ch[2] > 0.5] = 4   # ET
        rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for v, c in COLOURS.items():
            rgb[seg == v] = c
        return rgb

    fig, axes = plt.subplots(n_cases, 2, figsize=(5, 2.5 * n_cases))
    if n_cases == 1:
        axes = [axes]

    for row, name in enumerate(val_names[:n_cases]):
        vol = np.load(vol_dir / f"{name}_vol.npy").transpose(3, 0, 1, 2).astype(np.float32)
        seg = np.load(seg_dir / f"{name}_seg.npy")
        gt_regions = seg_to_regions(seg)

        vol_t = torch.from_numpy(vol).unsqueeze(0).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                mu_img, _ = image_vae.encode(vol_t)

        z0 = ddim_sample(unet, schedule, mu_img,
                         latent_shape=(1, mask_vae.latent_channels, *mu_img.shape[2:]),
                         device=device, n_steps=n_inf_steps)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                logits = mask_vae.decode(z0)
        pred = logits.sigmoid().squeeze(0).cpu().numpy()
        if subregion:
            pred_regions = subregions_to_regions(pred)
        else:
            pred_regions = (pred > 0.5).astype(np.float32)

        # Pick best tumour slice (along last axis)
        z_idx = int(gt_regions[0].sum(axis=(0, 1)).argmax())

        axes[row][0].imshow(_seg_rgb(gt_regions[:, :, :, z_idx]))
        axes[row][0].set_title(f"GT  {name[:12]}", fontsize=7)
        axes[row][0].axis("off")

        axes[row][1].imshow(_seg_rgb(pred_regions[:, :, :, z_idx]))
        axes[row][1].set_title("Generated", fontsize=7)
        axes[row][1].axis("off")

    plt.suptitle(f"Step {step}", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / f"step_{step:07d}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    latent_dir  = Path(args.latent_dir)
    vol_dir     = Path(args.image_data) / "vol"
    seg_dir     = Path(args.image_data) / "seg"
    splits_dir  = Path(args.splits_dir) if args.splits_dir else latent_dir

    # ── Load frozen VAEs ─────────────────────────────────────────────────────
    print("Loading ImageVAE …")
    image_vae = _load_image_vae(args.image_vae_ckpt, device)
    embed_dim = image_vae.embed_dim
    print(f"  embed_dim = {embed_dim}")

    print("Loading MaskVAE …")
    mask_vae, subregion = _load_mask_vae(args.mask_vae_ckpt, device)
    print(f"  latent_channels = {mask_vae.latent_channels}  "
          f"subregion = {subregion}")

    # ── Build UNet + schedule ────────────────────────────────────────────────
    channel_mults = tuple(int(m) for m in args.channel_mults.split(","))
    cfg = DiffusionConfig(
        latent_channels      = args.latent_channels,
        condition_channels   = embed_dim,
        cond_proj_channels   = args.cond_proj_channels,
        base_channels        = args.base_channels,
        channel_multipliers  = channel_mults,
        num_res_blocks_per_level = args.num_res_blocks,
        num_timesteps        = args.num_timesteps,
    )
    unet     = UNet3D(cfg).to(device)
    schedule = GaussianDiffusionSchedule(
        num_timesteps=args.num_timesteps
    ).to(device)

    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"UNet3D: {n_params:.1f}M params")

    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_names = _read_names(splits_dir / args.split_file)
    val_names   = _read_names(splits_dir / args.val_split_file)

    train_ds = LatentDiffusionDataset(latent_dir / "z_mask", vol_dir, train_names)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Run setup ────────────────────────────────────────────────────────────
    run_id  = new_run_id(STAGE)
    ckpt_dir = Path(args.checkpoint_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir  = ckpt_dir / "visualisations"

    logger = RunLogger(run_id, STAGE, config=vars(args))
    logger.note(
        f"ImageVAE {args.image_vae_ckpt} | MaskVAE {args.mask_vae_ckpt} | "
        f"embed_dim={embed_dim} | subregion={subregion} | unet={n_params:.1f}M"
    )

    print(f"\nTrain: {len(train_names)} cases  Val: {len(val_names)} cases")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Steps: {args.num_steps}  Batch: {args.batch_size}\n")

    # ── Resume ───────────────────────────────────────────────────────────────
    step = 0
    best_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step      = ckpt.get("step", 0)
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"Resumed from step {step}, best_dice={best_dice:.4f}")

    # ── Training loop ────────────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    dl_iter = iter(train_dl)

    with tqdm(total=args.num_steps, initial=step, desc="training") as pbar:
        while step < args.num_steps:
            # Refill iterator when exhausted
            try:
                z_mask, vol = next(dl_iter)
            except StopIteration:
                dl_iter = iter(train_dl)
                z_mask, vol = next(dl_iter)

            z_mask = z_mask.to(device)   # (B, 4, 32, 32, 32)
            vol    = vol.to(device)      # (B, 4, 128, 128, 128)

            # Random modality dropout — simulate missing-modality conditioning
            mod_mask = sample_modality_mask(vol.shape[0]).to(device)
            vol_masked = apply_modality_mask(vol, mod_mask)

            unet.train()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                # On-the-fly ImageVAE encoding (frozen)
                with torch.no_grad():
                    mu_img, _ = image_vae.encode(vol_masked)   # (B, embed_dim, 32, 32, 32)

                loss = diffusion_loss(unet, schedule, z_mask, mu_img)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                logger.log_metrics(step, csv="train", loss=loss.item())

            # ── Validation ──────────────────────────────────────────────────
            if step % args.val_every == 0:
                metrics = run_val(
                    unet, image_vae, mask_vae, schedule,
                    val_names, vol_dir, seg_dir,
                    device=device,
                    subregion=subregion,
                    n_inf_steps=args.num_inference_steps,
                    n_cases=args.num_val_cases,
                )
                logger.log_metrics(step, csv="val", **metrics)

                mean_d = metrics["mean_dice"]
                print(
                    f"\n[step {step}] "
                    f"WT={metrics['wt_dice']:.3f}  "
                    f"TC={metrics['tc_dice']:.3f}  "
                    f"ET={metrics['et_dice']:.3f}  "
                    f"mean={mean_d:.3f}"
                    f"{'  ★ best' if mean_d > best_dice else ''}"
                )

                if mean_d > best_dice:
                    best_dice = mean_d
                    torch.save({
                        "unet_state_dict":      unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step":                 step,
                        "best_dice":            best_dice,
                        "config":               vars(args),
                        "image_vae_ckpt":       args.image_vae_ckpt,
                        "mask_vae_ckpt":        args.mask_vae_ckpt,
                    }, ckpt_dir / "best.pth")

                save_vis(
                    unet, image_vae, mask_vae, schedule,
                    val_names, vol_dir, seg_dir, vis_dir, step,
                    device=device, subregion=subregion,
                    n_inf_steps=args.num_inference_steps,
                )
                unet.train()

            # ── Periodic checkpoint ─────────────────────────────────────────
            if step % args.ckpt_every == 0:
                torch.save({
                    "unet_state_dict":      unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step":                 step,
                    "best_dice":            best_dice,
                    "config":               vars(args),
                    "image_vae_ckpt":       args.image_vae_ckpt,
                    "mask_vae_ckpt":        args.mask_vae_ckpt,
                }, ckpt_dir / f"step_{step}.pth")

    # ── Final checkpoint ─────────────────────────────────────────────────────
    final_path = ckpt_dir / "final.pth"
    torch.save({
        "unet_state_dict":      unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 step,
        "best_dice":            best_dice,
        "config":               vars(args),
        "image_vae_ckpt":       args.image_vae_ckpt,
        "mask_vae_ckpt":        args.mask_vae_ckpt,
    }, final_path)

    logger.append_to_experiments_index(
        f"latent diffusion, {step} steps, best_dice={best_dice:.4f}, "
        f"unet={n_params:.1f}M, embed={embed_dim}, subregion={subregion}"
    )
    print(f"\nDone. best_dice={best_dice:.4f}  saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
