"""
Stage 0: Standalone Image VAE pretraining.

Trains the ImageVAE to reconstruct present MRI modalities from a
randomly-masked input. Missing modalities are zeroed before encoding and
contribute zero loss — the VAE only learns to compress what it can see.

The KL term forces sigma to reflect available information:
  - all 4 modalities present  → small sigma (confident)
  - only 1 modality present   → large sigma (uncertain)

This sigma is the uncertainty signal propagated through to the diffusion
U-Net when sampling N plausible segmentation masks at evaluation time.

Pretraining this VAE as a standalone stage (rather than end-to-end with
the U-Net) gives cleaner separation of concerns: the VAE learns to
represent MRI well, then the U-Net learns to segment from those features.

Metrics logged per epoch:
  - recon_loss (L1, present channels only)
  - kl_loss    (KL divergence)
  - total_loss
  - SSIM and PSNR per modality (on a fixed val set)

Run from repo root:
    python3 -m training.train_image_vae --data_root data/brats2023_processed
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask
from utils.early_stopping import EarlyStopping
from models.multiencoder.encoders import ImageVAE, vae_loss
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "image_vae"
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Processed data folder containing vol/ and seg/. Never modified.")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Folder with train.txt/val.txt/test.txt (from resplit_data.py). "
                             "Defaults to --data_root if not set.")
    parser.add_argument("--split_file", type=str, default="train.txt")
    parser.add_argument("--val_split_file", type=str, default="val.txt")
    parser.add_argument("--test_split_file", type=str, default="test.txt",
                        help="Test split used for reconstruction visualisation only (not metrics).")
    parser.add_argument("--crop_size", type=int, default=96)

    # Model
    parser.add_argument("--encoder_channels", type=str, default="64,128,256,256",
                        help="Comma-separated channel progression, e.g. '64,128,256,256'. "
                             "Last value becomes the bottleneck (embed) dim. "
                             "All-but-last stages use stride 2; last uses stride 1.")
    parser.add_argument("--encoder_num_res_units", type=int, default=2,
                        help="Residual units per encoder/decoder block. "
                             "More depth without extra downsampling. 2-3 is typical.")

    # VAE loss
    parser.add_argument("--vae_beta", type=float, default=1e-4,
                        help="Weight on KL term. Increase to 1e-3 if sigma collapses to zero.")

    # Patch masking (optional, MAE-style)
    parser.add_argument("--patch_mask_ratio", type=float, default=0.0,
                        help="Fraction of spatial patches to mask within present modalities. "
                             "0.0 disables. Try 0.5 for MAE-style training.")
    parser.add_argument("--patch_size", type=int, default=8)

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    # Logging / checkpointing
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=200,
                        help="Run SSIM/PSNR validation every N steps")
    parser.add_argument("--num_val_cases", type=int, default=5)

    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="Val checks with no SSIM improvement before stopping. "
                             "0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--num_test_vis_cases", type=int, default=4,
                        help="How many test cases to include in the post-training reconstruction grid")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Patch masking (same as train_pixel_test)
# ---------------------------------------------------------------------------

def apply_patch_mask(
    volume: torch.Tensor,
    patch_size: int = 8,
    mask_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, D, H, W = volume.shape
    gD, gH, gW = D // patch_size, H // patch_size, W // patch_size
    keep = (torch.rand(B, 1, gD, gH, gW, device=volume.device) > mask_ratio).float()
    patch_mask = F.interpolate(keep, size=(D, H, W), mode="nearest")
    present_mask = (volume.abs().sum(dim=1, keepdim=True) > 0).float()
    patch_mask = patch_mask * present_mask + (1 - present_mask)
    return volume * patch_mask, patch_mask


# ---------------------------------------------------------------------------
# Reconstruction quality metrics (SSIM and PSNR)
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def ssim_3d(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> float:
    """Approximate SSIM via sliding window mean/variance on 3D volumes."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel = torch.ones(1, 1, window_size, window_size, window_size,
                        device=pred.device) / (window_size ** 3)

    def conv(x):
        return F.conv3d(x, kernel, padding=window_size // 2)

    mu_x = conv(pred)
    mu_y = conv(target)
    mu_xx = conv(pred * pred)
    mu_yy = conv(target * target)
    mu_xy = conv(pred * target)

    sigma_x  = mu_xx - mu_x * mu_x
    sigma_y  = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float((numerator / denominator).mean().item())


def _best_slice(vol: np.ndarray) -> int:
    """Pick the D-axis slice with the most signal. vol shape: (C, H, W, D)."""
    signal_per_slice = vol.sum(axis=(0, 1, 2))  # (D,)
    return int(signal_per_slice.argmax())


@torch.no_grad()
def save_recon_visualisation(
    vae: ImageVAE,
    val_cases: list[torch.Tensor],
    device: torch.device,
    save_path: Path,
) -> None:
    """
    Save a PNG grid comparing original vs reconstructed MRI slices.

    Layout:
      Rows  = val cases
      Cols  = [T1_orig | T1_recon | T1ce_orig | T1ce_recon | T2_orig | T2_recon | FLAIR_orig | FLAIR_recon]

    Masked (missing) modalities are shown as grey panels labelled "MISSING".
    """
    vae.eval()
    n_cases = len(val_cases)
    n_cols = len(MODALITY_NAMES) * 2   # orig + recon per modality
    col_labels = []
    for m in MODALITY_NAMES:
        col_labels += [f"{m}\norig", f"{m}\nrecon"]

    fig, axes = plt.subplots(
        n_cases, n_cols,
        figsize=(n_cols * 2.2, n_cases * 2.2),
        squeeze=False,
    )
    fig.suptitle(f"VAE Reconstruction  |  step={save_path.stem.split('_')[-1]}", fontsize=10)

    for row, vol in enumerate(val_cases):
        vol = vol.to(device)                       # (1, 4, H, W, D)

        mu, logvar = vae.encode(vol)
        recon = vae.decode(mu)                     # deterministic, all modalities present

        vol_np   = vol[0].cpu().float().numpy()    # (4, H, W, D)
        recon_np = recon[0].cpu().float().numpy()
        mask_np  = np.ones(len(MODALITY_NAMES), dtype=bool)  # all present

        # pick best axial slice from the channel with most signal
        best_z = _best_slice(vol_np)

        for c, name in enumerate(MODALITY_NAMES):
            col_orig  = c * 2
            col_recon = c * 2 + 1
            ax_orig  = axes[row][col_orig]
            ax_recon = axes[row][col_recon]

            orig_slice = vol_np[c, :, :, best_z]       # (H, W)

            # normalise to [0, 1] for display
            vmin, vmax = orig_slice.min(), orig_slice.max()
            if vmax - vmin > 1e-6:
                orig_disp = (orig_slice - vmin) / (vmax - vmin)
            else:
                orig_disp = orig_slice

            ax_orig.imshow(orig_disp, cmap="gray", vmin=0, vmax=1)
            ax_orig.set_title(col_labels[col_orig], fontsize=7)
            ax_orig.axis("off")

            if mask_np[c]:                            # modality was present
                recon_slice = recon_np[c, :, :, best_z]
                recon_disp = (recon_slice - vmin) / (vmax - vmin + 1e-6)
                ax_recon.imshow(recon_disp, cmap="gray", vmin=0, vmax=1)
                ax_recon.set_title(col_labels[col_recon], fontsize=7)
            else:                                     # modality was masked/missing
                ax_recon.imshow(np.ones_like(orig_disp) * 0.5, cmap="gray", vmin=0, vmax=1)
                ax_recon.set_title(col_labels[col_recon], fontsize=7, color="red")
                ax_recon.text(
                    0.5, 0.5, "MISSING",
                    transform=ax_recon.transAxes,
                    ha="center", va="center",
                    fontsize=8, color="red", fontweight="bold",
                )
            ax_recon.axis("off")

        # row label: case index + number of present modalities
        n_present = int(mask_np.sum())
        axes[row][0].set_ylabel(f"case {row}\n({n_present}/4 mod)", fontsize=7)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {save_path}")


@torch.no_grad()
def run_val_metrics(
    vae: ImageVAE,
    val_cases: list[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Compute per-modality SSIM and PSNR on a fixed set of val volumes.
    Only computes metrics on present (non-zero) modality channels per case.
    Returns dict of {T1_ssim, T1_psnr, T1ce_ssim, ...} plus overall means.
    """
    vae.eval()
    per_modality_ssim = {m: [] for m in MODALITY_NAMES}
    per_modality_psnr = {m: [] for m in MODALITY_NAMES}

    for vol in val_cases:
        vol = vol.to(device)                          # (1, 4, D, H, W)

        # random modality mask (same distribution as training)
        modality_mask = sample_modality_mask(1).to(device)
        volume_masked = apply_modality_mask(vol, modality_mask)

        mu, logvar = vae.encode(volume_masked)
        z = mu                                        # deterministic at val
        recon = vae.decode(z)                         # (1, 4, D, H, W)

        for c, name in enumerate(MODALITY_NAMES):
            if volume_masked[0, c].abs().sum() == 0:
                continue                              # skip missing modalities
            p = recon[0, c].unsqueeze(0).unsqueeze(0)
            t = vol[0, c].unsqueeze(0).unsqueeze(0)
            # normalise to [0,1] per-volume for consistent metric range
            t_min, t_max = t.min(), t.max()
            if (t_max - t_min) < 1e-6:
                continue
            p_norm = (p - t_min) / (t_max - t_min)
            t_norm = (t - t_min) / (t_max - t_min)
            per_modality_ssim[name].append(ssim_3d(p_norm, t_norm))
            per_modality_psnr[name].append(psnr(p_norm, t_norm))

    metrics = {}
    all_ssim, all_psnr = [], []
    for name in MODALITY_NAMES:
        if per_modality_ssim[name]:
            s = float(np.mean(per_modality_ssim[name]))
            p = float(np.mean(per_modality_psnr[name]))
        else:
            s, p = 0.0, 0.0
        metrics[f"{name}_ssim"] = s
        metrics[f"{name}_psnr"] = p
        all_ssim.append(s)
        all_psnr.append(p)

    metrics["mean_ssim"] = float(np.mean(all_ssim))
    metrics["mean_psnr"] = float(np.mean(all_psnr))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    splits_dir = args.splits_dir if args.splits_dir is not None else args.data_root

    encoder_channels = tuple(int(c) for c in args.encoder_channels.split(","))
    vae = ImageVAE(
        in_channels=4,
        channels=encoder_channels,
        num_res_units=args.encoder_num_res_units,
    ).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        vae.load_state_dict(ckpt["vae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.split_file),
        crop_size=args.crop_size,
        volume_only=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # fixed val cases for periodic SSIM/PSNR evaluation
    val_cases: list[torch.Tensor] = []
    val_split_path = os.path.join(splits_dir, args.val_split_file)
    if os.path.exists(val_split_path):
        val_dataset = BraTSDataset(
            root=args.data_root,
            split_file=val_split_path,
            crop_size=args.crop_size,
            random_crop=False,
            volume_only=True,
        )
        for i in range(min(args.num_val_cases, len(val_dataset))):
            vol, _ = val_dataset[i]
            val_cases.append(vol.unsqueeze(0))
    else:
        print(f"  [warn] val split not found at {val_split_path}, skipping SSIM/PSNR eval")

    # test cases — used only for reconstruction visualisation
    test_cases: list[torch.Tensor] = []
    test_split_path = os.path.join(splits_dir, args.test_split_file)
    if os.path.exists(test_split_path):
        test_dataset = BraTSDataset(
            root=args.data_root,
            split_file=test_split_path,
            crop_size=args.crop_size,
            random_crop=False,
            volume_only=True,
        )
        for i in range(min(args.num_test_vis_cases, len(test_dataset))):
            vol, _ = test_dataset[i]
            test_cases.append(vol.unsqueeze(0))
        print(f"Test cases loaded for visualisation: {len(test_cases)}")
    else:
        print(f"  [warn] test split not found at {test_split_path}, skipping visualisation")

    run_id = new_run_id(STAGE)
    logger = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(loader)
    start_epoch = start_step // max(1, steps_per_epoch)
    total_steps = args.num_epochs * steps_per_epoch

    print(f"Training cases : {len(dataset)}  |  steps/epoch : {steps_per_epoch}  |  total steps : {total_steps}")

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    ) if args.early_stop_patience > 0 else None

    vae.train()
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc=f"[{STAGE}]", unit="step")

    for epoch in range(start_epoch, args.num_epochs):
        tqdm.write(f"\n[{STAGE}] Epoch {epoch + 1}/{args.num_epochs}")

        for volume, _ in loader:
            volume = volume.to(device)                        # (B, 4, 96, 96, 96)

            # random missing modalities
            modality_mask = sample_modality_mask(volume.shape[0]).to(device)
            volume_masked = apply_modality_mask(volume, modality_mask)

            # optional spatial patch masking
            if args.patch_mask_ratio > 0.0:
                encoder_input, patch_mask = apply_patch_mask(
                    volume_masked, args.patch_size, args.patch_mask_ratio
                )
                masked_region = (1 - patch_mask)
                recon_target = volume_masked * masked_region
            else:
                encoder_input = volume_masked
                recon_target = volume_masked
                masked_region = None

            # VAE forward
            recon, mu, logvar = vae(encoder_input)

            if masked_region is not None:
                recon = recon * masked_region

            total, r_loss, k_loss = vae_loss(recon, recon_target, mu, logvar, beta=args.vae_beta)
            # reconstruction_loss inside vae_loss uses MSE (Gaussian decoder assumption)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(
                epoch=f"{epoch + 1}/{args.num_epochs}",
                loss=f"{total.item():.4f}",
                recon=f"{r_loss.item():.4f}",
                kl=f"{k_loss.item():.6f}",
            )

            if step % args.log_every == 0:
                logger.log_metrics(
                    step=step, csv="train",
                    epoch=epoch + 1,
                    total_loss=total.item(),
                    recon_loss=r_loss.item(),
                    kl_loss=k_loss.item(),
                )
                tqdm.write(
                    f"[{STAGE}] epoch={epoch+1}  step={step}  "
                    f"total={total.item():.4f}  recon={r_loss.item():.4f}  "
                    f"kl={k_loss.item():.6f}"
                )

            if step % args.val_every == 0 and step > start_step and val_cases:
                metrics = run_val_metrics(vae, val_cases, device)
                logger.log_metrics(step=step, csv="val", **metrics)
                es_status = ""
                if early_stopper is not None:
                    should_stop = early_stopper.step(metrics["mean_ssim"], step)
                    es_status = f"  [{early_stopper.status}]"
                    if should_stop:
                        tqdm.write(f"  [early stop] no SSIM improvement for {early_stopper.patience} checks — stopping.")
                        pbar.close()
                        return

                tqdm.write(
                    f"  [val]  SSIM — "
                    + "  ".join(f"{m}={metrics[f'{m}_ssim']:.3f}" for m in MODALITY_NAMES)
                    + f"  mean={metrics['mean_ssim']:.3f}"
                    + es_status
                )
                tqdm.write(
                    f"  [val]  PSNR — "
                    + "  ".join(f"{m}={metrics[f'{m}_psnr']:.1f}" for m in MODALITY_NAMES)
                    + f"  mean={metrics['mean_psnr']:.1f} dB"
                )
                vae.train()

            if step % args.ckpt_every == 0 and step > start_step:
                ckpt_path = checkpoint_dir / f"step_{step}.pth"
                torch.save({
                    "vae_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": vars(args),
                }, ckpt_path)

            step += 1
            pbar.update(1)

    pbar.close()
    final_path = checkpoint_dir / "final.pth"
    torch.save({
        "vae_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": args.num_epochs,
        "config": vars(args),
    }, final_path)
    print(f"\nDone. Checkpoint saved → {final_path}")

    # --- Test evaluation: reconstruction visualisation on held-out test cases ---
    if test_cases:
        print("\nRunning reconstruction visualisation on test set...")
        vis_path = logger.run_dir / "vis" / "test_recon_final.png"
        save_recon_visualisation(vae, test_cases, device, vis_path)
        print(f"Test reconstruction grid saved → {vis_path}")
    else:
        print("  [warn] No test cases found — skipping test visualisation.")

    logger.append_to_experiments_index(
        f"Image VAE, {args.num_epochs} epochs ({step} steps), "
        f"channels={args.encoder_channels}, res_units={args.encoder_num_res_units}, "
        f"beta={args.vae_beta}, final_recon={r_loss.item():.4f}"
    )
    print(f"Use --image_vae_checkpoint {final_path} in train_pixel_test.py")


if __name__ == "__main__":
    main()
