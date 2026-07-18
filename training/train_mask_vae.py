"""
Stage 1b: Standalone Mask VAE pretraining.

Trains a MaskVAE to compress binary segmentation masks (WT/TC/ET) at 96³
into a small latent space (latent_channels=4, 12³).  KL regularisation
forces z ~ N(0, 1) so the latent is perfectly matched to DDPM Gaussian
noise — no logit transform needed in the diffusion stage.

Loss = Dice + BCE  (reconstruction)  +  beta * KL

The decoder outputs are used at diffusion inference time to convert the
denoised latent z_0 back to full-resolution binary masks.

Run from repo root:
    python3 -m training.train_mask_vae --data_root data/brats2023_processed

Then pass the resulting checkpoint to train_pixel_test.py:
    python3 -m training.train_pixel_test \\
        --mask_vae_checkpoint checkpoints/mask_vae_.../final.pth \\
        --diffusion ...
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

from data.brats_dataset import BraTSDataset, regions_to_seg, subregions_to_regions
from models.multiencoder.encoders import MaskVAE, mask_vae_loss
from utils.run_logger import RunLogger, new_run_id, set_seed
from utils.early_stopping import EarlyStopping

STAGE = "mask_vae"
REGION_NAMES = ["WT", "TC", "ET"]
LABEL_COLOURS = {0: (0, 0, 0), 1: (0, 0, 200), 2: (0, 200, 0), 4: (200, 0, 0)}

# pos_weight for 4-channel subregion mode [BG, NCR, ED, ET]
# NCR is ~2% of voxels → upweight heavily so it's not ignored
SUBREGION_POS_WEIGHT = torch.tensor([0.1, 10.0, 3.0, 5.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data_root",     type=str, required=True)
    parser.add_argument("--splits_dir",    type=str, default=None)
    parser.add_argument("--split_file",    type=str, default="train.txt")
    parser.add_argument("--val_split_file",type=str, default="val.txt")
    parser.add_argument("--crop_size",     type=int, default=96)

    parser.add_argument("--latent_channels",   type=int, default=4,
                        help="Latent space channels. 4 matches MedSegLatDiff.")
    parser.add_argument("--mask_vae_channels", type=str, default="32,64,128,128",
                        help="Comma-separated channel progression for MaskVAE encoder/decoder.")
    parser.add_argument("--num_res_units",     type=int, default=2)
    parser.add_argument("--subregion_mode",    action="store_true",
                        help="Predict 4-ch [BG,NCR,ED,ET] instead of 3-ch [WT,TC,ET]. "
                             "Gives NCR its own channel and loss gradient.")
    parser.add_argument("--vae_beta",          type=float, default=1e-2,
                        help="Final KL weight after annealing completes. "
                             "Needs to be ~1e-2 so KL is visible vs recon (~1.5). "
                             "Reduce to 1e-3 if reconstruction Dice drops too much.")
    parser.add_argument("--kl_warmup_steps",   type=int, default=1000,
                        help="Linearly ramp beta from 0 → vae_beta over this many steps. "
                             "Eliminates recon/KL oscillation early in training.")

    parser.add_argument("--num_epochs",    type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=2)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int, default=2)

    parser.add_argument("--log_every",     type=int, default=50)
    parser.add_argument("--ckpt_every",    type=int, default=1000)
    parser.add_argument("--val_every",     type=int, default=200)
    parser.add_argument("--num_val_cases", type=int, default=4)

    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="Val checks with no Dice improvement before stopping. "
                             "0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir",type=str, default="checkpoints")
    parser.add_argument("--resume",        type=str, default=None)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation metrics — Dice per region
# ---------------------------------------------------------------------------

def _dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


@torch.no_grad()
def run_val_metrics(
    vae: MaskVAE,
    val_cases: list[torch.Tensor],
    device: torch.device,
    subregion_mode: bool = False,
) -> dict[str, float]:
    vae.eval()
    scores: dict[str, list[float]] = {r: [] for r in REGION_NAMES}

    for gt in val_cases:
        gt = gt.to(device)
        recon_logits, _, _ = vae(gt)
        recon_bin = (recon_logits.sigmoid() > 0.5).cpu().numpy()[0]  # (C, D, H, W)
        gt_np     = gt.cpu().numpy()[0]

        if subregion_mode:
            # convert 4-ch [BG,NCR,ED,ET] → 3-ch [WT,TC,ET] for metric computation
            recon_regions = subregions_to_regions(recon_bin)
            gt_regions    = subregions_to_regions(gt_np)
        else:
            recon_regions = recon_bin
            gt_regions    = gt_np

        for i, r in enumerate(REGION_NAMES):
            scores[r].append(_dice(recon_regions[i].astype(bool), gt_regions[i].astype(bool)))

    metrics = {r: float(np.mean(scores[r])) for r in REGION_NAMES}
    metrics["mean"] = float(np.mean(list(metrics.values())))
    return metrics


# ---------------------------------------------------------------------------
# Visualisation — original vs reconstructed mask
# ---------------------------------------------------------------------------

def _labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label, colour in LABEL_COLOURS.items():
        rgb[label_map == label] = colour
    return rgb


def _best_tumour_slice(gt: np.ndarray) -> int:
    # gt is (D, H, W); we display gt[:, :, z], so pick W-index with most tumor
    tumour = (gt != 0)
    counts = tumour.sum(axis=(0, 1))  # sum over D,H → (W,)
    return int(counts.argmax()) if counts.max() > 0 else gt.shape[-1] // 2


def _to_seg(arr: np.ndarray, subregion_mode: bool) -> np.ndarray:
    """Convert VAE output channels to a (D,H,W) label map for visualisation."""
    if subregion_mode:
        # arr: (4,D,H,W) [BG,NCR,ED,ET] — direct label assignment
        seg = np.zeros(arr.shape[1:], dtype=np.uint8)
        seg[arr[2] > 0.5] = 2   # ED
        seg[arr[1] > 0.5] = 1   # NCR overwrites ED at boundaries
        seg[arr[3] > 0.5] = 4   # ET overwrites NCR
        return seg
    else:
        # arr: (3,D,H,W) [WT,TC,ET]
        return regions_to_seg(arr[0], arr[1], arr[2])


@torch.no_grad()
def save_recon_visualisation(
    vae: MaskVAE,
    val_cases: list[torch.Tensor],
    device: torch.device,
    save_path: Path,
    step: int,
    subregion_mode: bool = False,
) -> None:
    vae.eval()
    n = len(val_cases)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    fig.suptitle(f"MaskVAE reconstruction — step {step}\n"
                 f"black=BG  blue=NCR  green=ED  red=ET", fontsize=9)

    for row, gt_tensor in enumerate(val_cases):
        gt_dev  = gt_tensor.to(device)
        recon_logits, _, _ = vae(gt_dev)
        recon_bin = recon_logits.sigmoid().cpu().numpy()[0]   # (C,D,H,W)
        gt_np     = gt_dev[0].cpu().numpy()

        gt_seg    = _to_seg(gt_np,    subregion_mode)
        recon_seg = _to_seg(recon_bin, subregion_mode)

        z = _best_tumour_slice(gt_seg)

        axes[row][0].imshow(_labels_to_rgb(gt_seg[:, :, z]),    interpolation="nearest")
        axes[row][0].set_title(f"GT (case {row})",   fontsize=8)
        axes[row][0].axis("off")

        axes[row][1].imshow(_labels_to_rgb(recon_seg[:, :, z]), interpolation="nearest")
        axes[row][1].set_title(f"Recon (case {row})", fontsize=8)
        axes[row][1].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device     = torch.device(args.device)
    splits_dir = args.splits_dir or args.data_root

    num_classes = 4 if args.subregion_mode else 3
    channels    = tuple(int(c) for c in args.mask_vae_channels.split(","))
    pos_weight  = SUBREGION_POS_WEIGHT if args.subregion_mode else None

    vae = MaskVAE(
        num_classes=num_classes,
        latent_channels=args.latent_channels,
        channels=channels,
        num_res_units=args.num_res_units,
    ).to(device)

    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

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
        subregion_based=args.subregion_mode,
        region_based=not args.subregion_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_cases: list[torch.Tensor] = []
    val_path = os.path.join(splits_dir, args.val_split_file)
    if os.path.exists(val_path):
        val_ds = BraTSDataset(
            root=args.data_root,
            split_file=val_path,
            crop_size=args.crop_size,
            subregion_based=args.subregion_mode,
            region_based=not args.subregion_mode,
            random_crop=False,
        )
        for i in range(min(args.num_val_cases, len(val_ds))):
            _, mask = val_ds[i]
            val_cases.append(mask.unsqueeze(0))
    else:
        print(f"  [warn] val split not found at {val_path}, skipping val metrics")

    run_id        = new_run_id(STAGE)
    logger        = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = checkpoint_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)

    steps_per_epoch = len(loader)
    start_epoch     = start_step // max(1, steps_per_epoch)
    total_steps     = args.num_epochs * steps_per_epoch

    spatial = args.crop_size // 8
    print(f"Training cases  : {len(dataset)}")
    print(f"Mask channels   : {num_classes}  ({'subregion [BG,NCR,ED,ET]' if args.subregion_mode else 'region [WT,TC,ET]'})")
    print(f"Latent          : {args.latent_channels}ch × {spatial}³")
    print(f"Total steps     : {total_steps}")

    best_mean_dice = 0.0
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        warmup_steps=args.kl_warmup_steps,
        mode="max",
    ) if args.early_stop_patience > 0 else None

    vae.train()
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step,
                desc=f"[{STAGE}]", unit="step")

    for epoch in range(start_epoch, args.num_epochs):
        tqdm.write(f"\n[{STAGE}] Epoch {epoch + 1}/{args.num_epochs}")

        for _, mask in loader:
            mask = mask.to(device)                    # (B, 3, 96, 96, 96)

            # KL annealing: beta ramps 0 → vae_beta over kl_warmup_steps
            if args.kl_warmup_steps > 0:
                beta = args.vae_beta * min(1.0, step / args.kl_warmup_steps)
            else:
                beta = args.vae_beta

            recon_logits, mu, logvar = vae(mask)
            total, r_loss, k_loss = mask_vae_loss(
                recon_logits, mask, mu, logvar, beta=beta, pos_weight=pos_weight
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(
                epoch=f"{epoch + 1}/{args.num_epochs}",
                loss=f"{total.item():.4f}",
                recon=f"{r_loss.item():.4f}",
                kl=f"{k_loss.item():.6f}",
                beta=f"{beta:.2e}",
            )

            if step % args.log_every == 0:
                logger.log_metrics(
                    step=step, csv="train",
                    epoch=epoch + 1,
                    total_loss=total.item(),
                    recon_loss=r_loss.item(),
                    kl_loss=k_loss.item(),
                    beta=beta,
                )
                tqdm.write(
                    f"[{STAGE}] epoch={epoch+1}  step={step}  "
                    f"total={total.item():.4f}  recon={r_loss.item():.4f}  "
                    f"kl={k_loss.item():.6f}  beta={beta:.2e}"
                )

            if step % args.val_every == 0 and step > start_step and val_cases:
                metrics = run_val_metrics(vae, val_cases, device, subregion_mode=args.subregion_mode)
                logger.log_metrics(step=step, csv="val", **metrics)

                is_best = metrics["mean"] > best_mean_dice
                if is_best:
                    best_mean_dice = metrics["mean"]
                    torch.save({
                        "vae_state_dict":       vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step":   step,
                        "epoch":  epoch,
                        "config": vars(args),
                        "best_mean_dice": best_mean_dice,
                    }, checkpoint_dir / "best.pth")

                es_status = ""
                if early_stopper is not None:
                    should_stop = early_stopper.step(metrics["mean"], step)
                    es_status = f"  [{early_stopper.status}]"
                    if should_stop:
                        tqdm.write(f"  [early stop] no improvement for {early_stopper.patience} checks — stopping.")
                        pbar.close()
                        return

                tqdm.write(
                    f"  [val]  Dice — "
                    + "  ".join(f"{r}={metrics[r]:.3f}" for r in REGION_NAMES)
                    + f"  mean={metrics['mean']:.3f}"
                    + ("  ← best" if is_best else "")
                    + es_status
                )
                save_recon_visualisation(
                    vae, val_cases, device,
                    vis_dir / f"recon_step_{step:06d}.png",
                    step,
                    subregion_mode=args.subregion_mode,
                )
                vae.train()

            if step % args.ckpt_every == 0 and step > start_step:
                ckpt_path = checkpoint_dir / f"step_{step}.pth"
                torch.save({
                    "vae_state_dict":       vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step":   step,
                    "epoch":  epoch,
                    "config": vars(args),
                }, ckpt_path)

            step += 1
            pbar.update(1)

    pbar.close()
    final_path = checkpoint_dir / "final.pth"
    torch.save({
        "vae_state_dict":       vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step":   step,
        "epoch":  args.num_epochs,
        "config": vars(args),
    }, final_path)
    print(f"\nDone. Checkpoint → {final_path}")
    print(f"Pass to train_pixel_test.py with:  --mask_vae_checkpoint {final_path}")

    logger.append_to_experiments_index(
        f"Mask VAE, {args.num_epochs} epochs ({step} steps), "
        f"latent={args.latent_channels}ch, channels={args.mask_vae_channels}, "
        f"beta={args.vae_beta}, final_recon={r_loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
