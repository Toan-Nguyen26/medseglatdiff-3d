"""
Inference script for pixel-space diffusion (train_diffusion.py checkpoints).

No VAE, no encoder — just load PixelUNet3D and run DDIM denoising with the
MRI volume concatenated to the noisy mask at every step.

Metrics reported (per region: WT / TC / ET):
  Dice, IoU      — ensemble prediction vs GT
  GED            — Generalized Energy Distance (sample diversity vs GT)
  D_max          — max pairwise distance between samples
  Uncertainty    — mean voxel variance across N samples

Run from repo root:
    python3 -m eval.infer_pixel \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_20pct \\
        --checkpoint checkpoints/diffusion_<run_id>/best.pth \\
        --n_samples 5

    # Fix a specific missing-modality scenario (T2 missing):
    python3 -m eval.infer_pixel ... --modality_mask 1101 --n_samples 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import (
    BraTSDataset, apply_modality_mask, sample_modality_mask,
)
from monai.networks.schedulers import DDIMScheduler
from models.diffusion.config import PixelDiffusionConfig
from models.diffusion.unet3d import PixelUNet3D

MODALITY_NAMES = ["FLAIR", "T1ce", "T1", "T2"]
REGION_NAMES   = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data_root",  required=True)
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--test_split_file", default="test.txt")
    p.add_argument("--num_cases",  type=int, default=None,
                   help="How many test cases to run. Default: all in test.txt.")
    p.add_argument("--crop_size",  type=int, default=96)

    p.add_argument("--checkpoint", required=True,
                   help="best.pth or final.pth from train_diffusion.py")

    p.add_argument("--n_samples",           type=int, default=5,
                   help="Noise samples per case (ensemble size).")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM steps per sample (50 ≈ 1000-step quality).")
    p.add_argument("--threshold",           type=float, default=0.5)
    p.add_argument("--modality_mask",       default="all",
                   help="'all' | 'random' | 4-char binary e.g. '1101' (T2 missing).")

    p.add_argument("--output_dir", default="eval_output")
    p.add_argument("--device",
                   default="mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> tuple[PixelUNet3D, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt.get("config", {})

    channel_mults = tuple(int(x) for x in cfg.get("channel_mults", "1,2,4").split(","))
    config = PixelDiffusionConfig(
        base_channels=cfg.get("base_channels", 32),
        channel_multipliers=channel_mults,
        num_timesteps=cfg.get("num_timesteps", 1000),
    )
    unet = PixelUNet3D(config).to(device)
    unet.load_state_dict(ckpt["unet_state_dict"])
    unet.eval()
    print(f"PixelUNet3D  ← {ckpt_path}")
    print(f"  base_ch={config.base_channels}  mults={channel_mults}  "
          f"T={config.num_timesteps}  "
          f"best_val_dice={ckpt.get('best_mean_dice', 'n/a')}")
    return unet, cfg


# ---------------------------------------------------------------------------
# Modality mask
# ---------------------------------------------------------------------------

def parse_modality_mask(spec: str, device: torch.device) -> torch.Tensor | None:
    if spec == "all":
        return None
    if spec == "random":
        return sample_modality_mask(1).to(device)
    if len(spec) == 4 and all(c in "01" for c in spec):
        return torch.tensor([[bool(int(c)) for c in spec]], dtype=torch.bool).to(device)
    raise ValueError(f"Invalid --modality_mask '{spec}'.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(tp / ((pred | gt).sum() + eps))


def seg_dist(s1: np.ndarray, s2: np.ndarray) -> float:
    return 1.0 - dice(s1, s2)


def ged(samples: np.ndarray, gt: np.ndarray) -> float:
    n = len(samples)
    cross = float(np.mean([seg_dist(samples[i], gt) for i in range(n)]))
    if n > 1:
        pairwise = float(np.mean([
            seg_dist(samples[i], samples[j])
            for i in range(n) for j in range(n) if i != j
        ]))
    else:
        pairwise = 0.0
    return 2 * cross - pairwise


def d_max(samples: np.ndarray) -> float:
    n = len(samples)
    if n <= 1:
        return 0.0
    return float(max(
        seg_dist(samples[i], samples[j])
        for i in range(n) for j in range(i + 1, n)
    ))


def compute_metrics(
    stack_np: np.ndarray,   # (N, 3, D, H, W) float [0,1]
    gt_np: np.ndarray,      # (3, D, H, W) float binary
    unc_np: np.ndarray,     # (3, D, H, W) float variance
    threshold: float = 0.5,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for r, rname in enumerate(REGION_NAMES):
        gt_bin      = gt_np[r] > threshold
        ens_bin     = stack_np[:, r].mean(axis=0) > threshold
        samples_bin = stack_np[:, r] > threshold

        results[f"{rname}_dice"]        = dice(ens_bin, gt_bin)
        results[f"{rname}_iou"]         = iou(ens_bin, gt_bin)
        results[f"{rname}_ged"]         = ged(samples_bin, gt_bin)
        results[f"{rname}_dmax"]        = d_max(samples_bin)
        results[f"{rname}_uncertainty"] = float(unc_np[r].mean())

    for metric in ("dice", "iou", "ged", "dmax", "uncertainty"):
        results[f"mean_{metric}"] = float(
            np.mean([results[f"{r}_{metric}"] for r in REGION_NAMES])
        )
    return results


# ---------------------------------------------------------------------------
# N-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_n_samples(
    unet: PixelUNet3D,
    vol_masked: torch.Tensor,   # (1, 4, D, H, W)
    n_samples: int,
    num_timesteps: int,
    num_inference_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N independent DDIM chains. Each starts from a different x_T seed.

    Returns:
        stack_np   : (N, 3, D, H, W) float32 in [0, 1]
        unc_np     : (3, D, H, W)  voxel-wise variance across samples
    """
    inf_scheduler = DDIMScheduler(num_train_timesteps=num_timesteps)
    inf_scheduler.set_timesteps(num_inference_steps)
    spatial = vol_masked.shape[2:]

    use_amp = (device.type == "cuda")
    samples = []
    for _ in range(n_samples):
        x = torch.randn(1, 3, *spatial, device=device)
        for t in inf_scheduler.timesteps:
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            x_in    = torch.cat([vol_masked, x], dim=1)   # (1, 7, D, H, W)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=use_amp):
                pred_noise = unet(x_in, t_batch)
            x = inf_scheduler.step(pred_noise.float(), t, x)[0].to(device).float()

        # x in [-1, +1] → [0, 1]
        probs = ((x[0] + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()  # (3, D, H, W)
        samples.append(probs)

    stack_np = np.stack(samples, axis=0)                   # (N, 3, D, H, W)
    unc_np   = stack_np.var(axis=0)                        # (3, D, H, W)
    return stack_np, unc_np


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _best_slice(gt: np.ndarray) -> int:
    counts = gt.sum(axis=(0, 1, 2))
    return int(counts.argmax()) if counts.max() > 0 else gt.shape[-1] // 2


def save_vis(
    vol_np: np.ndarray,      # (4, D, H, W)
    gt_np: np.ndarray,       # (3, D, H, W)
    stack_np: np.ndarray,    # (N, 3, D, H, W)
    unc_np: np.ndarray,      # (3, D, H, W)
    metrics: dict[str, float],
    mod_present: list[bool],
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    n      = stack_np.shape[0]
    best_z = _best_slice(gt_np)
    n_cols = 2 + n + 2   # T2 | GT | samples... | ensemble | uncertainty
    n_rows = 3            # WT / TC / ET

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 2.4),
                             squeeze=False)
    mod_str = "".join(f"{m}={'✓' if p else '✗'}" for m, p in zip(MODALITY_NAMES, mod_present))
    fig.suptitle(
        f"N={n}  |  {mod_str}\n"
        f"Dice WT={metrics['WT_dice']:.3f}  TC={metrics['TC_dice']:.3f}  ET={metrics['ET_dice']:.3f}  "
        f"GED WT={metrics['WT_ged']:.3f}  TC={metrics['TC_ged']:.3f}  ET={metrics['ET_ged']:.3f}",
        fontsize=8,
    )

    # Use FLAIR (ch 0) as background — most common reference in BraTS papers
    bg = vol_np[0, :, :, best_z]
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)

    for row, rname in enumerate(REGION_NAMES):
        col = 0
        axes[row][col].imshow(bg, cmap="gray")
        axes[row][col].set_ylabel(rname, fontsize=9)
        axes[row][col].set_title("FLAIR" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        axes[row][col].imshow(bg, cmap="gray", alpha=0.5)
        axes[row][col].imshow(gt_np[row, :, :, best_z] > threshold,
                              cmap="Blues", vmin=0, vmax=1, alpha=0.6)
        axes[row][col].set_title("GT" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        for i in range(n):
            s = stack_np[i, row, :, :, best_z] > threshold
            axes[row][col].imshow(bg, cmap="gray", alpha=0.4)
            axes[row][col].imshow(s, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
            axes[row][col].set_title(f"s{i+1}" if row == 0 else "", fontsize=7)
            axes[row][col].axis("off")
            col += 1

        ens = stack_np[:, row].mean(axis=0)[:, :, best_z] > threshold
        axes[row][col].imshow(bg, cmap="gray", alpha=0.4)
        axes[row][col].imshow(ens, cmap="Greens", vmin=0, vmax=1, alpha=0.6)
        d = metrics[f"{rname}_dice"]
        axes[row][col].set_title(f"ensemble\nDice={d:.3f}" if row == 0
                                 else f"Dice={d:.3f}", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        unc = unc_np[row, :, :, best_z]
        im  = axes[row][col].imshow(unc, cmap="hot", vmin=0, vmax=max(unc.max(), 1e-6))
        g   = metrics[f"{rname}_ged"]
        axes[row][col].set_title(f"uncertainty\nGED={g:.3f}" if row == 0
                                 else f"GED={g:.3f}", fontsize=7)
        axes[row][col].axis("off")
        plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Vis  → {save_path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    device     = torch.device(args.device)
    splits_dir = args.splits_dir or args.data_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unet, train_cfg = load_model(args.checkpoint, device)
    num_timesteps   = train_cfg.get("num_timesteps", 1000)

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.test_split_file),
        crop_size=args.crop_size,
        region_based=True,
        random_crop=False,
    )
    n_cases = min(args.num_cases, len(dataset)) if args.num_cases else len(dataset)
    print(f"\nEvaluating {n_cases} / {len(dataset)} test cases  "
          f"|  n_samples={args.n_samples}  "
          f"|  ddim_steps={args.num_inference_steps}\n")

    mod_tensor  = parse_modality_mask(args.modality_mask, device)
    mod_present = [True] * 4 if mod_tensor is None else mod_tensor[0].cpu().tolist()
    present_str = "".join("1" if p else "0" for p in mod_present)

    all_rows = []
    agg: dict[str, list] = {}

    for i in range(n_cases):
        vol, gt = dataset[i]
        vol = vol.unsqueeze(0).to(device)   # (1, 4, D, H, W)
        vol_in = apply_modality_mask(vol, mod_tensor) if mod_tensor is not None else vol

        stack_np, unc_np = infer_n_samples(
            unet, vol_in,
            n_samples=args.n_samples,
            num_timesteps=num_timesteps,
            num_inference_steps=args.num_inference_steps,
            device=device,
        )
        gt_np  = gt.numpy()          # (3, D, H, W)
        vol_np = vol[0].cpu().numpy()

        metrics = compute_metrics(stack_np, gt_np, unc_np, args.threshold)

        print(f"Case {i:03d}")
        print(f"  {'Region':<6} {'Dice':>6} {'IoU':>6} {'GED':>6} {'D_max':>6} {'Unc':>8}")
        for r in REGION_NAMES:
            print(f"  {r:<6} "
                  f"{metrics[f'{r}_dice']:>6.3f} "
                  f"{metrics[f'{r}_iou']:>6.3f} "
                  f"{metrics[f'{r}_ged']:>6.3f} "
                  f"{metrics[f'{r}_dmax']:>6.3f} "
                  f"{metrics[f'{r}_uncertainty']:>8.5f}")
        print(f"  {'mean':<6} "
              f"{metrics['mean_dice']:>6.3f} "
              f"{metrics['mean_iou']:>6.3f} "
              f"{metrics['mean_ged']:>6.3f} "
              f"{metrics['mean_dmax']:>6.3f} "
              f"{metrics['mean_uncertainty']:>8.5f}\n")

        for k, v in metrics.items():
            agg.setdefault(k, []).append(v)

        row = {"case": i, "modalities": present_str, "n_samples": args.n_samples}
        row.update(metrics)
        all_rows.append(row)

        vis_path = output_dir / f"case_{i:03d}_n{args.n_samples}_mod{present_str}.png"
        save_vis(vol_np, gt_np, stack_np, unc_np, metrics, mod_present, vis_path, args.threshold)

    print("=" * 55)
    print(f"SUMMARY  ({n_cases} cases, N={args.n_samples}, mod={present_str})")
    print(f"  {'Region':<6} {'Dice':>6} {'IoU':>6} {'GED':>6} {'D_max':>6} {'Unc':>8}")
    for r in REGION_NAMES:
        print(f"  {r:<6} "
              f"{np.mean(agg[f'{r}_dice']):>6.3f} "
              f"{np.mean(agg[f'{r}_iou']):>6.3f} "
              f"{np.mean(agg[f'{r}_ged']):>6.3f} "
              f"{np.mean(agg[f'{r}_dmax']):>6.3f} "
              f"{np.mean(agg[f'{r}_uncertainty']):>8.5f}")
    print("=" * 55)

    save_csv(all_rows, output_dir / f"metrics_n{args.n_samples}_mod{present_str}.csv")
    print(f"\nDone. Output → {output_dir}/")


if __name__ == "__main__":
    main()
