"""
Evaluate a trained MaskVAE on the test split.

Produces three outputs:
  1. Dice / IoU / SSIM / PSNR per region (WT/TC/ET) for reconstruction
  2. A per-case CSV of the same, so the numbers survive a Kaggle session
  3. A PNG grid showing GT, reconstruction, and sampled generations side-by-side

Metrics are computed over the whole split by default; --n_cases only controls
how many rows the PNG grid has. Scoring just the eight cases that happen to be
drawn would give a table with an eight-case sample size.

Usage:
    python3 -m eval.eval_mask_vae \\
        --checkpoint checkpoints/mask_vae_xxx/best.pth \\
        --output_dir eval_out/mask_vae

Optional flags:
    --n_samples   2    # how many random generations to show per case (default 2)
    --n_cases     8    # how many test cases to draw in the grid (default 8)
    --max_cases   N    # cap the cases scored (default: the whole split)
    --split       test # which split file to use (default: test)
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.brats_dataset import (
    BraTSDataset,
    regions_to_seg,
    subregions_to_regions,
)
from models.multiencoder.encoders import MaskVAE


# ---------------------------------------------------------------------------
# Colour map: 0=BG 1=NCR(blue) 2=ED(green) 4=ET(red)
# ---------------------------------------------------------------------------
LABEL_COLOURS = {0: (0, 0, 0), 1: (0, 0, 200), 2: (0, 200, 0), 4: (200, 0, 0)}
REGION_NAMES  = ["WT", "TC", "ET"]


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for lbl, col in LABEL_COLOURS.items():
        rgb[seg == lbl] = col
    return rgb


def best_tumour_slice(seg: np.ndarray) -> int:
    """seg: (D, H, W) — return W-index with most tumour."""
    tumour = (seg != 0)
    counts = tumour.sum(axis=(0, 1))  # sum over D,H → (W,)
    return int(counts.argmax()) if counts.max() > 0 else seg.shape[-1] // 2


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / (union + eps))


# SSIM and PSNR are computed on the sigmoid probability against the binary
# annotation, i.e. the mask is treated as a [0,1] image. Both sides already
# live on exactly that range, so -- unlike the image autoencoder eval -- there
# is no min-max rescaling here and the PSNR peak of 1.0 is the true one.
# Duplicated from eval_image_vae rather than shared, to keep each script a
# single self-contained file that runs on Kaggle without a package import.

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim_approx(pred: torch.Tensor, target: torch.Tensor, win: int = 7) -> float:
    """Box-filter SSIM over a 3D volume (uniform window, not Gaussian)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(1, 1, win, win, win, device=pred.device) / (win ** 3)
    def conv(x): return F.conv3d(x, k, padding=win // 2)
    mx, my = conv(pred), conv(target)
    sx  = conv(pred * pred)   - mx * mx
    sy  = conv(target * target) - my * my
    sxy = conv(pred * target) - mx * my
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx ** 2 + my ** 2 + C1) * (sx + sy + C2)
    return float((num / den).mean().item())


METRIC_NAMES = ["dice", "iou", "ssim", "psnr"]


def soft_regions(prob: torch.Tensor) -> torch.Tensor:
    """
    Soft [WT, TC, ET] from [BG, NCR, ED, ET] sigmoid probabilities.

    subregions_to_regions() thresholds, which is right for Dice and IoU but
    destroys the continuous values SSIM and PSNR need. These definitions keep
    them, and on a binary ground truth -- where the four channels are one-hot
    -- they reduce exactly to the hard masks, so the target is unchanged.
    """
    bg, ncr, et = prob[:, 0:1], prob[:, 1:2], prob[:, 3:4]
    wt = 1.0 - bg
    tc = torch.clamp(ncr + et, 0.0, 1.0)
    return torch.cat([wt, tc, et], dim=1)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_vae(ckpt_path: str, device: torch.device) -> tuple[MaskVAE, dict, bool]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    channels = tuple(int(c) for c in cfg["mask_vae_channels"].split(","))
    # Width comes from the run that produced the checkpoint, not a constant:
    # train_mask_vae uses 4 mutually exclusive subregions under --subregion_mode
    # and 3 overlapping regions otherwise. Hardcoding 3 made a subregion
    # checkpoint fail to load at all.
    subregion = bool(cfg.get("subregion_mode", False))
    vae = MaskVAE(
        num_classes=4 if subregion else 3,
        latent_channels=cfg["latent_channels"],
        channels=channels,
        num_res_units=cfg["num_res_units"],
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    return vae, cfg, subregion


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    vae: MaskVAE,
    dataset: BraTSDataset,
    n_cases: int,
    n_samples: int,
    device: torch.device,
    output_dir: Path,
    latent_shape: tuple,
    subregion: bool = False,
    max_cases: int | None = None,
) -> None:
    # Every case in the split is scored; only the first n_grid are drawn.
    n_scored = len(dataset) if max_cases is None else min(max_cases, len(dataset))
    n_grid   = min(n_cases, n_scored)

    # cols: GT | Recon | Sample_1 | Sample_2 | ...
    n_cols = 2 + n_samples

    fig, axes = plt.subplots(
        n_grid, n_cols,
        figsize=(n_cols * 2.5, n_grid * 2.5),
        squeeze=False,
    )
    fig.suptitle(
        "MaskVAE — GT / Reconstruction / Random Samples\n"
        "black=BG  blue=NCR  green=ED  red=ET",
        fontsize=10,
    )

    # region -> metric -> list over cases
    scores: dict[str, dict[str, list[float]]] = {
        r: {m: [] for m in METRIC_NAMES} for r in REGION_NAMES
    }
    case_names: list[str] = []

    for row in range(n_scored):
        _, mask = dataset[row]         # (3, D, H, W) float
        x = mask.unsqueeze(0).to(device)

        # --- Encode → decode (reconstruction) ---
        logits_recon, mu, logvar = vae(x)
        prob = logits_recon.sigmoid().float()             # (1,C,D,H,W)
        gt_t = mask.unsqueeze(0).to(device).float()       # (1,C,D,H,W)

        if subregion:
            # Channels here are [BG, NCR, ED, ET], so channel 0 is background,
            # not WT. Binary metrics use the same hard conversion that
            # infer_latent applies to the diffusion samples, so this table is
            # the ceiling for that one rather than a slightly different number.
            recon_bin = subregions_to_regions(prob[0].cpu().numpy()) > 0.5
            gt_np     = subregions_to_regions(mask.numpy()) > 0.5
            prob_r    = soft_regions(prob)                # (1,3,D,H,W)
            gt_r      = soft_regions(gt_t)
        else:
            recon_bin = (prob > 0.5).cpu().numpy()[0]     # (3,D,H,W)
            gt_np     = mask.numpy() > 0.5
            prob_r, gt_r = prob, gt_t

        # --- Metrics for reconstruction, always in [WT, TC, ET] space ---
        for j, r in enumerate(REGION_NAMES):
            scores[r]["dice"].append(dice(recon_bin[j], gt_np[j]))
            scores[r]["iou"].append(iou(recon_bin[j], gt_np[j]))

            p_t = prob_r[:, j:j + 1]
            g_t = gt_r[:, j:j + 1]
            scores[r]["ssim"].append(ssim_approx(p_t, g_t))
            scores[r]["psnr"].append(psnr(p_t, g_t))

        case_names.append(dataset.names[row])

        if row >= n_grid:
            continue

        # --- Convert to colour seg maps ---
        gt_seg    = regions_to_seg(gt_np[0],    gt_np[1],    gt_np[2])
        recon_seg = regions_to_seg(recon_bin[0], recon_bin[1], recon_bin[2])

        z_slice = best_tumour_slice(gt_seg)

        def _show(ax, seg, title):
            ax.imshow(seg_to_rgb(seg[:, :, z_slice]), interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        _show(axes[row][0], gt_seg,    f"GT (case {row})")
        _show(axes[row][1], recon_seg,
              f"Recon\nWT={scores['WT']['dice'][-1]:.2f} "
              f"TC={scores['TC']['dice'][-1]:.2f} "
              f"ET={scores['ET']['dice'][-1]:.2f}")

        # --- Random generations: sample z ~ N(0,1) → decode ---
        for s in range(n_samples):
            z_sample = torch.randn(1, *latent_shape, device=device)
            logits_gen = vae.decode(z_sample)
            gen_prob   = logits_gen.sigmoid().float()[0].cpu().numpy()
            gen_bin    = (subregions_to_regions(gen_prob) > 0.5) if subregion \
                         else (gen_prob > 0.5)
            gen_seg    = regions_to_seg(gen_bin[0], gen_bin[1], gen_bin[2])
            _show(axes[row][2 + s], gen_seg, f"Sample {s+1}")

    plt.tight_layout()
    out_path = output_dir / "test_eval_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved → {out_path} ({n_grid} of {n_scored} scored cases)")

    # --- Per-case CSV, so the numbers outlive the Kaggle session ---
    csv_path = output_dir / "mask_vae_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + [f"{r}_{m}" for r in REGION_NAMES
                                          for m in METRIC_NAMES])
        for i, name in enumerate(case_names):
            w.writerow([name] + [f"{scores[r][m][i]:.6f}" for r in REGION_NAMES
                                                          for m in METRIC_NAMES])
    print(f"  CSV  saved → {csv_path}")

    # --- Print aggregate metrics ---
    print(f"\n{'='*62}")
    print(f"MaskVAE reconstruction (n={n_scored} cases)")
    print(f"{'='*62}")
    print(f"{'Region':<8} {'Dice':>9} {'IoU':>9} {'SSIM':>9} {'PSNR(dB)':>10}")
    print("-" * 62)
    for r in REGION_NAMES:
        vals = [np.mean(scores[r][m]) for m in METRIC_NAMES]
        print(f"{r:<8} {vals[0]:>9.4f} {vals[1]:>9.4f} "
              f"{vals[2]:>9.4f} {vals[3]:>10.2f}")
    means = [np.mean([np.mean(scores[r][m]) for r in REGION_NAMES])
             for m in METRIC_NAMES]
    print("-" * 62)
    print(f"{'mean':<8} {means[0]:>9.4f} {means[1]:>9.4f} "
          f"{means[2]:>9.4f} {means[3]:>10.2f}")

    # Spread on Dice only — the column the paper quotes as the ceiling.
    print(f"\nDice spread")
    for r in REGION_NAMES:
        v = scores[r]["dice"]
        print(f"  {r}: mean={np.mean(v):.4f}  std={np.std(v):.4f}  "
              f"min={np.min(v):.4f}  max={np.max(v):.4f}")


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="eval_out/mask_vae")
    p.add_argument("--split",      default="test")
    p.add_argument("--data_root",  default=None,
                   help="Override the data root recorded in the checkpoint. "
                        "Needed when evaluating somewhere other than the "
                        "machine that trained it.")
    p.add_argument("--splits_dir", default=None,
                   help="Override the splits directory recorded in the "
                        "checkpoint.")

    p.add_argument("--n_cases",    type=int, default=8,
                   help="Cases drawn in the PNG grid (metrics use the whole split)")
    p.add_argument("--max_cases",  type=int, default=None,
                   help="Cap the cases scored. Omit to score the whole split.")
    p.add_argument("--n_samples",  type=int, default=2,
                   help="Random samples from N(0,1) prior to show per case")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    vae, cfg, subregion = load_vae(args.checkpoint, device)

    # The checkpoint records the paths of the machine that trained it, which
    # do not exist anywhere else. Command-line values win when given.
    data_root  = args.data_root  or cfg["data_root"]
    splits_dir = args.splits_dir or cfg.get("splits_dir") or data_root
    split_file = Path(splits_dir) / f"{args.split}.txt"
    if not split_file.exists():
        raise SystemExit(
            f"Split file not found: {split_file}\n"
            f"  checkpoint recorded data_root={cfg.get('data_root')!r} "
            f"splits_dir={cfg.get('splits_dir')!r}\n"
            f"  pass --data_root / --splits_dir to point at this machine."
        )

    dataset = BraTSDataset(
        root=data_root,
        split_file=split_file,
        crop_size=cfg["crop_size"],
        # Must match the checkpoint: a subregion model outputs 4 channels, so
        # the target has to be the 4-channel one-hot, not the 3 regions.
        subregion_based=subregion,
        region_based=not subregion,
        random_crop=False,
    )
    print(f"Test cases : {len(dataset)}")
    print(f"Mode       : {'subregion (4ch)' if subregion else 'region (3ch)'}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Step       : {torch.load(args.checkpoint, map_location='cpu', weights_only=True)['step']}")
    print(f"Best Dice  : {torch.load(args.checkpoint, map_location='cpu', weights_only=True).get('best_mean_dice', 'N/A')}")

    latent_ch = cfg["latent_channels"]
    spatial   = cfg["crop_size"] // 8          # 8x downsampling → 16 for 128³
    latent_shape = (latent_ch, spatial, spatial, spatial)
    print(f"Latent     : {latent_shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate(vae, dataset, args.n_cases, args.n_samples, device, output_dir,
             latent_shape, subregion=subregion, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
