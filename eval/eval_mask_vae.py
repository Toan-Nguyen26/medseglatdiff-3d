"""
Evaluate a trained MaskVAE on the test split.

Produces two outputs:
  1. Dice metrics (WT/TC/ET) for reconstruction and random generation
  2. A PNG grid showing GT, reconstruction, and sampled generations side-by-side

Usage:
    python3 -m eval.eval_mask_vae \\
        --checkpoint checkpoints/mask_vae_xxx/best.pth \\
        --output_dir eval_out/mask_vae

Optional flags:
    --n_samples   2    # how many random generations to show per case (default 2)
    --n_cases     8    # how many test cases to include in the grid (default 8)
    --split       test # which split file to use (default: test)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.brats_dataset import BraTSDataset, regions_to_seg
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


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_vae(ckpt_path: str, device: torch.device) -> tuple[MaskVAE, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    channels = tuple(int(c) for c in cfg["mask_vae_channels"].split(","))
    vae = MaskVAE(
        num_classes=3,
        latent_channels=cfg["latent_channels"],
        channels=channels,
        num_res_units=cfg["num_res_units"],
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    return vae, cfg


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
) -> None:
    n_cases = min(n_cases, len(dataset))
    # cols: GT | Recon | Sample_1 | Sample_2 | ...
    n_cols = 2 + n_samples
    col_labels = ["GT", "Recon"] + [f"Sample {i+1}" for i in range(n_samples)]

    fig, axes = plt.subplots(
        n_cases, n_cols,
        figsize=(n_cols * 2.5, n_cases * 2.5),
        squeeze=False,
    )
    fig.suptitle(
        "MaskVAE — GT / Reconstruction / Random Samples\n"
        "black=BG  blue=NCR  green=ED  red=ET",
        fontsize=10,
    )

    recon_scores: dict[str, list[float]] = {r: [] for r in REGION_NAMES}

    for row in range(n_cases):
        _, mask = dataset[row]         # (3, D, H, W) float
        x = mask.unsqueeze(0).to(device)

        # --- Encode → decode (reconstruction) ---
        logits_recon, mu, logvar = vae(x)
        recon_bin = (logits_recon.sigmoid() > 0.5).cpu().numpy()[0]  # (3,D,H,W)
        gt_np     = mask.numpy()                                      # (3,D,H,W)

        # --- Dice for reconstruction ---
        for j, r in enumerate(REGION_NAMES):
            recon_scores[r].append(
                dice(recon_bin[j].astype(bool), gt_np[j].astype(bool))
            )

        # --- Convert to colour seg maps ---
        gt_seg    = regions_to_seg(gt_np[0],    gt_np[1],    gt_np[2])
        recon_seg = regions_to_seg(recon_bin[0], recon_bin[1], recon_bin[2])

        z_slice = best_tumour_slice(gt_seg)

        def _show(ax, seg, title):
            ax.imshow(seg_to_rgb(seg[:, :, z_slice]), interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        _show(axes[row][0], gt_seg,    f"GT (case {row})")
        _show(axes[row][1], recon_seg, f"Recon\nWT={recon_scores['WT'][-1]:.2f} "
                                       f"TC={recon_scores['TC'][-1]:.2f} "
                                       f"ET={recon_scores['ET'][-1]:.2f}")

        # --- Random generations: sample z ~ N(0,1) → decode ---
        for s in range(n_samples):
            z_sample = torch.randn(1, *latent_shape, device=device)
            logits_gen = vae.decode(z_sample)
            gen_bin    = (logits_gen.sigmoid() > 0.5).cpu().numpy()[0]
            gen_seg    = regions_to_seg(gen_bin[0], gen_bin[1], gen_bin[2])
            _show(axes[row][2 + s], gen_seg, f"Sample {s+1}")

    plt.tight_layout()
    out_path = output_dir / "test_eval_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved → {out_path}")

    # --- Print aggregate metrics ---
    print(f"\n{'='*50}")
    print(f"Reconstruction Dice (n={n_cases} test cases)")
    print(f"{'='*50}")
    for r in REGION_NAMES:
        vals = recon_scores[r]
        print(f"  {r}: mean={np.mean(vals):.4f}  "
              f"std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  "
              f"max={np.max(vals):.4f}")
    mean_all = np.mean([np.mean(v) for v in recon_scores.values()])
    print(f"  mean: {mean_all:.4f}")

    # --- Per-case table ---
    print(f"\n{'Case':<30} {'WT':>6} {'TC':>6} {'ET':>6}")
    print("-" * 48)
    for i in range(n_cases):
        name = dataset.names[i]
        wt = recon_scores["WT"][i]
        tc = recon_scores["TC"][i]
        et = recon_scores["ET"][i]
        print(f"  {name:<28} {wt:>6.3f} {tc:>6.3f} {et:>6.3f}")


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="eval_out/mask_vae")
    p.add_argument("--split",      default="test")
    p.add_argument("--n_cases",    type=int, default=8)
    p.add_argument("--n_samples",  type=int, default=2,
                   help="Random samples from N(0,1) prior to show per case")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    vae, cfg = load_vae(args.checkpoint, device)

    splits_dir = cfg.get("splits_dir") or cfg["data_root"]
    split_file = Path(splits_dir) / f"{args.split}.txt"

    dataset = BraTSDataset(
        root=cfg["data_root"],
        split_file=split_file,
        crop_size=cfg["crop_size"],
        region_based=True,
        random_crop=False,
    )
    print(f"Test cases : {len(dataset)}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Step       : {torch.load(args.checkpoint, map_location='cpu', weights_only=True)['step']}")
    print(f"Best Dice  : {torch.load(args.checkpoint, map_location='cpu', weights_only=True).get('best_mean_dice', 'N/A')}")

    latent_ch = cfg["latent_channels"]
    spatial   = cfg["crop_size"] // 8          # 8x downsampling → 16 for 128³
    latent_shape = (latent_ch, spatial, spatial, spatial)
    print(f"Latent     : {latent_shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate(vae, dataset, args.n_cases, args.n_samples, device, output_dir, latent_shape)


if __name__ == "__main__":
    main()
