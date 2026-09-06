"""
Render the four MRI sequences for one case, on the slice the combo grid uses.

The combo-grid figure shows only FLAIR in its first column, so the paper has
no panel establishing what the model is actually given. This draws that panel
from the preprocessed ROI volume, which already holds all four sequences at
128^3 -- the same array the model sees, so the picture and the results are of
the same data.

Slice selection is imported from eval.infer_latent rather than reimplemented,
so this figure lands on exactly the same slice as the combo grid for the case.

Usage:
    python3 scripts/plot_modalities.py \\
        --case      BraTS-GLI-01023-000 \\
        --data_root data/brats_roi128_2023 \\
        --out       latex/figures/modalities.pdf

Writes the given path plus a .png alongside it for quick viewing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import seg_to_regions          # noqa: E402
from eval.infer_latent import best_slice               # noqa: E402

# Channel order is fixed by preprocess_brats.py: t2f, t1c, t1n, t2w.
MODALITY_NAMES = ["FLAIR", "T1ce", "T1", "T2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case",      required=True, help="e.g. BraTS-GLI-01023-000")
    p.add_argument("--data_root", default="data/brats_roi128_2023")
    p.add_argument("--out",       default="figures/modalities.pdf")
    p.add_argument("--slice",     type=int, default=None,
                   help="Override the slice index. Default: same as combo grid.")
    p.add_argument("--contour",   action="store_true",
                   help="Overlay the whole-tumour boundary on each sequence.")
    p.add_argument("--opaque", action="store_true",
                   help="Keep the white figure background. The default is a "
                        "transparent margin with the image panels left opaque.")
    p.add_argument("--clip", type=float, default=1.0,
                   help="Display percentile clip. Intensities are z-scored, so "
                        "a few extreme voxels otherwise flatten the window.")
    return p.parse_args()


def window(img: np.ndarray, clip: float) -> np.ndarray:
    """Percentile-clipped [0,1] mapping, for display only."""
    lo, hi = np.percentile(img, clip), np.percentile(img, 100 - clip)
    if hi - lo < 1e-6:
        lo, hi = float(img.min()), float(img.max())
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)

    vol_path = root / "vol" / f"{args.case}_vol.npy"
    seg_path = root / "seg" / f"{args.case}_seg.npy"
    for p in (vol_path, seg_path):
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    vol = np.load(vol_path)                    # (H, W, D, 4)
    seg = np.load(seg_path)                    # (H, W, D)
    vol = np.transpose(vol, (3, 0, 1, 2))      # (4, H, W, D)
    gt  = seg_to_regions(seg)                  # (3, H, W, D)

    z = args.slice if args.slice is not None else best_slice(gt)
    print(f"{args.case}: vol {vol.shape}, slice z={z}")

    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.4, 2.7), squeeze=False)
    for c, name in enumerate(MODALITY_NAMES):
        ax = axes[0][c]
        ax.imshow(window(vol[c, :, :, z], args.clip),
                  cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if args.contour and gt[0, :, :, z].any():
            ax.contour(gt[0, :, :, z], levels=[0.5],
                       colors="#39ff14", linewidths=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # transparent=True clears the figure and axes patches only. The four
    # panels are drawn images and stay opaque; the margin around them and the
    # gaps between them become see-through.
    tr = not args.opaque
    fig.savefig(out, bbox_inches="tight", transparent=tr)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=tr)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"  -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
