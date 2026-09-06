"""
Dice against the number of sampling seeds, to justify the choice of N.

Runs inference on the full-modality configuration at several values of N and
records the ensemble Dice for each. The resulting curve shows where averaging
more samples stops paying for itself, which is the evidence for fixing N in
the main experiments.

Only one modality configuration is swept, so this is far cheaper than the
15-combo evaluation: roughly sum(N_values) x cases x DDIM steps forward passes.

Usage:
    python3 scripts/sweep_n_samples.py \\
        --diffusion_ckpt checkpoints/latent_diffusion_.../best.pth \\
        --data_root      data/brats_roi128 \\
        --splits_dir     splits/brats_roi128_full \\
        --n_values       1,2,3,4,5,6,8,10 \\
        --device         cuda

Writes:
    <output_dir>/n_sweep.csv          one row per N
    <output_dir>/n_sweep.pdf          the figure for the paper
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--diffusion_ckpt", required=True)
    p.add_argument("--data_root",      required=True)
    p.add_argument("--splits_dir",     required=True)
    p.add_argument("--image_vae_ckpt", default=None)
    p.add_argument("--mask_vae_ckpt",  default=None)
    p.add_argument("--n_values",   default="1,2,3,4,5,6,8,10",
                   help="Comma-separated values of N to evaluate.")
    p.add_argument("--num_cases",  type=int, default=None,
                   help="Test cases per N. Omit for the full test split.")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--output_dir", default="eval_output/n_sweep")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def run_one(args, n: int, work: Path) -> float | None:
    """Evaluate at a single N and return the mean WT Dice."""
    out = work / f"n{n}"
    cmd = [
        sys.executable, "-m", "eval.infer_latent",
        "--diffusion_ckpt",      args.diffusion_ckpt,
        "--data_root",           args.data_root,
        "--splits_dir",          args.splits_dir,
        "--output_dir",          str(out),
        "--n_samples",           str(n),
        "--num_inference_steps", str(args.num_inference_steps),
        "--modality_mask",       "all",       # full modality only
        "--regions",             "wt",
        "--num_vis_cases",       "0",         # no figures needed here
        "--device",              args.device,
    ]
    if args.image_vae_ckpt:
        cmd += ["--image_vae_ckpt", args.image_vae_ckpt]
    if args.mask_vae_ckpt:
        cmd += ["--mask_vae_ckpt", args.mask_vae_ckpt]
    if args.num_cases:
        cmd += ["--num_cases", str(args.num_cases)]

    print(f"\n=== N = {n} ===")
    if subprocess.run(cmd).returncode != 0:
        print(f"  [warn] N={n} failed, skipping")
        return None

    # single-combo runs write metrics_<combo>_n<N>.csv
    hits = list(out.glob("metrics_*.csv"))
    if not hits:
        print(f"  [warn] no metrics CSV for N={n}")
        return None
    rows = list(csv.DictReader(hits[0].open()))
    return float(np.mean([float(r["WT_dice"]) for r in rows]))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_values = [int(v) for v in args.n_values.split(",")]
    results: list[tuple[int, float]] = []

    for n in n_values:
        dice = run_one(args, n, out)
        if dice is not None:
            results.append((n, dice))
            print(f"  N={n}  WT Dice = {dice:.4f}")

    if not results:
        raise SystemExit("No runs completed.")

    csv_path = out / "n_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "wt_dice"])
        w.writerows(results)
    print(f"\n  CSV -> {csv_path}")

    plot(results, out / "n_sweep.pdf")


def plot(results: list[tuple[int, float]], path: Path) -> None:
    ns = [r[0] for r in results]
    ds = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ns, ds, marker="o", lw=1.6, color="#1f4e79", zorder=3)

    # Mark the chosen operating point: the smallest N within 0.002 Dice of the
    # best result. Reading the elbow off the curve by eye is not reproducible.
    best = max(ds)
    chosen = next(n for n, d in results if d >= best - 0.002)
    ci = ns.index(chosen)
    ax.plot([chosen], [ds[ci]], marker="o", ms=11, mfc="none",
            mec="#cc0000", mew=2, zorder=4)
    ax.annotate(f"N = {chosen}", xy=(chosen, ds[ci]),
                xytext=(6, -14), textcoords="offset points",
                fontsize=9, color="#cc0000")

    ax.set_xlabel("Number of sampling seeds $N$")
    ax.set_ylabel("Whole-tumour Dice")
    ax.set_xticks(ns)
    ax.grid(alpha=0.3, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Figure -> {path}")
    print(f"  Chosen N = {chosen} (within 0.002 Dice of the best, {best:.4f})")


if __name__ == "__main__":
    main()
