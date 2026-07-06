"""
Full missing-modality evaluation for pixel-space diffusion — sweeps all 15
non-empty modality combinations automatically.

For each combination × each test case:
  - Run N-sample DDIM inference with PixelUNet3D
  - Compute Dice, IoU, GED, D_max per region (WT / TC / ET)

Outputs:
  <output_dir>/metrics_full.csv    — one row per (combo, case)
  <output_dir>/summary.csv         — one row per combo, mean across cases
  <output_dir>/summary_table.txt   — human-readable table (paste into paper)

Usage (quick test on MPS):
    python3 -m eval.eval_missing_modality_pixel \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_smoke \\
        --checkpoint checkpoints/diffusion_<run_id>/best.pth \\
        --num_cases 2 \\
        --n_samples 1 \\
        --num_inference_steps 10 \\
        --device mps

Full run (H200):
    python3 -m eval.eval_missing_modality_pixel \\
        --data_root data/brats_combined \\
        --splits_dir splits/brats_combined_full \\
        --checkpoint checkpoints/diffusion_<run_id>/best.pth \\
        --n_samples 5 \\
        --num_inference_steps 50 \\
        --device cuda
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import (
    ALL_MODALITY_COMBINATIONS,
    BraTSDataset,
    apply_modality_mask,
)
from eval.infer_pixel import (
    compute_metrics,
    infer_n_samples,
    load_model,
)

MODALITY_NAMES = ["FLAIR", "T1ce", "T1", "T2"]
REGION_NAMES   = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data_root",       required=True)
    p.add_argument("--splits_dir",      default=None)
    p.add_argument("--test_split_file", default="test.txt")
    p.add_argument("--num_cases",       type=int, default=None,
                   help="Test cases per combo. Omit to use the full test split.")
    p.add_argument("--crop_size",       type=int, default=96)

    p.add_argument("--checkpoint",          required=True,
                   help="best.pth or final.pth from train_diffusion.py")

    p.add_argument("--n_samples",           type=int, default=5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--threshold",           type=float, default=0.5)

    p.add_argument("--output_dir", default="eval_output/missing_modality_pixel")
    p.add_argument("--device",
                   default="mps"  if torch.backends.mps.is_available()
                   else   "cuda"  if torch.cuda.is_available()
                   else   "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def combo_to_str(combo: tuple[bool, ...]) -> str:
    return "".join("1" if c else "0" for c in combo)


def combo_to_label(combo: tuple[bool, ...]) -> str:
    return "+".join(m for m, c in zip(MODALITY_NAMES, combo) if c)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_and_save_summary(summary: list[dict], output_dir: Path) -> None:
    col_w = 7
    header = (
        f"{'Combo':<6}  {'Modalities':<24}"
        + "".join(f"  {r[:2]}_Dice{'':<1}" for r in REGION_NAMES)
        + "".join(f"  {r[:2]}_GED{'':<2}" for r in REGION_NAMES)
        + f"  {'mDice':>{col_w}}  {'mGED':>{col_w}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for row in summary:
        dice_vals = "".join(f"  {row[f'{r}_dice']:>{col_w}.3f}" for r in REGION_NAMES)
        ged_vals  = "".join(f"  {row[f'{r}_ged']:>{col_w}.3f}"  for r in REGION_NAMES)
        lines.append(
            f"{row['combo']:<6}  {row['modalities_label'][:24]:<24}"
            + dice_vals + ged_vals
            + f"  {row['mean_dice']:>{col_w}.3f}  {row['mean_ged']:>{col_w}.3f}"
        )

    lines.append(sep)
    overall_dice = float(np.mean([r["mean_dice"] for r in summary]))
    overall_ged  = float(np.mean([r["mean_ged"]  for r in summary]))
    lines.append(
        f"{'ALL':<6}  {'(mean over 15 combos)':<24}"
        + " " * (col_w * 6 + 12)
        + f"  {overall_dice:>{col_w}.3f}  {overall_ged:>{col_w}.3f}"
    )
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)
    path = output_dir / "summary_table.txt"
    path.write_text(table)
    print(f"\n  Table → {path}")


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
    n_cases = len(dataset) if args.num_cases is None else min(args.num_cases, len(dataset))

    total_fwd = len(ALL_MODALITY_COMBINATIONS) * n_cases * args.n_samples * args.num_inference_steps
    print(f"\nCases per combo : {n_cases}  |  Combos: {len(ALL_MODALITY_COMBINATIONS)}")
    print(f"N samples       : {args.n_samples}  |  DDIM steps: {args.num_inference_steps}")
    print(f"Total UNet fwds : {total_fwd:,}\n")

    all_rows: list[dict] = []
    summary:  list[dict] = []

    combo_bar = tqdm(ALL_MODALITY_COMBINATIONS, desc="combos", unit="combo")
    for combo in combo_bar:
        combo_str   = combo_to_str(combo)
        combo_label = combo_to_label(combo)
        combo_bar.set_description(f"[{combo_str}] {combo_label:<20}")

        mod_tensor = torch.tensor([list(combo)], dtype=torch.bool).to(device)
        agg: dict[str, list[float]] = {}

        for i in tqdm(range(n_cases), desc="  cases", leave=False):
            vol, gt = dataset[i]
            vol    = vol.unsqueeze(0).to(device)
            vol_in = apply_modality_mask(vol, mod_tensor)

            stack_np, unc_np = infer_n_samples(
                unet, vol_in,
                n_samples=args.n_samples,
                num_timesteps=num_timesteps,
                num_inference_steps=args.num_inference_steps,
                device=device,
            )
            gt_np   = gt.numpy()
            metrics = compute_metrics(stack_np, gt_np, unc_np, args.threshold)

            for k, v in metrics.items():
                agg.setdefault(k, []).append(v)

            row = {
                "combo":            combo_str,
                "modalities_label": combo_label,
                "n_modalities":     sum(combo),
                "case":             i,
                "n_samples":        args.n_samples,
            }
            row.update(metrics)
            all_rows.append(row)

        combo_summary = {
            "combo":            combo_str,
            "modalities_label": combo_label,
            "n_modalities":     sum(combo),
        }
        for k, vals in agg.items():
            combo_summary[k] = float(np.mean(vals))
        summary.append(combo_summary)

        tqdm.write(
            f"  [{combo_str}] {combo_label:<20}  "
            f"Dice WT={combo_summary['WT_dice']:.3f}  "
            f"TC={combo_summary['TC_dice']:.3f}  "
            f"ET={combo_summary['ET_dice']:.3f}  "
            f"mean={combo_summary['mean_dice']:.3f}"
        )

    # Save full CSV
    full_csv = output_dir / "metrics_full.csv"
    with full_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Full CSV  → {full_csv}")

    # Save summary CSV
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Summary   → {summary_csv}")

    print_and_save_summary(summary, output_dir)


if __name__ == "__main__":
    main()
