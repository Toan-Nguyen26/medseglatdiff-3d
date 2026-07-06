"""
Full missing-modality evaluation — sweeps all 15 non-empty modality combinations.

For each combination × each test case:
  - Run N-sample inference
  - Compute Dice, IoU, GED, D_max per region (WT / TC / ET)

Outputs:
  <output_dir>/metrics_full.csv   — one row per (combo, case)
  <output_dir>/summary.csv        — one row per combo, mean across cases
  <output_dir>/summary_table.txt  — human-readable table (paste into paper)

Usage (quick test on M4):
    python3 -m eval.eval_missing_modality \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_20pct \\
        --seg_checkpoint checkpoints/pixel_test_.../final.pth \\
        --num_cases 4 \\
        --n_samples 3 \\
        --num_inference_steps 10 \\
        --device mps

Full run (H200):
    python3 -m eval.eval_missing_modality \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_full \\
        --seg_checkpoint checkpoints/pixel_test_.../final.pth \\
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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import (
    ALL_MODALITY_COMBINATIONS,
    BraTSDataset,
    apply_modality_mask,
)
from eval.infer import (
    compute_all_metrics,
    infer_n_samples,
    load_models,
)

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
REGION_NAMES   = ["WT", "TC", "ET"]
METRICS        = ["dice", "iou", "ged", "dmax", "uncertainty"]


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

    p.add_argument("--seg_checkpoint",       required=True)
    p.add_argument("--image_vae_checkpoint", default=None)

    p.add_argument("--n_samples",            type=int, default=5)
    p.add_argument("--num_inference_steps",  type=int, default=50)
    p.add_argument("--threshold",            type=float, default=0.5)

    p.add_argument("--output_dir", default="eval_output/missing_modality")
    p.add_argument("--device",
                   default="mps"  if torch.backends.mps.is_available()
                   else   "cuda"  if torch.cuda.is_available()
                   else   "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Combo helpers
# ---------------------------------------------------------------------------

def combo_to_str(combo: tuple[bool, ...]) -> str:
    """(True, False, True, True) → '1011'"""
    return "".join("1" if c else "0" for c in combo)


def combo_to_label(combo: tuple[bool, ...]) -> str:
    """(True, False, True, True) → 'T1+T2+FLAIR'"""
    return "+".join(m for m, c in zip(MODALITY_NAMES, combo) if c)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_and_save_summary(
    summary: list[dict],
    output_dir: Path,
) -> None:
    """Print a formatted summary table and save it to summary_table.txt."""

    col_w = 7
    header_cols = ["Combo", "Modalities"] + [
        f"{r[:2]}_{m[:3]}" for r in REGION_NAMES for m in ("dice", "ged")
    ] + ["mean_dice", "mean_ged"]

    # header
    header = f"{'Combo':<6}  {'Modalities':<22}" + "".join(
        f"  {c:>{col_w}}" for c in header_cols[2:]
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for row in summary:
        combo_s  = row["combo"]
        mods_s   = row["modalities_label"][:22]
        vals = []
        for r in REGION_NAMES:
            vals.append(f"{row[f'{r}_dice']:>{col_w}.3f}")
            vals.append(f"{row[f'{r}_ged']:>{col_w}.3f}")
        vals.append(f"{row['mean_dice']:>{col_w}.3f}")
        vals.append(f"{row['mean_ged']:>{col_w}.3f}")
        lines.append(f"{combo_s:<6}  {mods_s:<22}" + "".join(vals))
    lines.append(sep)

    # overall mean across all combos
    overall_dice = float(np.mean([r["mean_dice"] for r in summary]))
    overall_ged  = float(np.mean([r["mean_ged"]  for r in summary]))
    lines.append(f"{'ALL':<6}  {'(mean over 15 combos)':<22}"
                 + " " * (col_w * 6 + 12)
                 + f"  {overall_dice:>{col_w}.3f}  {overall_ged:>{col_w}.3f}")
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)

    table_path = output_dir / "summary_table.txt"
    table_path.write_text(table)
    print(f"\n  Table → {table_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    device     = torch.device(args.device)
    splits_dir = args.splits_dir or args.data_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, unet, args = load_models(args, device)

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.test_split_file),
        crop_size=args.crop_size,
        region_based=True,
        random_crop=False,
    )
    n_cases = len(dataset) if args.num_cases is None else min(args.num_cases, len(dataset))
    print(f"\nCases per combo : {n_cases}")
    print(f"Combos          : {len(ALL_MODALITY_COMBINATIONS)}")
    print(f"N samples       : {args.n_samples}")
    print(f"Inference steps : {args.num_inference_steps}")
    print(f"Total UNet fwd  : "
          f"{len(ALL_MODALITY_COMBINATIONS) * n_cases * args.n_samples * args.num_inference_steps}"
          f" (diffusion) or "
          f"{len(ALL_MODALITY_COMBINATIONS) * n_cases * args.n_samples}"
          f" (direct)\n")

    all_rows: list[dict] = []
    summary:  list[dict] = []

    combo_bar = tqdm(ALL_MODALITY_COMBINATIONS, desc="Combos", unit="combo")
    for combo in combo_bar:
        combo_str   = combo_to_str(combo)
        combo_label = combo_to_label(combo)
        combo_bar.set_description(f"[{combo_str}] {combo_label}")

        mod_tensor = torch.tensor([list(combo)], dtype=torch.bool).to(device)

        agg: dict[str, list[float]] = {}

        for i in tqdm(range(n_cases), desc="  cases", leave=False):
            vol, gt = dataset[i]
            vol       = vol.unsqueeze(0).to(device)
            vol_input = apply_modality_mask(vol, mod_tensor)

            ensemble, uncertainty, stack = infer_n_samples(
                encoder, unet, vol_input,
                n_samples=args.n_samples,
                num_classes=args.num_classes,
                num_inference_steps=args.num_inference_steps,
                num_train_timesteps=args.num_timesteps,
                use_diffusion=args.use_diffusion,
                device=device,
            )

            gt_np    = gt.numpy()
            stack_np = stack[:, 0].cpu().numpy()
            unc_np   = uncertainty[0].cpu().numpy()

            metrics = compute_all_metrics(stack_np, gt_np, unc_np, args.threshold)

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

        # per-combo mean across cases
        combo_summary = {
            "combo":            combo_str,
            "modalities_label": combo_label,
            "n_modalities":     sum(combo),
        }
        for k, vals in agg.items():
            combo_summary[k] = float(np.mean(vals))
        summary.append(combo_summary)

    # ---- Save full CSV ----
    full_csv = output_dir / "metrics_full.csv"
    with full_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Full CSV → {full_csv}")

    # ---- Save summary CSV ----
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Summary  → {summary_csv}")

    # ---- Print + save table ----
    print_and_save_summary(summary, output_dir)


if __name__ == "__main__":
    main()
