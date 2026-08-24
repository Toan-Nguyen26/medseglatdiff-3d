"""
Merge the output of a sharded multi-GPU eval run.

`eval.infer_latent --num_shards N --shard_index i` splits the case list
across N processes, each writing its own output directory. This stitches
those back into one result: concatenated per-case metrics, per-combo means
recomputed over all cases, and the summary table redrawn.

Combo grid PNGs are per-case and need no merging — they are collected into
the merged directory as they are.

Usage:
    python3 scripts/merge_shards.py \\
        --shard_dirs eval_output/shard0 eval_output/shard1 \\
        --output_dir eval_output/merged
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.infer_latent import MODALITY_NAMES, print_summary_table


def _combo_label(bits: str) -> str:
    """'1101' -> 'FLAIR+T1ce+T2'. The per-case CSV rows carry only the
    bitstring, so the readable label is rebuilt here."""
    return "+".join(m for m, b in zip(MODALITY_NAMES, bits) if b == "1") or "none"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--shard_dirs", nargs="+", required=True,
                   help="Output directories of the individual shards.")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Concatenate per-case rows ────────────────────────────────────────────
    rows: list[dict] = []
    for d in args.shard_dirs:
        csv_path = Path(d) / "metrics_full.csv"
        if not csv_path.exists():
            print(f"  [warn] no metrics_full.csv in {d} — skipping")
            continue
        with csv_path.open() as fh:
            shard_rows = list(csv.DictReader(fh))
        rows.extend(shard_rows)
        print(f"  {csv_path}  ({len(shard_rows)} rows)")

    if not rows:
        raise SystemExit("No shard metrics found — nothing to merge.")

    # A case appearing twice means the shards overlapped, which would bias
    # every mean below. Stride-sharding cannot do that, but a hand-built run
    # can, so check rather than assume.
    seen = {(r["case"], r["combo"]) for r in rows}
    if len(seen) != len(rows):
        raise SystemExit(
            f"Duplicate (case, combo) pairs: {len(rows)} rows but only "
            f"{len(seen)} unique. Shards overlap — check --shard_index values."
        )

    float_keys = [
        k for k, v in rows[0].items()
        if k not in ("case", "combo", "modalities", "n_samples", "n_mods")
        and _is_float(v)
    ]

    full_csv = out / "metrics_full.csv"
    with full_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Merged rows → {full_csv}  ({len(rows)} rows)")

    # ── Per-combo means over every case from every shard ─────────────────────
    by_combo: dict[str, list[dict]] = {}
    for r in rows:
        by_combo.setdefault(r["combo"], []).append(r)

    # Preserve the order the shards evaluated combos in.
    summary: list[dict] = []
    for combo in dict.fromkeys(r["combo"] for r in rows):
        crows = by_combo[combo]
        entry = {
            "combo":      combo,
            "modalities": crows[0].get("modalities") or _combo_label(combo),
            "n_mods":     combo.count("1"),
            "n_cases":    len(crows),
        }
        for k in float_keys:
            entry[k] = float(np.mean([float(r[k]) for r in crows]))
        summary.append(entry)

    summary_csv = out / "summary.csv"
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"  Summary    → {summary_csv}")

    n_cases = {e["n_cases"] for e in summary}
    if len(n_cases) > 1:
        print(f"  [warn] uneven case counts per combo: {sorted(n_cases)} — "
              f"a shard may have failed partway")

    # ── Collect combo grids ─────────────────────────────────────────────────
    grid_out = out / "combo_grids"
    copied = 0
    for d in args.shard_dirs:
        src = Path(d) / "combo_grids"
        if not src.is_dir():
            continue
        grid_out.mkdir(exist_ok=True)
        for png in src.glob("*.png"):
            shutil.copy2(png, grid_out / png.name)
            copied += 1
    if copied:
        print(f"  Combo grids→ {grid_out}  ({copied} files)")

    # ── Redraw the table ────────────────────────────────────────────────────
    print_summary_table(summary, out)


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
