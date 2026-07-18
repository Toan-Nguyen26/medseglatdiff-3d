#!/bin/bash
# ============================================================
#  collect_results.sh
#
#  Run this after ALL training and evaluation is done.
#  Gathers the 3 model checkpoints, training logs, visualisations,
#  and evaluation results into a single zip to send back.
#
#  Usage:
#    bash collect_results.sh
#
#  Output:
#    results_<timestamp>.zip   (~200–600 MB depending on model sizes)
# ============================================================
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints}"
LOGS_DIR="logs"
EVAL_DIR="eval_output/missing_modality"
SPLITS_DIR="${SPLITS_DIR:-splits/brats_roi128_full}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT="results_${TIMESTAMP}"
ZIP="${OUT}.zip"

banner() { echo ""; echo "────────────────────────────────────────"; echo "  $*"; echo "────────────────────────────────────────"; }
ok()     { echo "  ✓  $*"; }
warn()   { echo "  ⚠  $*"; }

banner "Collecting results → ${OUT}/"
mkdir -p "$OUT"

# ─── 1. Model checkpoints ───────────────────────────────────
banner "1 — Model checkpoints"
mkdir -p "$OUT/checkpoints"

_copy_best() {
    local glob="$1" label="$2"
    # find the most recent run dir matching the glob
    local dir
    dir=$(ls -td "${CKPT_DIR}/${glob}" 2>/dev/null | head -1)
    if [ -z "$dir" ]; then
        warn "No checkpoint found for ${label} (${glob})"
        return
    fi
    if [ -f "${dir}/best.pth" ]; then
        cp "${dir}/best.pth"  "$OUT/checkpoints/${label}_best.pth"
        ok "${label}_best.pth  ← ${dir}/best.pth"
    else
        warn "best.pth missing for ${label} — copying final.pth instead"
        [ -f "${dir}/final.pth" ] && cp "${dir}/final.pth" "$OUT/checkpoints/${label}_best.pth"
    fi
}

_copy_best "image_vae_*"       "image_vae"
_copy_best "mask_vae_*"        "mask_vae"
_copy_best "latent_diffusion_*" "latent_diffusion"

# ─── 2. Training logs ───────────────────────────────────────
banner "2 — Training logs (CSVs + notes)"
mkdir -p "$OUT/logs"

if [ -d "$LOGS_DIR/runs" ]; then
    for run_dir in "$LOGS_DIR"/runs/*/; do
        run_id=$(basename "$run_dir")
        dest="$OUT/logs/${run_id}"
        mkdir -p "$dest"
        # copy CSVs, config, notes — skip tensorboard event files (large)
        for f in "${run_dir}"*.csv "${run_dir}"*.yaml "${run_dir}"*.json "${run_dir}"notes.md; do
            [ -f "$f" ] && cp "$f" "$dest/" && ok "logs/${run_id}/$(basename "$f")"
        done
    done
    [ -f "$LOGS_DIR/EXPERIMENTS.md" ] && cp "$LOGS_DIR/EXPERIMENTS.md" "$OUT/logs/" && ok "EXPERIMENTS.md"
else
    warn "No logs/runs/ directory found"
fi

# ─── 3. Visualisations (training reconstructions) ───────────
banner "3 — Visualisations (sample reconstruction PNGs)"
mkdir -p "$OUT/visualisations"

for glob in "image_vae_*" "mask_vae_*" "latent_diffusion_*"; do
    dir=$(ls -td "${CKPT_DIR}/${glob}" 2>/dev/null | head -1)
    [ -z "$dir" ] && continue
    label=$(echo "$glob" | tr -d '*')
    if [ -d "${dir}/visualisations" ]; then
        # copy only last 5 vis PNGs (keeps zip size reasonable)
        mkdir -p "$OUT/visualisations/${label}"
        find "${dir}/visualisations" -name "*.png" | sort | tail -5 | while read -r f; do
            cp "$f" "$OUT/visualisations/${label}/"
        done
        ok "${label}: last 5 reconstruction snapshots"
    fi
done

# ─── 4. Evaluation results ──────────────────────────────────
banner "4 — Evaluation results"
if [ -d "$EVAL_DIR" ]; then
    mkdir -p "$OUT/eval"
    cp -r "$EVAL_DIR"/. "$OUT/eval/"
    ok "eval/ (metrics_full.csv, summary.csv, summary_table.txt)"
else
    warn "No eval output found at ${EVAL_DIR} — run eval/infer_latent.py first"
fi

# ─── 5. Data splits ─────────────────────────────────────────
banner "5 — Data splits (for reproducibility)"
if [ -d "$SPLITS_DIR" ]; then
    mkdir -p "$OUT/splits"
    cp -r "$SPLITS_DIR"/. "$OUT/splits/"
    ok "splits/ (train.txt, val.txt, test.txt)"
else
    warn "No splits dir found at ${SPLITS_DIR}"
fi

# ─── 6. Run info ────────────────────────────────────────────
banner "6 — run_info.txt"
{
    echo "Collected : $(date)"
    echo "Host      : $(hostname)"
    echo ""
    echo "=== GPU ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
    echo ""
    echo "=== Checkpoint sizes ==="
    find "$OUT/checkpoints" -name "*.pth" -exec ls -lh {} \; 2>/dev/null
    echo ""
    echo "=== Python / PyTorch ==="
    python3 -c "import torch; print('torch', torch.__version__, '/ CUDA', torch.version.cuda)"
} > "$OUT/run_info.txt"
ok "run_info.txt"

# ─── 7. Zip ─────────────────────────────────────────────────
banner "7 — Zipping"
zip -r "$ZIP" "$OUT/" -x "*.DS_Store"
rm -rf "$OUT"

SIZE=$(du -sh "$ZIP" | cut -f1)
echo ""
echo "════════════════════════════════════════"
echo "  Done.  →  ${ZIP}  (${SIZE})"
echo "════════════════════════════════════════"
echo ""
echo "Contents:"
echo "  checkpoints/   image_vae_best.pth, mask_vae_best.pth, latent_diffusion_best.pth"
echo "  logs/          per-run train.csv, val.csv, config.yaml, notes.md"
echo "  visualisations/ last 5 reconstruction snapshots per stage"
echo "  eval/          metrics_full.csv, summary.csv, summary_table.txt"
echo "  splits/        train.txt, val.txt, test.txt"
echo "  run_info.txt   GPU, checkpoint sizes, library versions"
echo ""
echo "Send ${ZIP} back via WeTransfer / Google Drive / scp."
