#!/bin/bash
# ============================================================
#  End-to-end smoke test — "does the pipeline run?"
#
#  Pulls a handful of cases straight out of the HuggingFace tars,
#  preprocesses them, and runs inference with the existing checkpoints.
#  Everything is tiny and fast; the point is to find crashes, not to
#  produce meaningful numbers.
#
#  THE DICE SCORES THIS PRODUCES ARE NOT VALID RESULTS.
#  The cases are whatever came first in the archive, so most of them were
#  in the training set. Treat the output as "it ran", nothing more.
#
#  Usage:
#    bash run_smoke.sh                       # 6 cases, 1 modality combo
#    NUM_CASES=10 ALL_COMBOS=1 bash run_smoke.sh
#    DEVICE=cuda bash run_smoke.sh           # on Kaggle
#
#  Expects the three checkpoints under $CKPT_SRC (default: cache/).
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NUM_CASES="${NUM_CASES:-10}"
DATASET="${DATASET:-brats2023}"
DEVICE="${DEVICE:-$(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')}"

SMOKE_DIR="${SMOKE_DIR:-data/smoke}"
RAW_DIR="$SMOKE_DIR/raw/$DATASET"
FULL_DIR="$SMOKE_DIR/full"
PROC_DIR="$SMOKE_DIR/roi128"
SPLITS_DIR="$SMOKE_DIR/splits"
OUT_DIR="${OUT_DIR:-eval_output/smoke}"

CKPT_SRC="${CKPT_SRC:-cache}"
IMAGE_VAE_CKPT="${IMAGE_VAE_CKPT:-$CKPT_SRC/image_vae/final.pth}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-$CKPT_SRC/mask_vae/best.pth}"
DIFF_CKPT="${DIFF_CKPT:-$CKPT_SRC/latent_diffusion/step_150000.pth}"

# Measured: runtime is ~linear in INFER_STEPS, which is ~89% of the cost at
# 50 steps. 10 cases x 7 combos x 5 samples x 50 steps is ~20 min on a T4.
N_SAMPLES="${N_SAMPLES:-5}"
INFER_STEPS="${INFER_STEPS:-50}"
# "wt" = whole tumour only: the standard binary setting for probabilistic
# segmentation, and the region where uncertainty is interpretable.
REGIONS="${REGIONS:-all}"

banner() { echo ""; echo "════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════"; }

# ════════════════════════════════════════════════════════════
banner "Step 0 — Checks"
# ════════════════════════════════════════════════════════════
missing=0
for f in "$IMAGE_VAE_CKPT" "$MASK_VAE_CKPT" "$DIFF_CKPT"; do
    if [ -f "$f" ]; then echo "  ok      $f"
    else echo "  MISSING $f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo ""; echo "Set CKPT_SRC, or IMAGE_VAE_CKPT / MASK_VAE_CKPT / DIFF_CKPT."; exit 1; }
echo "  device  $DEVICE"
echo "  cases   $NUM_CASES  ($DATASET)"

# ════════════════════════════════════════════════════════════
banner "Step 1 — Fetch $NUM_CASES cases from HuggingFace"
# ════════════════════════════════════════════════════════════
# Streams the tar and stops early, so this reads a few hundred MB rather
# than the full ~32 GB archive.
if [ -d "$RAW_DIR" ] && [ "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    echo "[1] Already fetched — skipping  ($RAW_DIR)"
else
    python3 scripts/fetch_sample_cases.py \
        --dataset    "$DATASET" \
        --num_cases  "$NUM_CASES" \
        --output_dir "$RAW_DIR"
fi

# ════════════════════════════════════════════════════════════
banner "Step 2 — Preprocess → .npy → 128³ ROI crops"
# ════════════════════════════════════════════════════════════
if [ -d "$PROC_DIR/vol" ] && [ "$(ls -A "$PROC_DIR/vol" 2>/dev/null)" ]; then
    echo "[2] Already preprocessed — skipping  ($PROC_DIR)"
else
    python3 scripts/preprocess_brats.py \
        --data_root  "$RAW_DIR" \
        --output_dir "$FULL_DIR"

    python3 scripts/preprocess_roi.py \
        --data_root  "$FULL_DIR" \
        --output_dir "$PROC_DIR" \
        --crop_size  128

    # Full-res intermediates are ~143 MB per case and no longer needed.
    rm -rf "$FULL_DIR"
fi

# ════════════════════════════════════════════════════════════
banner "Step 3 — Split file (every case in test)"
# ════════════════════════════════════════════════════════════
# Not resplit_data.py: with this few cases a 70/20/10 split would leave
# test with one case. We want all of them evaluated.
mkdir -p "$SPLITS_DIR"
python3 - "$PROC_DIR" "$SPLITS_DIR" <<'PY'
import sys
from pathlib import Path
proc, splits = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name.replace("_vol.npy", "") for p in (proc / "vol").glob("*_vol.npy"))
if not names:
    raise SystemExit(f"No cases found in {proc/'vol'}")
for fn in ("test.txt", "train.txt", "val.txt"):
    (splits / fn).write_text("\n".join(names))
print(f"  {len(names)} cases → {splits}/test.txt")
PY

# ════════════════════════════════════════════════════════════
banner "Step 4 — Inference"
# ════════════════════════════════════════════════════════════
COMBO_FLAG=(--modality_mask all)
if [ "${ALL_COMBOS:-0}" = "1" ]; then
    COMBO_FLAG=(--all_combos)
    echo "[4] Sweeping all 15 modality combinations"
elif [ -n "${COMBO_SET:-}" ]; then
    COMBO_FLAG=(--combo_set "$COMBO_SET")
    echo "[4] Combo set: $COMBO_SET"
else
    echo "[4] Single combo (all modalities present)."
    echo "    COMBO_SET=focused for the 7-combo set, ALL_COMBOS=1 for all 15."
fi

_run_shard() {   # $1 = shard index, $2 = num shards, $3 = output dir
    python3 -m eval.infer_latent \
        --diffusion_ckpt      "$DIFF_CKPT" \
        --image_vae_ckpt      "$IMAGE_VAE_CKPT" \
        --mask_vae_ckpt       "$MASK_VAE_CKPT" \
        --data_root           "$PROC_DIR" \
        --splits_dir          "$SPLITS_DIR" \
        --output_dir          "$3" \
        --n_samples           "$N_SAMPLES" \
        --num_inference_steps "$INFER_STEPS" \
        --num_shards          "$2" \
        --shard_index         "$1" \
        --regions             "$REGIONS" \
        "${COMBO_FLAG[@]}" \
        --device              "$DEVICE"
}

# How many GPUs to spread the cases over. Every (case, combo) is independent,
# so this is one process per GPU with no communication — not DDP.
if [ "$DEVICE" = "cuda" ]; then
    DETECTED=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)
else
    DETECTED=1
fi
NUM_GPUS="${NUM_GPUS:-$DETECTED}"

if [ "$NUM_GPUS" -le 1 ]; then
    echo "[4] Single process (NUM_GPUS=$NUM_GPUS)"
    _run_shard 0 1 "$OUT_DIR"
else
    echo "[4] Sharding cases across $NUM_GPUS GPUs"
    SHARD_DIRS=()
    PIDS=()
    for ((g = 0; g < NUM_GPUS; g++)); do
        SDIR="${OUT_DIR}/shard${g}"
        SHARD_DIRS+=("$SDIR")
        mkdir -p "$SDIR"
        echo "    GPU $g → $SDIR"
        CUDA_VISIBLE_DEVICES="$g" _run_shard "$g" "$NUM_GPUS" "$SDIR" \
            > "$SDIR/log.txt" 2>&1 &
        PIDS+=("$!")
    done

    # Wait on each explicitly so one shard crashing fails the whole run
    # instead of silently producing a partial table.
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "    shard $i done"
        else
            echo "    shard $i FAILED — see ${SHARD_DIRS[$i]}/log.txt"
            tail -20 "${SHARD_DIRS[$i]}/log.txt" | sed 's|^|      |'
            FAILED=1
        fi
    done
    [ "$FAILED" -eq 0 ] || { echo "[4] A shard failed — not merging."; exit 1; }

    echo "[4] Merging shards"
    python3 scripts/merge_shards.py \
        --shard_dirs "${SHARD_DIRS[@]}" \
        --output_dir "$OUT_DIR"
fi

# ════════════════════════════════════════════════════════════
banner "Step 5 — Package results"
# ════════════════════════════════════════════════════════════
# One file to grab from Kaggle's Output tab instead of clicking through
# a directory tree of PNGs.
ZIP_PATH="${ZIP_PATH:-$OUT_DIR/../smoke_results.zip}"
rm -f "$ZIP_PATH"
( cd "$(dirname "$OUT_DIR")" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$OUT_DIR")" )
echo "  $(du -h "$ZIP_PATH" | cut -f1)  →  $ZIP_PATH"

# ════════════════════════════════════════════════════════════
banner "Smoke test passed"
# ════════════════════════════════════════════════════════════
echo "  Output → $OUT_DIR"
ls -1 "$OUT_DIR" 2>/dev/null | sed 's|^|    |'
echo "  Zip    → $ZIP_PATH"
echo ""
echo "  The pipeline runs end to end. The Dice numbers above are NOT valid"
echo "  results — these cases were almost certainly in the training set."
