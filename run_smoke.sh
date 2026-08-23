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

NUM_CASES="${NUM_CASES:-6}"
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

# Deliberately tiny — this is a crash test, not an evaluation.
N_SAMPLES="${N_SAMPLES:-2}"
INFER_STEPS="${INFER_STEPS:-10}"

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
else
    echo "[4] Single combo (all modalities present). ALL_COMBOS=1 to sweep."
fi

python3 -m eval.infer_latent \
    --diffusion_ckpt      "$DIFF_CKPT" \
    --image_vae_ckpt      "$IMAGE_VAE_CKPT" \
    --mask_vae_ckpt       "$MASK_VAE_CKPT" \
    --data_root           "$PROC_DIR" \
    --splits_dir          "$SPLITS_DIR" \
    --output_dir          "$OUT_DIR" \
    --n_samples           "$N_SAMPLES" \
    --num_inference_steps "$INFER_STEPS" \
    "${COMBO_FLAG[@]}" \
    --device              "$DEVICE"

# ════════════════════════════════════════════════════════════
banner "Smoke test passed"
# ════════════════════════════════════════════════════════════
echo "  Output → $OUT_DIR"
ls -1 "$OUT_DIR" 2>/dev/null | sed 's|^|    |'
echo ""
echo "  The pipeline runs end to end. The Dice numbers above are NOT valid"
echo "  results — these cases were almost certainly in the training set."
