#!/bin/bash
# ============================================================
#  Autoencoder evaluation — the ceiling for the segmentation table.
#
#  Both autoencoders are frozen before diffusion, so their reconstruction
#  fidelity bounds what any diffusion model in that latent space can reach.
#  This produces those numbers:
#
#    MaskVAE   Dice, IoU, SSIM, PSNR per region (WT / TC / ET)
#    ImageVAE  SSIM, PSNR per sequence (FLAIR / T1ce / T1 / T2)
#
#  Both write a per-case CSV as well as the console table, so the numbers
#  survive a Kaggle session ending.
#
#  Data is fetched and preprocessed the same way run_smoke.sh does it, into
#  the same data/smoke cache, so having run either one makes the other skip
#  straight to inference. Pass checkpoints and nothing else:
#
#    export IMAGE_VAE_CKPT=/kaggle/input/.../image_vae/best.pth
#    export MASK_VAE_CKPT=/kaggle/input/.../mask_vae/best.pth
#    export NUM_CASES=50 DEVICE=cuda
#    bash run_vae_eval.sh
#
#  !! THE NUMBERS FROM FETCHED DATA ARE NOT VALID PAPER RESULTS. The cases
#     are whatever came first in the archive, so most were in the training
#     set and reconstruction fidelity on them is optimistic. For Table 1,
#     point DATA_ROOT and SPLITS_DIR at the real held-out test split.
#
#  Usage:
#    bash run_vae_eval.sh                    # fetch, then evaluate
#    NUM_CASES=50 bash run_vae_eval.sh       # more cases
#    DEVICE=cuda bash run_vae_eval.sh        # on Kaggle
#    DATA_ROOT=... SPLITS_DIR=... bash run_vae_eval.sh   # real test split
#
#  DATA_ROOT / SPLITS_DIR override the fetch entirely when both are set.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DEVICE="${DEVICE:-$(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')}"

CKPT_SRC="${CKPT_SRC:-cache}"
IMAGE_VAE_CKPT="${IMAGE_VAE_CKPT:-$CKPT_SRC/image_vae/best.pth}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-$CKPT_SRC/mask_vae/best.pth}"

OUT_DIR="${OUT_DIR:-eval_out}"
SPLIT="${SPLIT:-test}"

# Cases drawn in the PNG grids. Metrics always cover the whole split unless
# MAX_CASES caps them -- scoring only the eight cases that happen to be drawn
# would give a table with a sample size of eight.
N_CASES="${N_CASES:-8}"
MAX_CASES="${MAX_CASES:-}"

# Same cache and same variable names as run_smoke.sh, so whichever script runs
# first pays the download and the other skips it.
NUM_CASES="${NUM_CASES:-10}"
DATASET="${DATASET:-brats2023}"
SMOKE_DIR="${SMOKE_DIR:-data/smoke}"
RAW_DIR="$SMOKE_DIR/raw/$DATASET"
FULL_DIR="$SMOKE_DIR/full"
PROC_DIR="$SMOKE_DIR/roi128"
SMOKE_SPLITS="$SMOKE_DIR/splits"

# Set both to evaluate on real data instead of the fetched sample.
DATA_ROOT="${DATA_ROOT:-}"
SPLITS_DIR="${SPLITS_DIR:-}"

banner() { echo ""; echo "════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════"; }

# ════════════════════════════════════════════════════════════
banner "Step 0 — Checks"
# ════════════════════════════════════════════════════════════
missing=0
for f in "$IMAGE_VAE_CKPT" "$MASK_VAE_CKPT"; do
    if [ -f "$f" ]; then echo "  ok      $f"
    else echo "  MISSING $f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo ""; echo "Set CKPT_SRC, or IMAGE_VAE_CKPT / MASK_VAE_CKPT."; exit 1; }

echo "  device  $DEVICE"
echo "  split   $SPLIT"

# What the checkpoints expect, so a path mismatch is visible here rather than
# as a confusing failure several steps later.
python3 - "$IMAGE_VAE_CKPT" "$MASK_VAE_CKPT" <<'PY'
import sys, torch
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=True).get("config", {})
    print(f"  recorded  {p.split('/')[-2]:<10} data_root={c.get('data_root')!r} "
          f"splits_dir={c.get('splits_dir')!r} subregion={c.get('subregion_mode')}")
PY

# ════════════════════════════════════════════════════════════
banner "Step 1 — Data"
# ════════════════════════════════════════════════════════════
if [ -n "$DATA_ROOT" ] && [ -n "$SPLITS_DIR" ]; then
    echo "[1] Using the data given: $DATA_ROOT"
    echo "    splits: $SPLITS_DIR"
    FETCHED=0
else
    [ -z "$DATA_ROOT$SPLITS_DIR" ] || {
        echo "Set both DATA_ROOT and SPLITS_DIR, or neither."; exit 1; }

    # Same steps and same cache as run_smoke.sh, so running either one first
    # means the other skips the download.
    if [ -d "$PROC_DIR/vol" ] && [ "$(ls -A "$PROC_DIR/vol" 2>/dev/null)" ]; then
        echo "[1] Already prepared — skipping  ($PROC_DIR)"
    else
        if [ -d "$RAW_DIR" ] && [ "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
            echo "[1] Already fetched — skipping  ($RAW_DIR)"
        else
            echo "[1] Fetching $NUM_CASES cases from HuggingFace"
            python3 scripts/fetch_sample_cases.py \
                --dataset    "$DATASET" \
                --num_cases  "$NUM_CASES" \
                --output_dir "$RAW_DIR"
        fi

        echo "[1] Preprocessing → 128³ ROI crops"
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

    # Every fetched case goes in test: a 70/20/10 split of this few would
    # leave one case to evaluate.
    mkdir -p "$SMOKE_SPLITS"
    python3 - "$PROC_DIR" "$SMOKE_SPLITS" <<'PY'
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
    DATA_ROOT="$PROC_DIR"
    SPLITS_DIR="$SMOKE_SPLITS"
    FETCHED=1
fi

# Shared flags, built once so the two runs cannot drift apart.
COMMON=(--split "$SPLIT" --n_cases "$N_CASES" --device "$DEVICE"
        --data_root "$DATA_ROOT" --splits_dir "$SPLITS_DIR")
[ -n "$MAX_CASES" ] && COMMON+=(--max_cases "$MAX_CASES")

# ════════════════════════════════════════════════════════════
banner "Step 2 — MaskVAE  (Dice, IoU, SSIM, PSNR)"
# ════════════════════════════════════════════════════════════
# The annotation ceiling. Channel width is read from the checkpoint, so this
# handles both the 3-channel region model and the 4-channel subregion one.
python3 -m eval.eval_mask_vae \
    --checkpoint "$MASK_VAE_CKPT" \
    --output_dir "$OUT_DIR/mask_vae" \
    "${COMMON[@]}"

# ════════════════════════════════════════════════════════════
banner "Step 3 — ImageVAE  (SSIM, PSNR per sequence)"
# ════════════════════════════════════════════════════════════
# --drop_mods additionally reconstructs from a random modality subset, which
# is the condition the diffusion model actually operates under.
python3 -m eval.eval_image_vae \
    --checkpoint "$IMAGE_VAE_CKPT" \
    --output_dir "$OUT_DIR/image_vae" \
    --drop_mods \
    "${COMMON[@]}"

# ════════════════════════════════════════════════════════════
banner "Step 4 — Package results"
# ════════════════════════════════════════════════════════════
# One file to grab from Kaggle's Output tab.
ZIP_PATH="${ZIP_PATH:-$OUT_DIR/../vae_eval_results.zip}"
rm -f "$ZIP_PATH"
( cd "$(dirname "$OUT_DIR")" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$OUT_DIR")" )
echo "  $(du -h "$ZIP_PATH" | cut -f1)  →  $ZIP_PATH"

# ════════════════════════════════════════════════════════════
banner "Done"
# ════════════════════════════════════════════════════════════
if [ "$FETCHED" = "1" ]; then
    echo "  !! These came from fetched sample cases, most of which were in the"
    echo "     training set. NOT valid Table 1 numbers -- for those, set"
    echo "     DATA_ROOT and SPLITS_DIR to the real held-out test split."
else
    echo "  Tables above are the autoencoder ceiling for the paper."
fi
echo ""
echo "  CSVs:"
find "$OUT_DIR" -name "*_metrics.csv" | sed 's|^|    |'
echo "  Zip  → $ZIP_PATH"
