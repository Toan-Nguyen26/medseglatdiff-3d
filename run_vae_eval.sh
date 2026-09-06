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
#  Usage:
#    bash run_vae_eval.sh                    # whole test split
#    MAX_CASES=50 bash run_vae_eval.sh       # quick pass
#    DEVICE=cuda bash run_vae_eval.sh        # on Kaggle
#
#  On Kaggle, point it at the checkpoints and the data:
#    export IMAGE_VAE_CKPT=/kaggle/input/.../image_vae/best.pth
#    export MASK_VAE_CKPT=/kaggle/input/.../mask_vae/best.pth
#    export DATA_ROOT=/kaggle/input/.../brats_roi128_2023
#    export SPLITS_DIR=/kaggle/input/.../splits_full
#    export MAX_CASES=50 DEVICE=cuda
#    bash run_vae_eval.sh
#
#  DATA_ROOT / SPLITS_DIR may be left unset when running on the machine that
#  trained the checkpoints; the paths recorded inside them are used instead.
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

# Empty means "use whatever the checkpoint recorded".
DATA_ROOT="${DATA_ROOT:-}"
SPLITS_DIR="${SPLITS_DIR:-}"

banner() { echo ""; echo "════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════"; }

# Shared flags, built once so the two runs cannot drift apart.
COMMON=(--split "$SPLIT" --n_cases "$N_CASES" --device "$DEVICE")
[ -n "$MAX_CASES"  ] && COMMON+=(--max_cases  "$MAX_CASES")
[ -n "$DATA_ROOT"  ] && COMMON+=(--data_root  "$DATA_ROOT")
[ -n "$SPLITS_DIR" ] && COMMON+=(--splits_dir "$SPLITS_DIR")

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
echo "  cases   ${MAX_CASES:-all}  (grid draws $N_CASES)"
echo "  data    ${DATA_ROOT:-<from checkpoint>}"
echo "  splits  ${SPLITS_DIR:-<from checkpoint>}"

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
banner "Step 1 — MaskVAE  (Dice, IoU, SSIM, PSNR)"
# ════════════════════════════════════════════════════════════
# The annotation ceiling. Channel width is read from the checkpoint, so this
# handles both the 3-channel region model and the 4-channel subregion one.
python3 -m eval.eval_mask_vae \
    --checkpoint "$MASK_VAE_CKPT" \
    --output_dir "$OUT_DIR/mask_vae" \
    "${COMMON[@]}"

# ════════════════════════════════════════════════════════════
banner "Step 2 — ImageVAE  (SSIM, PSNR per sequence)"
# ════════════════════════════════════════════════════════════
# --drop_mods additionally reconstructs from a random modality subset, which
# is the condition the diffusion model actually operates under.
python3 -m eval.eval_image_vae \
    --checkpoint "$IMAGE_VAE_CKPT" \
    --output_dir "$OUT_DIR/image_vae" \
    --drop_mods \
    "${COMMON[@]}"

# ════════════════════════════════════════════════════════════
banner "Step 3 — Package results"
# ════════════════════════════════════════════════════════════
# One file to grab from Kaggle's Output tab.
ZIP_PATH="${ZIP_PATH:-$OUT_DIR/../vae_eval_results.zip}"
rm -f "$ZIP_PATH"
( cd "$(dirname "$OUT_DIR")" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$OUT_DIR")" )
echo "  $(du -h "$ZIP_PATH" | cut -f1)  →  $ZIP_PATH"

# ════════════════════════════════════════════════════════════
banner "Done"
# ════════════════════════════════════════════════════════════
echo "  Tables above are the autoencoder ceiling for the paper."
echo ""
echo "  CSVs:"
find "$OUT_DIR" -name "*_metrics.csv" | sed 's|^|    |'
echo "  Zip  → $ZIP_PATH"
