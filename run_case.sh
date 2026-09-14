#!/bin/bash
# ============================================================
#  Run one named case -- by default the one shown in the paper.
#
#  Reproduces the paper's combo-grid figure for a single study: the seven
#  modality configurations in figure order (FLAIR, T1CE, T1, T2, T1+T2,
#  T1CE+T1+T2, all four), N sampling seeds each, whole tumour only.
#
#  Much faster than a full sweep: one case x 7 configurations.
#
#  On Kaggle, pass the checkpoints and nothing else:
#    export IMAGE_VAE_CKPT=/kaggle/input/.../image_vae/best.pth
#    export MASK_VAE_CKPT=/kaggle/input/.../mask_vae/best.pth
#    export DIFF_CKPT=/kaggle/input/.../diffusion/best.pth
#    export DEVICE=cuda
#    bash run_case.sh
#
#  Options (environment variables):
#    CASE=BraTS-GLI-01023-000   case to run (comma-separate for several)
#    ALL_COMBOS=1               all 15 configurations instead of the paper's 7
#    N_SAMPLES=5  INFER_STEPS=200  SEED=42
#                               samples per configuration, DDIM steps, noise seed
#    DATA_ROOT=...              use existing preprocessed data (vol/, seg/)
#                               instead of fetching the case
#    SPLITS_DIR=...             your real split; warns if CASE is not in test
#    TRANSPARENT=1              save the figure with a transparent background
#
#  !! Check the case is in YOUR test split before quoting its numbers. In the
#     local 2023 split BraTS-GLI-01023-000 is in train.txt; set SPLITS_DIR to
#     the split behind the paper and the run will warn if it is not a test case.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CASE="${CASE:-BraTS-GLI-01023-000}"
DATASET="${DATASET:-brats2023}"
DEVICE="${DEVICE:-$(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')}"
N_SAMPLES="${N_SAMPLES:-5}"
INFER_STEPS="${INFER_STEPS:-200}"
SEED="${SEED:-42}"

CKPT_SRC="${CKPT_SRC:-cache}"
IMAGE_VAE_CKPT="${IMAGE_VAE_CKPT:-$CKPT_SRC/image_vae/best.pth}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-$CKPT_SRC/mask_vae/best.pth}"
DIFF_CKPT="${DIFF_CKPT:-$CKPT_SRC/latent_diffusion/best.pth}"

CASE_DIR="${CASE_DIR:-data/case}"
RAW_DIR="$CASE_DIR/raw/$DATASET"
FULL_DIR="$CASE_DIR/full"
PROC_DIR="$CASE_DIR/roi128"
OUT_DIR="${OUT_DIR:-eval_output/case_n${N_SAMPLES}_steps${INFER_STEPS}_seed${SEED}}"

DATA_ROOT="${DATA_ROOT:-}"
SPLITS_DIR="${SPLITS_DIR:-}"

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
echo "  case    $CASE"
echo "  device  $DEVICE"
echo "  samples $N_SAMPLES   DDIM steps $INFER_STEPS   seed $SEED"

IFS=',' read -ra CASES <<< "$CASE"

# ════════════════════════════════════════════════════════════
banner "Step 1 — Data"
# ════════════════════════════════════════════════════════════
have_all() {   # $1 = data root; true if every requested case is preprocessed
    for c in "${CASES[@]}"; do [ -f "$1/vol/${c}_vol.npy" ] || return 1; done
}

if [ -n "$DATA_ROOT" ]; then
    have_all "$DATA_ROOT" || { echo "Not all of $CASE found under $DATA_ROOT/vol/"; exit 1; }
    echo "[1] Using existing data: $DATA_ROOT"
elif have_all "$PROC_DIR"; then
    echo "[1] Already prepared — skipping  ($PROC_DIR)"
    DATA_ROOT="$PROC_DIR"
else
    echo "[1] Fetching $CASE from HuggingFace (streams until found)"
    python3 scripts/fetch_sample_cases.py \
        --dataset    "$DATASET" \
        --cases      "$CASE" \
        --output_dir "$RAW_DIR"

    echo "[1] Preprocessing → 128³ ROI crop"
    python3 scripts/preprocess_brats.py --data_root "$RAW_DIR"  --output_dir "$FULL_DIR"
    python3 scripts/preprocess_roi.py   --data_root "$FULL_DIR" --output_dir "$PROC_DIR" --crop_size 128
    rm -rf "$FULL_DIR"
    have_all "$PROC_DIR" || { echo "Preprocessing did not produce $CASE under $PROC_DIR/vol/"; exit 1; }
    DATA_ROOT="$PROC_DIR"
fi

# ════════════════════════════════════════════════════════════
banner "Step 2 — Inference"
# ════════════════════════════════════════════════════════════
if [ "${ALL_COMBOS:-0}" = "1" ]; then
    COMBO_FLAG=(--all_combos);          echo "[2] All 15 configurations"
else
    COMBO_FLAG=(--combo_set paper);     echo "[2] The paper's 7 configurations"
fi
EXTRA=()
[ -n "$SPLITS_DIR" ]              && EXTRA+=(--splits_dir "$SPLITS_DIR")
[ "${TRANSPARENT:-0}" = "1" ]     && EXTRA+=(--transparent)

python3 -m eval.infer_latent \
    --diffusion_ckpt      "$DIFF_CKPT" \
    --image_vae_ckpt      "$IMAGE_VAE_CKPT" \
    --mask_vae_ckpt       "$MASK_VAE_CKPT" \
    --data_root           "$DATA_ROOT" \
    --cases               "$CASE" \
    --regions             wt \
    --n_samples           "$N_SAMPLES" \
    --num_inference_steps "$INFER_STEPS" \
    --seed                "$SEED" \
    --num_vis_cases       "${#CASES[@]}" \
    --output_dir          "$OUT_DIR" \
    --device              "$DEVICE" \
    "${COMBO_FLAG[@]}" "${EXTRA[@]}"

# ════════════════════════════════════════════════════════════
banner "Step 3 — Package results"
# ════════════════════════════════════════════════════════════
ZIP_PATH="${ZIP_PATH:-$(dirname "$OUT_DIR")/$(basename "$OUT_DIR").zip}"
rm -f "$ZIP_PATH"
( cd "$(dirname "$OUT_DIR")" && zip -qr "$(basename "$ZIP_PATH")" "$(basename "$OUT_DIR")" )
echo "  $(du -h "$ZIP_PATH" | cut -f1)  →  $ZIP_PATH"

banner "Done"
echo "  Figure → $OUT_DIR/combo_grids/"
echo "  Table  → $OUT_DIR/summary_table.txt"
echo "  Zip    → $ZIP_PATH"
