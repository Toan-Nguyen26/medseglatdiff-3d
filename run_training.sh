#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Full training pipeline for BraTS missing-modality segmentation
# Run inside Docker container or directly on H200
#
# Expected data layout (mount or download into DATA_RAW before running):
#   $DATA_RAW/brats2023/   — raw BraTS 2023 GLI training cases
#   $DATA_RAW/brats2024/   — raw BraTS 2024 GLI training cases
# ---------------------------------------------------------------------------

DATA_RAW="${DATA_RAW:-/workspace/data/raw}"
DATA_PROCESSED="/workspace/data/brats_combined"
SPLITS_DIR="/workspace/splits/brats_combined_full"
CHECKPOINT_DIR="/workspace/checkpoints"
DEVICE="${DEVICE:-cuda}"

# H200 training settings
EPOCHS=300
BATCH_SIZE=4
BASE_CHANNELS=64
CHANNEL_MULTS="1,2,4,8"
NUM_TIMESTEPS=1000
NUM_INFERENCE_STEPS=10   # fast val checks during training; use 50 at eval
VAL_EVERY=500
LOG_EVERY=100
NUM_WORKERS=8

echo "============================================================"
echo " BraTS pixel-space diffusion — full pipeline"
echo " Device : $DEVICE"
echo " Data   : $DATA_PROCESSED"
echo " Splits : $SPLITS_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0 — Download raw data (skip if already present via volume mount)
# ---------------------------------------------------------------------------

HF_2023="https://huggingface.co/tom-ngh/brats-data/resolve/main/brats2023.tar"
HF_2024="https://huggingface.co/tom-ngh/brats-data/resolve/main/brats2024.tar"

mkdir -p "$DATA_RAW"

if [ ! -d "$DATA_RAW/brats2023" ]; then
    echo ""
    echo "[0/4] Downloading BraTS 2023 from HuggingFace..."
    wget -q --show-progress "$HF_2023" -O "$DATA_RAW/brats2023.tar"
    echo "  Extracting..."
    mkdir -p "$DATA_RAW/brats2023"
    tar -xf "$DATA_RAW/brats2023.tar" -C "$DATA_RAW/brats2023" --strip-components=1
    rm "$DATA_RAW/brats2023.tar"
fi

if [ ! -d "$DATA_RAW/brats2024" ]; then
    echo ""
    echo "[0/4] Downloading BraTS 2024 from HuggingFace..."
    wget -q --show-progress "$HF_2024" -O "$DATA_RAW/brats2024.tar"
    echo "  Extracting..."
    mkdir -p "$DATA_RAW/brats2024"
    tar -xf "$DATA_RAW/brats2024.tar" -C "$DATA_RAW/brats2024" --strip-components=1
    rm "$DATA_RAW/brats2024.tar"
fi

echo "[0/4] Raw data ready at $DATA_RAW"

# ---------------------------------------------------------------------------
# Step 1 — Preprocess raw data
# ---------------------------------------------------------------------------

mkdir -p "$DATA_PROCESSED"

if [ ! -f "$DATA_PROCESSED/.processed_2023" ]; then
    echo ""
    echo "[1/4] Preprocessing BraTS 2023..."
    python3 scripts/preprocess_brats.py \
        --data_root "$DATA_RAW/brats2023" \
        --output_dir "$DATA_PROCESSED"
    touch "$DATA_PROCESSED/.processed_2023"
else
    echo "[1/4] BraTS 2023 already preprocessed, skipping."
fi

if [ ! -f "$DATA_PROCESSED/.processed_2024" ]; then
    echo ""
    echo "[1/4] Preprocessing BraTS 2024..."
    python3 scripts/preprocess_brats.py \
        --data_root "$DATA_RAW/brats2024" \
        --output_dir "$DATA_PROCESSED"
    touch "$DATA_PROCESSED/.processed_2024"
else
    echo "[1/4] BraTS 2024 already preprocessed, skipping."
fi

# ---------------------------------------------------------------------------
# Step 2 — Create train/val/test splits
# ---------------------------------------------------------------------------

if [ ! -f "$SPLITS_DIR/train.txt" ]; then
    echo ""
    echo "[2/4] Creating splits..."
    python3 scripts/resplit_data.py \
        --data_root "$DATA_PROCESSED" \
        --output_dir "$SPLITS_DIR"
else
    echo "[2/4] Splits already exist, skipping."
fi

# ---------------------------------------------------------------------------
# Step 3 — Train
# ---------------------------------------------------------------------------

echo ""
echo "[3/4] Starting training..."
python3 -m training.train_diffusion \
    --data_root      "$DATA_PROCESSED" \
    --splits_dir     "$SPLITS_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device         "$DEVICE" \
    --num_epochs     "$EPOCHS" \
    --batch_size     "$BATCH_SIZE" \
    --base_channels  "$BASE_CHANNELS" \
    --channel_mults  "$CHANNEL_MULTS" \
    --num_timesteps        "$NUM_TIMESTEPS" \
    --num_inference_steps  "$NUM_INFERENCE_STEPS" \
    --val_every      "$VAL_EVERY" \
    --log_every      "$LOG_EVERY" \
    --roi_crop_ratio 0.7 \
    --roi_max_offset 20 \
    --num_workers    "$NUM_WORKERS"

# ---------------------------------------------------------------------------
# Step 4 — Evaluate on all 15 modality combos
# ---------------------------------------------------------------------------

BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/diffusion_*/best.pth 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "[4/4] No best.pth found, using final.pth..."
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/diffusion_*/final.pth 2>/dev/null | head -1)
fi

if [ -n "$BEST_CKPT" ]; then
    echo ""
    echo "[4/4] Evaluating checkpoint: $BEST_CKPT"
    python3 -m eval.eval_missing_modality_pixel \
        --data_root  "$DATA_PROCESSED" \
        --splits_dir "$SPLITS_DIR" \
        --checkpoint "$BEST_CKPT" \
        --n_samples           5 \
        --num_inference_steps 50 \
        --device "$DEVICE"
else
    echo "[4/4] No checkpoint found — skipping eval."
fi

echo ""
echo "Done."
