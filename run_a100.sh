#!/bin/bash
# ============================================================
#  BraTS latent-diffusion pipeline  —  A100 / no Docker
#
#  Usage:
#    export DATA_RAW=/path/to/raw-brats-nii-gz-cases
#    bash run_a100.sh
#
#  Or pass DATA_RAW inline:
#    DATA_RAW=/data/brats2023 bash run_a100.sh
#
#  The script is idempotent — re-running skips already-done steps.
# ============================================================
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# ─── PATHS — edit or override with env vars ─────────────────
DATA_RAW="${DATA_RAW:-data/raw}"                     # raw BraTS .nii.gz folders (auto-downloaded)
DATA_PROC="${DATA_PROC:-data/brats_roi128}"          # 128³ ROI crops
LATENT_DIR="${LATENT_DIR:-data/brats_latents}"       # cached mask latents
SPLITS_DIR="${SPLITS_DIR:-splits/brats_roi128_full}" # train/val/test splits
CKPT_DIR="${CKPT_DIR:-checkpoints}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_VAE="${BATCH_VAE:-4}"       # increase to 8 on H200
BATCH_DIFF="${BATCH_DIFF:-32}"    # increase to 64 on H200

# ─── ENV SETUP ──────────────────────────────────────────────
ENV_NAME="medseg"
CUDA_WHEEL="cu124"    # change to cu118 / cu121 if needed for your driver

_setup_env() {
    # Try conda first, fall back to venv
    CONDA_INIT=""
    for f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
              "$HOME/anaconda3/etc/profile.d/conda.sh" \
              "/opt/conda/etc/profile.d/conda.sh"; do
        [ -f "$f" ] && { CONDA_INIT="$f"; break; }
    done

    if [ -n "$CONDA_INIT" ]; then
        # shellcheck source=/dev/null
        source "$CONDA_INIT"
        conda env list | grep -q "^${ENV_NAME} " || \
            conda create -y -n "$ENV_NAME" python=3.11
        conda activate "$ENV_NAME"
    else
        echo "[env] conda not found — using virtualenv"
        [ -d ".venv" ] || python3 -m venv .venv
        # shellcheck source=/dev/null
        source .venv/bin/activate
    fi

    # Install torch if not already there
    python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
        echo "[env] Installing PyTorch (${CUDA_WHEEL})..."
        pip install -q torch torchvision \
            --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
    }

    pip install -q -r requirements.txt
    echo "[env] Environment ready  ($(python3 --version))"
}

# ─── HuggingFace repo ───────────────────────────────────────
HF_REPO="tom-ngh/brats-data"

# ─── HELPERS ────────────────────────────────────────────────
_latest_ckpt() {
    # $1 = glob pattern, $2 = filename (best.pth / final.pth)
    ls -t "$CKPT_DIR"/$1/"$2" 2>/dev/null | head -1 || true
}

banner() { echo ""; echo "════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════"; }

# ════════════════════════════════════════════════════════════
banner "Step 0 — Environment"
# ════════════════════════════════════════════════════════════
_setup_env

# ════════════════════════════════════════════════════════════
banner "Step 1 — Download BraTS data from HuggingFace"
# ════════════════════════════════════════════════════════════
mkdir -p "$DATA_RAW"

_hf_download() {
    local name="$1"
    if [ ! -d "$DATA_RAW/$name" ]; then
        echo "[1] Downloading ${name}.tar from HuggingFace..."
        huggingface-cli download "$HF_REPO" "${name}.tar" --local-dir "$DATA_RAW"
        echo "[1] Extracting $name..."
        mkdir -p "$DATA_RAW/$name"
        tar -xf "$DATA_RAW/${name}.tar" -C "$DATA_RAW/$name" --strip-components=1
        rm "$DATA_RAW/${name}.tar"
    else
        echo "[1] $name already present — skipping download"
    fi
}

_hf_download "brats2023"
_hf_download "brats2024"

# ════════════════════════════════════════════════════════════
banner "Step 2 — Preprocess raw BraTS → 128³ ROI crops"
# ════════════════════════════════════════════════════════════
PROC_SENTINEL="$DATA_PROC/.roi_preprocessed_128"
if [ ! -f "$PROC_SENTINEL" ]; then
    FULL_DIR="${DATA_PROC}_full"
    mkdir -p "$FULL_DIR"

    for dataset in brats2023 brats2024; do
        SENTINEL="$FULL_DIR/.preprocessed_${dataset}"
        if [ ! -f "$SENTINEL" ]; then
            echo "[2a] Converting $dataset .nii.gz → .npy..."
            python3 scripts/preprocess_brats.py \
                --data_root  "$DATA_RAW/$dataset" \
                --output_dir "$FULL_DIR"
            touch "$SENTINEL"
        else
            echo "[2a] $dataset already converted — skipping"
        fi
    done

    echo "[2b] Extracting 128³ ROI crops centred on tumour..."
    python3 scripts/preprocess_roi.py \
        --data_root  "$FULL_DIR" \
        --output_dir "$DATA_PROC" \
        --crop_size  128

    echo "[2c] Removing full-res intermediates (saves ~400 GB)..."
    rm -rf "$FULL_DIR"

    touch "$PROC_SENTINEL"
else
    echo "[2] Already done — skipping  ($DATA_PROC)"
fi

# ════════════════════════════════════════════════════════════
banner "Step 3 — Create train / val / test splits"
# ════════════════════════════════════════════════════════════
if [ ! -f "$SPLITS_DIR/train.txt" ]; then
    python3 scripts/resplit_data.py \
        --data_root  "$DATA_PROC" \
        --output_dir "$SPLITS_DIR"
else
    echo "[3] Splits already exist — skipping  ($SPLITS_DIR)"
fi

# ════════════════════════════════════════════════════════════
banner "Step 4 — Train ImageVAE  (4× compression, 128→32)"
# ════════════════════════════════════════════════════════════
IMAGE_VAE_CKPT=$(_latest_ckpt "image_vae_*" "best.pth")
if [ -z "$IMAGE_VAE_CKPT" ]; then
    python3 -m training.train_image_vae  \
        --data_root        "$DATA_PROC" \
        --splits_dir       "$SPLITS_DIR" \
        --checkpoint_dir   "$CKPT_DIR" \
        --crop_size        128 \
        --encoder_channels 64,128,256 \
        --num_epochs       200 \
        --batch_size       "$BATCH_VAE" \
        --num_workers      "$NUM_WORKERS" \
        --val_every        200 \
        --early_stop_patience 25 \
        --device           "$DEVICE"
    IMAGE_VAE_CKPT=$(_latest_ckpt "image_vae_*" "best.pth")
else
    echo "[4] Found ImageVAE checkpoint — skipping  ($IMAGE_VAE_CKPT)"
fi

# ════════════════════════════════════════════════════════════
banner "Step 5 — Train MaskVAE  (4× compression, subregion mode)"
# ════════════════════════════════════════════════════════════
MASK_VAE_CKPT=$(_latest_ckpt "mask_vae_*" "best.pth")
if [ -z "$MASK_VAE_CKPT" ]; then
    python3 -m training.train_mask_vae \
        --data_root          "$DATA_PROC" \
        --splits_dir         "$SPLITS_DIR" \
        --checkpoint_dir     "$CKPT_DIR" \
        --crop_size          128 \
        --mask_vae_channels  32,64,128 \
        --latent_channels    4 \
        --num_epochs         200 \
        --batch_size         "$BATCH_VAE" \
        --num_workers        "$NUM_WORKERS" \
        --val_every          200 \
        --early_stop_patience 25 \
        --device             "$DEVICE" \
        --subregion_mode
    MASK_VAE_CKPT=$(_latest_ckpt "mask_vae_*" "best.pth")
else
    echo "[5] Found MaskVAE checkpoint — skipping  ($MASK_VAE_CKPT)"
fi

# ════════════════════════════════════════════════════════════
banner "Step 6 — Cache mask latents"
# ════════════════════════════════════════════════════════════
# We cache only mask latents (~0.6 GB total for ~1250 cases at 32³).
# Image latents are NOT cached — 15 modality combos × 1251 cases = 620 GB.
# The diffusion UNet runs the frozen ImageVAE encoder on-the-fly instead.
if [ -z "$MASK_VAE_CKPT" ]; then
    echo "[6] ERROR: no MaskVAE checkpoint found — cannot cache latents"
    exit 1
fi

CACHE_SENTINEL="$LATENT_DIR/.cached_mask"
if [ ! -f "$CACHE_SENTINEL" ]; then
    python3 scripts/cache_latents.py \
        --data_root     "$DATA_PROC" \
        --splits_dir    "$SPLITS_DIR" \
        --output_dir    "$LATENT_DIR" \
        --mask_vae_ckpt "$MASK_VAE_CKPT" \
        --batch_size    8 \
        --device        "$DEVICE"
    touch "$CACHE_SENTINEL"
else
    echo "[6] Latents already cached — skipping  ($LATENT_DIR)"
fi

# ════════════════════════════════════════════════════════════
banner "Step 7 — Train latent diffusion UNet"
# ════════════════════════════════════════════════════════════
DIFF_CKPT=$(_latest_ckpt "latent_diffusion_*" "best.pth")
if [ -z "$DIFF_CKPT" ]; then
    python3 -m training.train_latent_diffusion \
        --latent_dir     "$LATENT_DIR" \
        --image_data     "$DATA_PROC" \
        --image_vae_ckpt "$IMAGE_VAE_CKPT" \
        --mask_vae_ckpt  "$MASK_VAE_CKPT" \
        --splits_dir     "$SPLITS_DIR" \
        --checkpoint_dir "$CKPT_DIR" \
        --num_steps      200000 \
        --batch_size     "$BATCH_DIFF" \
        --num_workers    "$NUM_WORKERS" \
        --device         "$DEVICE"
else
    echo "[7] Found diffusion checkpoint — skipping  ($DIFF_CKPT)"
fi

# ════════════════════════════════════════════════════════════
echo ""
echo "Done."
echo ""
echo "VAE checkpoints:"
echo "  ImageVAE  : $IMAGE_VAE_CKPT"
echo "  MaskVAE   : $MASK_VAE_CKPT"
echo "  Latents   : $LATENT_DIR"
