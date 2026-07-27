#!/bin/bash
# ============================================================
#  BraTS latent-diffusion — multi-GPU (DDP) training
#
#  Retrains ImageVAE and the diffusion UNet across all visible GPUs.
#  Does NOT preprocess, cache latents, or retrain the MaskVAE — those are
#  already done and unchanged. Use run_a100.sh / run_h200.sh for those.
#
#  Usage:
#    bash run_multigpu.sh                # auto-detect GPU count
#    NUM_GPUS=4 bash run_multigpu.sh     # pin GPU count
#    SKIP_VAE=1 bash run_multigpu.sh     # diffusion only
#
#  Re-running skips whichever stage already has a checkpoint.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ─── PATHS ──────────────────────────────────────────────────
DATA_PROC="${DATA_PROC:-data/brats_roi128}"
LATENT_DIR="${LATENT_DIR:-data/brats_latents}"
SPLITS_DIR="${SPLITS_DIR:-splits/brats_roi128_full}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ─── SCALE ──────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"

# Per-GPU batch. Effective batch = value × NUM_GPUS.
BATCH_VAE="${BATCH_VAE:-8}"
BATCH_DIFF="${BATCH_DIFF:-32}"

# Validation set size. The old default of 4 made val metrics pure sampling
# noise — mean_dice swung 0.36↔0.66 between checks on a flat model.
NUM_VAL_VAE="${NUM_VAL_VAE:-50}"
NUM_VAL_DIFF="${NUM_VAL_DIFF:-30}"

# Artefact retention. Without a cap these runs write ~200 periodic .pth per
# trainer and ~1000 validation PNGs. Only the newest N of each are kept;
# best.pth and final.pth are never pruned.
KEEP_CKPTS="${KEEP_CKPTS:-10}"
KEEP_VIS="${KEEP_VIS:-20}"

# Budgets.
#
# EPOCHS_VAE is per-rank work: DistributedSampler shards the data, so one
# epoch is NUM_GPUS× fewer optimizer steps than on a single GPU. The default
# below is scaled by GPU count so the step count matches a long single-GPU
# run regardless of how many GPUs you use. The previous run did 200 epochs
# ≈ 11.2k steps and was still improving when it stopped — hence the jump.
EPOCHS_VAE_BASE="${EPOCHS_VAE_BASE:-800}"
EPOCHS_VAE="${EPOCHS_VAE:-$((EPOCHS_VAE_BASE * NUM_GPUS))}"

# Diffusion counts optimizer steps, which are GPU-count independent — but
# each step now covers NUM_GPUS× more samples.
STEPS_DIFF="${STEPS_DIFF:-1000000}"

banner() { echo ""; echo "════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════"; }

_latest_ckpt() { ls -t "$CKPT_DIR"/$1/"$2" 2>/dev/null | head -1 || true; }
_resolve_ckpt() {
    local found
    found=$(_latest_ckpt "$1" "best.pth")
    [ -n "$found" ] || found=$(_latest_ckpt "$1" "final.pth")
    echo "$found"
}

TORCHRUN=(torchrun --standalone --nproc_per_node="$NUM_GPUS")

# ════════════════════════════════════════════════════════════
banner "Configuration"
# ════════════════════════════════════════════════════════════
echo "  GPUs              : $NUM_GPUS"
echo "  ImageVAE  batch   : $BATCH_VAE per GPU  →  $((BATCH_VAE * NUM_GPUS)) effective"
echo "  Diffusion batch   : $BATCH_DIFF per GPU  →  $((BATCH_DIFF * NUM_GPUS)) effective"
echo "  ImageVAE  epochs  : $EPOCHS_VAE  (${EPOCHS_VAE_BASE} × ${NUM_GPUS} GPUs)"
echo "  Diffusion steps   : $STEPS_DIFF"
echo "  Val cases         : VAE $NUM_VAL_VAE  |  diffusion $NUM_VAL_DIFF"
echo "  Keep              : $KEEP_CKPTS checkpoints  |  $KEEP_VIS visualisations"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# ════════════════════════════════════════════════════════════
banner "Stage 1 — Retrain ImageVAE (DDP)"
# ════════════════════════════════════════════════════════════
# The previous ImageVAE reached mean_ssim 0.566 and was still climbing when
# the epoch budget ran out — it is the weakest link in the pipeline, so it
# gets retrained first. Retraining it invalidates any diffusion checkpoint
# trained against the old latent space.
if [ "${SKIP_VAE:-0}" = "1" ]; then
    echo "[1] SKIP_VAE=1 — skipping"
    IMAGE_VAE_CKPT=$(_resolve_ckpt "image_vae_*")
else
    "${TORCHRUN[@]}" -m training.train_image_vae_ddp \
        --data_root        "$DATA_PROC" \
        --splits_dir       "$SPLITS_DIR" \
        --checkpoint_dir   "$CKPT_DIR" \
        --crop_size        128 \
        --encoder_channels 64,128,256 \
        --num_epochs       "$EPOCHS_VAE" \
        --batch_size       "$BATCH_VAE" \
        --num_workers      "$NUM_WORKERS" \
        --val_every        200 \
        --num_val_cases    "$NUM_VAL_VAE" \
        --keep_checkpoints "$KEEP_CKPTS" \
        --early_stop_patience 25 \
        --device           cuda
    IMAGE_VAE_CKPT=$(_resolve_ckpt "image_vae_*")
fi

if [ -z "$IMAGE_VAE_CKPT" ]; then
    echo "[1] ERROR: no ImageVAE checkpoint found"
    exit 1
fi
echo "[1] ImageVAE → $IMAGE_VAE_CKPT"

# ════════════════════════════════════════════════════════════
banner "Stage 2 — Retrain latent diffusion UNet (DDP)"
# ════════════════════════════════════════════════════════════
MASK_VAE_CKPT=$(_resolve_ckpt "mask_vae_*")
if [ -z "$MASK_VAE_CKPT" ]; then
    echo "[2] ERROR: no MaskVAE checkpoint found — expected an existing one"
    exit 1
fi
echo "[2] MaskVAE  → $MASK_VAE_CKPT"

# Must be a fresh run: the UNet conditions on ImageVAE latents, so a new
# ImageVAE means the old diffusion weights are meaningless. Not a resume.
"${TORCHRUN[@]}" -m training.train_latent_diffusion_ddp \
    --latent_dir     "$LATENT_DIR" \
    --image_data     "$DATA_PROC" \
    --image_vae_ckpt "$IMAGE_VAE_CKPT" \
    --mask_vae_ckpt  "$MASK_VAE_CKPT" \
    --splits_dir     "$SPLITS_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --num_steps      "$STEPS_DIFF" \
    --batch_size     "$BATCH_DIFF" \
    --num_workers    "$NUM_WORKERS" \
    --num_val_cases  "$NUM_VAL_DIFF" \
    --keep_checkpoints "$KEEP_CKPTS" \
    --keep_vis       "$KEEP_VIS" \
    --device         cuda

DIFF_CKPT=$(_resolve_ckpt "latent_diffusion_*")

# ════════════════════════════════════════════════════════════
banner "Done"
# ════════════════════════════════════════════════════════════
echo "  ImageVAE  : $IMAGE_VAE_CKPT"
echo "  MaskVAE   : $MASK_VAE_CKPT"
echo "  Diffusion : $DIFF_CKPT"
echo ""
echo "Next:  bash run_a100.sh          # Steps 8-9: eval + package"
echo "   or: bash collect_results.sh   # package only"
