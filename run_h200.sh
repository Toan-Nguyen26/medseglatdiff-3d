#!/bin/bash
# ============================================================
#  BraTS latent-diffusion pipeline  —  H200 (141 GB HBM3e)
#
#  Identical pipeline to run_a100.sh but with larger batches
#  to take advantage of the extra VRAM.
#
#  Usage:
#    bash run_h200.sh
# ============================================================
export BATCH_VAE=8     # A100 default: 4
export BATCH_DIFF=64   # A100 default: 32

exec bash "$(dirname "$0")/run_a100.sh" "$@"
