# MedSegLatDiff-3D

3D latent diffusion model for brain tumor segmentation with missing modalities, combining:

- **MedSegLatDiff**'s one-to-many diffusion segmentation framework (diffusion model operating in a VAE latent space, conditioned on the input image, generating multiple plausible masks → confidence map).
- A **4-encoder CNN sum-fusion** conditioning scheme in place of the image-side VAE encoder: one small 3D CNN per MRI modality (FLAIR/T1ce/T1/T2), fused by element-wise sum. Missing modalities are zeroed before encoding, so a missing encoder contributes ~0 to the sum — robustness comes from training on all 15 non-empty modality combinations, not from a pretraining stage.

This replaces an earlier design that used **UniME's Uni-Encoder** for the same role (see Legacy section below) — that code is still in-tree but no longer part of the active pipeline.

See [docs/decisions.md](docs/decisions.md) for the architecture decisions made so far and why (note: not yet updated for the encoder pivot below).

## Status

Active pipeline is the simplified 2-stage one. `training/train_joint.py` (Stage 2) has been smoke-tested end-to-end on synthetic data; it has not yet been run on real BraTS2023 data. `train_mask_vae.py` (Stage 1) has not been written yet — the `models/mask_vae/` module exists but has no training script, which currently blocks running Stage 2 on real data.

## Pipeline (2 sequential training stages)

1. `training/train_mask_vae.py` **(not yet implemented)** — 3D mask VAE (`models/mask_vae/`), independent of everything else. Downsampling factor must match the encoders' (8× per axis, 96³ → 12³).
2. `training/train_joint.py` — 4 modality-specific CNN encoders (`models/multiencoder/`) and the diffusion U-Net (`models/diffusion/`) trained jointly from scratch, with the mask VAE frozen. No separate encoder-pretraining step: the diffusion loss backpropagates through the encoders directly, and `sample_modality_mask()` draws a random modality-availability combination every step so the encoders learn to work from any subset of modalities.

## Evaluation

Planned: `eval/eval_missing_modality.py` (not yet implemented — `eval/` is currently a stub) would follow UniME's protocol of evaluating one model across all 15 non-empty modality-availability combinations, reporting DSC/HD95 for WT/TC/ET.

## Legacy: Uni-Encoder pipeline (superseded)

`models/uniencoder/` and `training/train_diffusion_frozen.py` implement the original 4-stage design — Uni-Encoder masked self-supervised pretraining, mask VAE, diffusion U-Net with the Uni-Encoder frozen, then joint fine-tune with layer-wise LR decay. It's kept in-tree for reference (and because no pretrained Uni-Encoder checkpoint was ever obtained upstream — see decisions log) but is not used by the current pipeline above.

## Logging

Every run uses `utils/run_logger.RunLogger`:
- Per-run metrics CSV, notes, and config snapshot under `logs/runs/<run_id>/` (gitignored — sync to external storage).
- Checkpoint sidecar metadata (`*.meta.json`) recording stage, parent checkpoint, git commit, config, seed — needed to track lineage across interrupted Kaggle sessions.
- A one-line summary appended to the git-tracked [logs/EXPERIMENTS.md](logs/EXPERIMENTS.md) master index.

See `training/_example_usage.py` for the intended usage pattern.

## Attribution

Legacy Uni-Encoder code (`models/uniencoder/`) adapted from [Hooorace-S/UniME](https://github.com/Hooorace-S/UniME) (official implementation of *Uni-Encoder Meets Multi-Encoders*, CVPR). No pretrained weights are released upstream as of writing — see decisions log.

MedSegLatDiff framework based on *Diffusion Model in Latent Space for Medical Image Segmentation Task* (FJCAI 2026).
