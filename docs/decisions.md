# Architecture & training decisions log

## 2026-06-29 — Diffusion U-Net + Stage 3 training script built and verified end-to-end

- **Built:** `models/diffusion/` (sinusoidal time embedding, `TimestepResBlock3D`, `UNet3D` operating on the mask latent grid with the Uni-Encoder's conditioning feature concatenated at the input, `GaussianDiffusionSchedule` for forward/reverse DDPM), `data/brats_dataset.py` (loader matching UniME's actual on-disk format: `root/vol/{name}_vol.npy` (H,W,D,4) float32, `root/seg/{name}_seg.npy` (H,W,D) uint8, plus the 15-combination modality masking used during this stage's training), and `training/train_diffusion_frozen.py` (Stage 3: trains the U-Net with both the Uni-Encoder and mask VAE frozen).
- **Mask latent target uses the VAE encoder's mean (`mu`), not a reparameterized sample.** The paper's VQ-VAE produces a deterministic quantized latent with no separate stochastic step; using `mu` from our continuous VAE is the closest analogue, and avoids stacking VAE-sampling noise on top of diffusion noise during training.
- **Bug caught while writing the up-path of `UNet3D`:** initial draft re-concatenated the same skip tensor at every block within a level (with dead `if/else` code doing the same thing either branch). Fixed to the standard pattern: concat once per level, then process normally.
- **Reverse-process formula:** implemented the standard DDPM posterior (Ho et al.) rather than literally transcribing the paper's Eq. `reverse-modify`, which appears to have two algebra-inconsistent terms relative to its own cited derivation (extra sqrt on the `ε_θ` coefficient; `γ_t` instead of `√β_t` as the injected-noise scale). Forward process and loss (Eqs. `forward`/`nice-property`/`dm-loss`) match the paper exactly — confirmed by direct comparison, not just by similarity.
- **Confirmed working via full end-to-end run, not just unit shape checks:** generated a synthetic BraTS-format dataset + fake Stage 1/2 checkpoints, ran `train_diffusion_frozen.py` for real steps (loss computed, backward, optimizer step, checkpoint+metadata saved, `EXPERIMENTS.md` auto-appended), then verified `--resume` correctly picks up mid-training from a saved checkpoint and continues to the same target step count.
- **Caught and fixed a real CLI gap during this test:** `mask_vae_base_channels` wasn't exposed as a script argument, so a mask VAE trained with non-default `base_channels` would silently mismatch when this script tried to load it. Added as a required-to-match CLI flag.
- **Param running total:** Uni-Encoder Tiny (54.3M) + Mask VAE (22.8M) + Diffusion U-Net (31.3M, full scale) ≈ **108.4M**, all confirmed by instantiation, not estimated.


## 2026-06-29 — Mask VAE built and verified against Uni-Encoder grid; Uni-Encoder ported

- **Built:** `models/uniencoder/` (full port of the Uni-Encoder backbone + Stage-1 masking from `Hooorace-S/UniME`, CNN multi-encoder/fusion network deliberately excluded — not needed for diffusion conditioning) and `models/mask_vae/` (continuous 3D VAE, GroupNorm/SiLU ResBlocks, downsample/upsample via strided conv + nearest-interp, WCE reconstruction loss + small-weight KL term).
- **Bug caught during smoke-testing:** initial `MaskEncoder3D`/`MaskDecoder3D` only downsampled/upsampled *between* levels (skipping the last), so 3 `channel_multipliers` levels gave 4x spatial downsampling, not the 8x `MaskVAEConfig.downsample_factor` claimed. Fixed by downsampling/upsampling after every level. Caught because the smoke test asserted the latent's spatial shape against the Uni-Encoder's conditioning grid shape and they didn't match (8³ vs 4³) — this is exactly why shape-assertion smoke tests are worth keeping even for "obviously correct" code.
- **Confirmed working, not just estimated:** at full scale (96³ crop, base_channels=64), `MaskVAE3D` is **22.79M params**, latent shape `(8, 12, 12, 12)` — matches `UniEncoderConditioning`'s `(embed_dim, 12, 12, 12)` output exactly, so the planned channel-wise concat in the diffusion U-Net needs no resampling.
- **Param running total (no diffusion U-Net yet):** Uni-Encoder Tiny (54.3M, confirmed) + Mask VAE (22.8M, confirmed) ≈ 77M.


Running log of decisions made and why. Append new entries at the top (most recent first). This is for decisions and rationale — for routine training observations ("OOM at batch=4, dropped to batch=2"), use the per-run `notes.md` under `logs/runs/<run_id>/` instead.

---

## 2026-06-29 — Image-side VAE replaced by Uni-Encoder, not the mask VAE

- **Decision:** Swap MedSegLatDiff's image VQ-VAE (`E_X`) for a pretrained Uni-Encoder (from `Hooorace-S/UniME`). Keep the mask side as a normal (non-VQ) 3D VAE, trained independently exactly as in MedSegLatDiff.
- **Why:** `E_X`'s output is only used as a conditioning signal (concatenated with the noisy mask latent, never itself denoised or decoded), so no decoder/codebook needs to be reproduced — this is the structurally easier of the two possible swaps. The Uni-Encoder's masked self-supervised pretraining is specifically designed to produce coherent representations under missing modalities, which a plain VQ-VAE has no mechanism for.
- **Constraint this introduces:** mask VAE's total downsampling factor must match the Uni-Encoder's patch size (8×8×8 on a 96³ crop → both produce 12³ grids) so the channel-wise concat in the diffusion U-Net works without resampling.

## 2026-06-29 — Sequential training, not joint-from-scratch

- **Decision:** Train in 4 sequential stages: (1) Uni-Encoder masked self-supervised pretrain, (2) mask VAE pretrain, (3) diffusion U-Net with Uni-Encoder frozen, (4) joint fine-tune with layer-wise LR decay (LLRD, ω≈0.75 per UniME's own ablation).
- **Why:** Joint training from random init would let diffusion-loss gradients corrupt the Uni-Encoder's pretrained semantics before the U-Net has learned anything useful to condition on. UniME's own ablation found `ω=0.75` LLRD beat both full freezing and `ω=1.00` (uniform/naive joint fine-tuning).
- **Implication:** budget for 4 separate training runs, each resumable independently — not one big end-to-end job.

## 2026-06-29 — Starting scale: UniEncoderTiny / Nano, not Original

- **Decision:** Use `UniEncoderTiny` (54.3M params) or `UniEncoderNano` (22.8M) rather than the paper's Original/Base scale (147M each).
- **Why:** Compute constraint — likely training on Kaggle free tier (16GB GPU, ~30 GPU-hrs/week, 12h/session cap). The Original-scale UniME Stage 2 alone needed ~23.8 GiB on a dedicated RTX 6000 in the paper; adding a 3D diffusion U-Net and 3D mask VAE on top makes the larger scales impractical on free-tier hardware.
- **Open question:** whether Tiny/Nano retains enough representational capacity for the diffusion conditioning role — UniME's own ablation (Table 4) only tested Small/Base/Large for the *segmentation* task, not Tiny/Nano, and not in a diffusion-conditioning role. Worth re-validating once a baseline is running.

## 2026-06-29 — No pretrained checkpoint available upstream

- **Finding:** `Hooorace-S/UniME` has no GitHub Releases, no external weight links (HF/Drive/Zenodo), and its own `.gitignore` excludes `log_pretrain/` (where `ema_best_checkpoint.pth` is written). Stage 1 pretraining must be run from scratch on our own compute unless the authors share a checkpoint directly.
- **Action taken:** emailed first author (Peibo Song, confirmed via CVPR poster) asking if a checkpoint is available, specifying Tiny/Nano scale. Treated as a parallel, non-blocking track — own Stage 1 pretraining pipeline proceeds regardless of reply.
