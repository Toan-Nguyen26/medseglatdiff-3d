Master index of training runs. One line per run, appended automatically via `RunLogger.append_to_experiments_index(...)`. Keep this file committed to git — it's the cheap, scannable history; the heavy artifacts (checkpoints, full metric CSVs) live ungit-tracked under `logs/runs/<run_id>/` and should be synced to external storage (Kaggle Datasets / cloud bucket / HF Hub).

| Date | Run ID | Stage | Summary |
|---|---|---|---|
| 2026-06-29 | stage3_diffusion_frozen_20260629_111439 | stage3_diffusion_frozen | Tiny diffusion U-Net, 4 steps, final_loss=1.0130 |
| 2026-06-29 | stage3_diffusion_frozen_20260629_111507 | stage3_diffusion_frozen | Tiny diffusion U-Net, 6 steps, final_loss=1.0135 |
| 2026-07-01 | joint_20260701_093357 | joint | 4-encoder joint, 4 steps, embed_dim=64, final_loss=0.9815 |
