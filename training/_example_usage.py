"""
Not a real training script — shows the intended RunLogger usage pattern
that pretrain_uniencoder.py / train_mask_vae.py / etc. should follow.
"""

from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "stage1_uniencoder_pretrain"
SEED = 42

if __name__ == "__main__":
    set_seed(SEED)
    run_id = new_run_id(STAGE)
    logger = RunLogger(
        run_id=run_id,
        stage=STAGE,
        config={"scale": "UniEncoderTiny", "crop_shape": [96, 96, 96], "lr": 1e-5},
    )

    for step in range(3):
        train_loss = 1.0 / (step + 1)  # placeholder
        logger.log_metrics(step=step, train_loss=train_loss)

    logger.note("Resumed from Kaggle session 2; previous session hit the 12h wall clock.")

    checkpoint_path = f"checkpoints/{run_id}/last.pth"
    logger.save_checkpoint_meta(
        checkpoint_path,
        step=3,
        epoch=1,
        parent_checkpoint=None,
        seed=SEED,
    )

    logger.append_to_experiments_index("UniEncoderTiny, 3 steps (smoke test), train_loss=0.33")
