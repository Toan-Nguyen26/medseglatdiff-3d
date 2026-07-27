"""
Multi-GPU (DDP) ImageVAE training.

Same model, loss, metrics and checkpoint format as train_image_vae.py —
every helper is imported from it, so there is exactly one implementation of
the VAE loss, the SSIM/PSNR metrics and the reconstruction visualisation.
Only the training loop is reimplemented here, because that is the part DDP
actually changes.

Launch with torchrun:

    torchrun --nproc_per_node=4 -m training.train_image_vae_ddp \\
        --data_root      data/brats_roi128 \\
        --splits_dir     splits/brats_roi128_full \\
        --checkpoint_dir checkpoints \\
        --crop_size      128 \\
        --num_epochs     800 \\
        --batch_size     8 \\
        --num_val_cases  50

Running without torchrun falls back to single-process and behaves like the
original script.

Two things to know before scaling up:

  --num_epochs means less work per epoch here. DistributedSampler gives
  each rank 1/N of the data, so N GPUs make one epoch N× fewer optimizer
  steps. The startup banner prints the arithmetic; to match a single-GPU
  run's step count, multiply --num_epochs by the GPU count.

  --lr is scaled by world size (linear scaling rule) with a short warmup,
  since the effective batch grows N×. Disable with --no_lr_scale.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask
from models.multiencoder.encoders import ImageVAE, vae_loss
from utils.ddp import (
    apply_lr,
    ddp_cleanup,
    ddp_setup,
    print_ddp_banner,
    rank0_print,
    scale_lr,
    unwrap,
    warmup_factor,
)
from utils.early_stopping import EarlyStopping
from utils.retention import prune_oldest
from utils.run_logger import RunLogger, new_run_id, set_seed

# Single source of truth for everything that is not the training loop.
from training.train_image_vae import (
    MODALITY_NAMES,
    STAGE,
    apply_patch_mask,
    parse_args as _base_parse_args,
    run_val_metrics,
    save_recon_visualisation,
)


# ---------------------------------------------------------------------------
# Args — base args plus the DDP-specific ones
# ---------------------------------------------------------------------------

def parse_args():
    """
    Pull the DDP-only flags out of argv, then hand the rest to the original
    parser untouched — so the two scripts can never drift on shared flags.
    """
    import argparse

    ddp = argparse.ArgumentParser(add_help=False)
    ddp.add_argument("--no_lr_scale", action="store_true",
                     help="Do not multiply --lr by world size.")
    ddp.add_argument("--warmup_steps", type=int, default=500,
                     help="Linear LR warmup steps; needed when LR is scaled up.")
    ddp_args, remaining = ddp.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = _base_parse_args()
    args.no_lr_scale  = ddp_args.no_lr_scale
    args.warmup_steps = ddp_args.warmup_steps
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = ddp_setup()
    distributed = world_size > 1

    # Offset the seed per rank so modality-dropout draws differ across ranks,
    # while staying reproducible.
    set_seed(args.seed + rank)

    splits_dir = args.splits_dir if args.splits_dir is not None else args.data_root

    # ── Model ────────────────────────────────────────────────────────────────
    encoder_channels = tuple(int(c) for c in args.encoder_channels.split(","))
    vae = ImageVAE(
        in_channels=4,
        channels=encoder_channels,
        num_res_units=args.encoder_num_res_units,
    ).to(device)

    lr_used = scale_lr(args.lr, world_size, enabled=not args.no_lr_scale)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr_used,
                                  weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        vae.load_state_dict(ckpt["vae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        rank0_print(rank, f"Resumed from step {start_step}")

    if distributed:
        vae = DDP(vae, device_ids=[local_rank] if device.type == "cuda" else None)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.split_file),
        crop_size=args.crop_size,
        volume_only=True,
    )

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Validation and visualisation run on rank 0 only, so only rank 0 loads them.
    val_cases: list[torch.Tensor] = []
    test_cases: list[torch.Tensor] = []
    if rank == 0:
        val_split_path = os.path.join(splits_dir, args.val_split_file)
        if os.path.exists(val_split_path):
            val_ds = BraTSDataset(root=args.data_root, split_file=val_split_path,
                                  crop_size=args.crop_size, random_crop=False,
                                  volume_only=True)
            for i in range(min(args.num_val_cases, len(val_ds))):
                vol, _ = val_ds[i]
                val_cases.append(vol.unsqueeze(0))
            print(f"Val cases: {len(val_cases)}")
        else:
            print(f"  [warn] val split not found at {val_split_path}")

        test_split_path = os.path.join(splits_dir, args.test_split_file)
        if os.path.exists(test_split_path):
            test_ds = BraTSDataset(root=args.data_root, split_file=test_split_path,
                                   crop_size=args.crop_size, random_crop=False,
                                   volume_only=True)
            for i in range(min(args.num_test_vis_cases, len(test_ds))):
                vol, _ = test_ds[i]
                test_cases.append(vol.unsqueeze(0))

    # ── Run setup (rank 0 owns the run directory) ────────────────────────────
    logger = None
    checkpoint_dir = vis_dir = None
    if rank == 0:
        run_id = new_run_id(STAGE)
        logger = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
        logger.note(f"DDP world_size={world_size} effective_batch="
                    f"{args.batch_size * world_size} lr={lr_used:.2e}")
        checkpoint_dir = Path(args.checkpoint_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Alongside the checkpoints, so collect_results.sh picks these up.
        vis_dir = checkpoint_dir / "visualisations"
        vis_dir.mkdir(exist_ok=True)

    steps_per_epoch = len(loader)
    total_steps     = args.num_epochs * steps_per_epoch
    start_epoch     = start_step // max(1, steps_per_epoch)

    print_ddp_banner(
        rank, world_size,
        per_gpu_batch=args.batch_size,
        lr_base=args.lr,
        lr_used=lr_used,
        steps_per_epoch=steps_per_epoch,
        num_epochs=args.num_epochs,
    )

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    ) if (args.early_stop_patience > 0 and rank == 0) else None

    best_mean_ssim = -float("inf")
    step = start_step
    stop_early = False

    pbar = tqdm(total=total_steps, initial=start_step, desc=f"[{STAGE}]",
                unit="step", disable=(rank != 0))

    for epoch in range(start_epoch, args.num_epochs):
        if stop_early:
            break
        if sampler is not None:
            # Without this every epoch draws the identical shuffle.
            sampler.set_epoch(epoch)

        vae.train()
        for volume, _ in loader:
            volume = volume.to(device, non_blocking=True)

            # LR warmup — matters because the scaled LR starts high.
            if args.warmup_steps > 0 and step < args.warmup_steps:
                apply_lr(optimizer, lr_used * warmup_factor(step, args.warmup_steps))

            modality_mask = sample_modality_mask(volume.shape[0]).to(device)
            volume_masked = apply_modality_mask(volume, modality_mask)

            if args.patch_mask_ratio > 0.0:
                encoder_input, patch_mask = apply_patch_mask(
                    volume_masked, args.patch_size, args.patch_mask_ratio
                )
                masked_region = (1 - patch_mask)
                recon_target  = volume_masked * masked_region
            else:
                encoder_input = volume_masked
                recon_target  = volume_masked
                masked_region = None

            recon, mu, logvar = vae(encoder_input)
            if masked_region is not None:
                recon = recon * masked_region

            total, r_loss, k_loss = vae_loss(recon, recon_target, mu, logvar,
                                             beta=args.vae_beta)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            if rank == 0:
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{args.num_epochs}",
                    loss=f"{total.item():.4f}",
                    recon=f"{r_loss.item():.4f}",
                )

                if step % args.log_every == 0:
                    logger.log_metrics(
                        step=step, csv="train",
                        epoch=epoch + 1,
                        total_loss=total.item(),
                        recon_loss=r_loss.item(),
                        kl_loss=k_loss.item(),
                    )

                if step % args.val_every == 0 and step > start_step and val_cases:
                    # run_val_metrics expects a bare model, not the DDP wrapper.
                    metrics = run_val_metrics(unwrap(vae), val_cases, device)
                    logger.log_metrics(step=step, csv="val", **metrics)

                    if metrics["mean_ssim"] > best_mean_ssim:
                        best_mean_ssim = metrics["mean_ssim"]
                        torch.save({
                            "vae_state_dict":       unwrap(vae).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step":   step,
                            "epoch":  epoch,
                            "config": vars(args),
                            "best_mean_ssim": best_mean_ssim,
                        }, checkpoint_dir / "best.pth")

                    es_status = ""
                    if early_stopper is not None:
                        if early_stopper.step(metrics["mean_ssim"], step):
                            tqdm.write(f"  [early stop] no SSIM improvement for "
                                       f"{early_stopper.patience} checks — stopping.")
                            tqdm.write(f"  [early stop] best mean_ssim="
                                       f"{best_mean_ssim:.4f} → best.pth")
                            stop_early = True
                        es_status = f"  [{early_stopper.status}]"

                    tqdm.write(
                        f"  [val]  SSIM — "
                        + "  ".join(f"{m}={metrics[f'{m}_ssim']:.3f}" for m in MODALITY_NAMES)
                        + f"  mean={metrics['mean_ssim']:.3f}" + es_status
                    )
                    vae.train()

                if step % args.ckpt_every == 0 and step > start_step:
                    torch.save({
                        "vae_state_dict":       unwrap(vae).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step":   step,
                        "epoch":  epoch,
                        "config": vars(args),
                    }, checkpoint_dir / f"step_{step}.pth")
                    prune_oldest(checkpoint_dir, "step_*.pth", args.keep_checkpoints)

                if args.vis_every > 0 and step % args.vis_every == 0 \
                        and step > start_step and val_cases:
                    # First few val cases only — the grid is one row per case.
                    save_recon_visualisation(
                        unwrap(vae), val_cases[:args.num_test_vis_cases], device,
                        vis_dir / f"recon_step_{step:07d}.png",
                    )
                    prune_oldest(vis_dir, "recon_step_*.png", args.keep_vis)
                    vae.train()

            step += 1
            if rank == 0:
                pbar.update(1)

            # Every rank must learn about the stop, or the others hang at the
            # next collective. Broadcast rank 0's decision.
            if distributed:
                flag = torch.tensor([1 if stop_early else 0], device=device)
                torch.distributed.broadcast(flag, src=0)
                stop_early = bool(flag.item())
            if stop_early:
                break

    pbar.close()

    # ── Final artefacts (rank 0) ─────────────────────────────────────────────
    if rank == 0:
        final_path = checkpoint_dir / "final.pth"
        torch.save({
            "vae_state_dict":       unwrap(vae).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step":   step,
            "epoch":  args.num_epochs,
            "config": vars(args),
        }, final_path)
        print(f"\nDone. Checkpoint saved → {final_path}")

        best_path = checkpoint_dir / "best.pth"
        if best_path.exists():
            print(f"Best checkpoint (mean_ssim={best_mean_ssim:.4f}) → {best_path}")
            print("Use best.pth downstream — final.pth is the resume point.")

        if test_cases:
            vis_path = vis_dir / "test_recon_final.png"
            save_recon_visualisation(unwrap(vae), test_cases, device, vis_path)
            print(f"Test reconstruction grid saved → {vis_path}")

        logger.append_to_experiments_index(
            f"Image VAE (DDP×{world_size}), {args.num_epochs} epochs ({step} steps), "
            f"channels={args.encoder_channels}, eff_batch={args.batch_size * world_size}, "
            f"best_mean_ssim={best_mean_ssim:.4f}"
        )

    ddp_cleanup()


if __name__ == "__main__":
    main()
