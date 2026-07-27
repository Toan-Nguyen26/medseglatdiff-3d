"""
Multi-GPU (DDP) latent-diffusion training.

Same model, loss, sampler, validation and checkpoint format as
train_latent_diffusion.py — every helper (ddim_sample, run_val, save_vis,
the VAE loaders, the dataset) is imported from it, so there is exactly one
implementation of each. Only the training loop is reimplemented, because
that is the part DDP changes.

Launch with torchrun:

    torchrun --nproc_per_node=4 -m training.train_latent_diffusion_ddp \\
        --latent_dir     data/brats_latents \\
        --image_data     data/brats_roi128 \\
        --image_vae_ckpt checkpoints/image_vae_.../best.pth \\
        --mask_vae_ckpt  checkpoints/mask_vae_.../best.pth \\
        --splits_dir     splits/brats_roi128_full \\
        --num_steps      1000000 \\
        --batch_size     32 \\
        --num_val_cases  30

Running without torchrun falls back to single-process.

Notes:

  --num_steps counts optimizer steps, which are the same on every rank —
  so unlike the epoch-based VAE trainer, the step budget means the same
  thing regardless of GPU count. What changes is that each step now covers
  world_size × batch_size samples, so a given --num_steps sees N× more data.

  --lr is scaled by world size (linear scaling rule) with warmup, since the
  effective batch grows N×. Disable with --no_lr_scale.

  Validation and visualisation run on rank 0 only. DDIM sampling is
  sequential and would not benefit from sharding at this scale.

This file deliberately does not touch train_latent_diffusion.py. Known
issues in that file (the float16 x_t in ddim_sample, the GradScaler left
enabled with autocast commented out) are inherited here unchanged — they
are owned by whoever made those changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import apply_modality_mask, sample_modality_mask
from models.diffusion.config import DiffusionConfig
from models.diffusion.schedule import GaussianDiffusionSchedule, diffusion_loss
from models.diffusion.unet3d import UNet3D
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
from utils.retention import prune_oldest
from utils.run_logger import RunLogger, new_run_id, set_seed

# Single source of truth for everything that is not the training loop.
from training.train_latent_diffusion import (
    STAGE,
    LatentDiffusionDataset,
    _load_image_vae,
    _load_mask_vae,
    _read_names,
    parse_args as _base_parse_args,
    run_val,
    save_vis,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    """Strip the DDP-only flags, then hand the rest to the original parser."""
    import argparse

    ddp = argparse.ArgumentParser(add_help=False)
    ddp.add_argument("--no_lr_scale", action="store_true",
                     help="Do not multiply --lr by world size.")
    ddp.add_argument("--warmup_steps", type=int, default=1000,
                     help="Linear LR warmup steps; needed when LR is scaled up.")
    # Retention lives here rather than in the base parser because
    # train_latent_diffusion.py is not ours to modify.
    ddp.add_argument("--keep_checkpoints", type=int, default=20,
                     help="Rolling window of periodic step_*.pth files to keep. "
                          "best.pth and final.pth are never pruned. 0 disables.")
    ddp.add_argument("--keep_vis", type=int, default=20,
                     help="Rolling window of validation PNGs to keep. 0 disables.")
    ddp_args, remaining = ddp.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = _base_parse_args()
    args.no_lr_scale      = ddp_args.no_lr_scale
    args.warmup_steps     = ddp_args.warmup_steps
    args.keep_checkpoints = ddp_args.keep_checkpoints
    args.keep_vis         = ddp_args.keep_vis
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = ddp_setup()
    distributed = world_size > 1

    # Offset per rank so modality dropout and timestep draws differ across
    # ranks, while staying reproducible.
    set_seed(args.seed + rank)

    latent_dir = Path(args.latent_dir)
    vol_dir    = Path(args.image_data) / "vol"
    seg_dir    = Path(args.image_data) / "seg"
    splits_dir = Path(args.splits_dir) if args.splits_dir else latent_dir

    # ── Frozen VAEs (every rank needs its own copy for on-the-fly encoding) ──
    rank0_print(rank, "Loading ImageVAE …")
    image_vae = _load_image_vae(args.image_vae_ckpt, device)
    embed_dim = image_vae.embed_dim

    rank0_print(rank, "Loading MaskVAE …")
    mask_vae, subregion = _load_mask_vae(args.mask_vae_ckpt, device)
    rank0_print(rank, f"  embed_dim={embed_dim}  "
                      f"latent_channels={mask_vae.latent_channels}  "
                      f"subregion={subregion}")

    # ── UNet + schedule ──────────────────────────────────────────────────────
    channel_mults = tuple(int(m) for m in args.channel_mults.split(","))
    cfg = DiffusionConfig(
        latent_channels          = args.latent_channels,
        condition_channels       = embed_dim,
        cond_proj_channels       = args.cond_proj_channels,
        base_channels            = args.base_channels,
        channel_multipliers      = channel_mults,
        num_res_blocks_per_level = args.num_res_blocks,
        num_timesteps            = args.num_timesteps,
    )
    unet     = UNet3D(cfg).to(device)
    schedule = GaussianDiffusionSchedule(num_timesteps=args.num_timesteps)

    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    rank0_print(rank, f"UNet3D: {n_params:.1f}M params")

    lr_used   = scale_lr(args.lr, world_size, enabled=not args.no_lr_scale)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr_used,
                                  weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume (before DDP wrap, so keys have no module. prefix) ─────────────
    step = 0
    best_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step      = ckpt.get("step", 0)
        best_dice = ckpt.get("best_dice", 0.0)
        rank0_print(rank, f"Resumed from step {step}, best_dice={best_dice:.4f}")

    if distributed:
        unet = DDP(unet, device_ids=[local_rank] if device.type == "cuda" else None)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_names = _read_names(splits_dir / args.split_file)
    val_names   = _read_names(splits_dir / args.val_split_file)

    train_ds = LatentDiffusionDataset(latent_dir / "z_mask", vol_dir, train_names)
    sampler  = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── Run setup (rank 0 owns the run directory) ────────────────────────────
    logger = None
    ckpt_dir = vis_dir = None
    if rank == 0:
        run_id   = new_run_id(STAGE)
        ckpt_dir = Path(args.checkpoint_dir) / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        vis_dir  = ckpt_dir / "visualisations"

        logger = RunLogger(run_id, STAGE, config=vars(args))
        logger.note(
            f"DDP world_size={world_size} effective_batch={args.batch_size * world_size} "
            f"lr={lr_used:.2e} | ImageVAE {args.image_vae_ckpt} | "
            f"MaskVAE {args.mask_vae_ckpt} | unet={n_params:.1f}M"
        )
        print(f"\nTrain: {len(train_names)} cases  Val: {len(val_names)} cases")
        print(f"Checkpoint dir: {ckpt_dir}")

    print_ddp_banner(
        rank, world_size,
        per_gpu_batch=args.batch_size,
        lr_base=args.lr,
        lr_used=lr_used,
    )

    def _ckpt_payload() -> dict:
        return {
            "unet_state_dict":      unwrap(unet).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step":                 step,
            "best_dice":            best_dice,
            "config":               vars(args),
            "image_vae_ckpt":       args.image_vae_ckpt,
            "mask_vae_ckpt":        args.mask_vae_ckpt,
        }

    # ── Training loop ────────────────────────────────────────────────────────
    dl_iter = iter(train_dl)
    epoch   = 0

    pbar = tqdm(total=args.num_steps, initial=step, desc="training",
                disable=(rank != 0))

    while step < args.num_steps:
        try:
            z_mask, vol = next(dl_iter)
        except StopIteration:
            # New pass over the data. set_epoch keeps ranks from replaying
            # the identical shuffle every time.
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            dl_iter = iter(train_dl)
            z_mask, vol = next(dl_iter)

        z_mask = z_mask.to(device, non_blocking=True)
        vol    = vol.to(device, non_blocking=True)

        if args.warmup_steps > 0 and step < args.warmup_steps:
            apply_lr(optimizer, lr_used * warmup_factor(step, args.warmup_steps))

        mod_mask   = sample_modality_mask(vol.shape[0]).to(device)
        vol_masked = apply_modality_mask(vol, mod_mask)

        unet.train()
        with torch.no_grad():
            mu_img, _ = image_vae.encode(vol_masked)

        # Pass the wrapped module — the backward pass through it is what
        # triggers the gradient all-reduce.
        loss = diffusion_loss(unet, schedule, z_mask, mu_img)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if rank == 0:
            pbar.update(1)

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                logger.log_metrics(step, csv="train", loss=loss.item())

            # ── Validation (rank 0 only; DDIM sampling is sequential) ───────
            if step % args.val_every == 0:
                metrics = run_val(
                    unwrap(unet), image_vae, mask_vae, schedule,
                    val_names, vol_dir, seg_dir,
                    device=device,
                    subregion=subregion,
                    n_inf_steps=args.num_inference_steps,
                    n_cases=args.num_val_cases,
                )
                logger.log_metrics(step, csv="val", **metrics)

                mean_d = metrics["mean_dice"]
                print(
                    f"\n[step {step}] "
                    f"WT={metrics['wt_dice']:.3f}  "
                    f"TC={metrics['tc_dice']:.3f}  "
                    f"ET={metrics['et_dice']:.3f}  "
                    f"mean={mean_d:.3f}"
                    f"{'  ★ best' if mean_d > best_dice else ''}"
                )

                if mean_d > best_dice:
                    best_dice = mean_d
                    torch.save(_ckpt_payload(), ckpt_dir / "best.pth")

                save_vis(
                    unwrap(unet), image_vae, mask_vae, schedule,
                    val_names, vol_dir, seg_dir, vis_dir, step,
                    device=device, subregion=subregion,
                    n_inf_steps=args.num_inference_steps,
                )
                prune_oldest(vis_dir, "step_*.png", args.keep_vis)
                unet.train()

            if step % args.ckpt_every == 0:
                torch.save(_ckpt_payload(), ckpt_dir / f"step_{step}.pth")
                prune_oldest(ckpt_dir, "step_*.pth", args.keep_checkpoints)

    pbar.close()

    # ── Final artefacts (rank 0) ─────────────────────────────────────────────
    if rank == 0:
        torch.save(_ckpt_payload(), ckpt_dir / "final.pth")
        logger.append_to_experiments_index(
            f"latent diffusion (DDP×{world_size}), {step} steps, "
            f"best_dice={best_dice:.4f}, unet={n_params:.1f}M, "
            f"eff_batch={args.batch_size * world_size}, subregion={subregion}"
        )
        print(f"\nDone. best_dice={best_dice:.4f}  saved to {ckpt_dir}")
        print("Use best.pth downstream — final.pth is the resume point.")

    ddp_cleanup()


if __name__ == "__main__":
    main()
