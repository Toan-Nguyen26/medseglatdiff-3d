"""
Inference script: load pretrained VAE + UNet, run N noise samples per test case,
compute segmentation metrics and uncertainty scores.

Metrics reported (per region: WT / TC / ET):
  Dice, IoU          — ensemble prediction vs GT
  GED                — Generalized Energy Distance (sample diversity vs GT)
  D_max              — max pairwise distance between samples (spread)
  Uncertainty (mean) — mean voxel variance across N samples

All results saved to eval_output/metrics.csv.

Model architecture (encoder channels, UNet base channels, etc.) is never
passed on the command line — it's read from the checkpoint's own embedded
"config" (or, for older checkpoints, the sibling logs/runs/<run_id>/config.yaml
written at training time). This is required, not best-effort: if neither is
found, the script errors out rather than guessing.

Usage:
    python3 -m eval.infer \\
        --data_root data/brats2023_processed \\
        --splits_dir splits/brats2023_20pct \\
        --seg_checkpoint checkpoints/pixel_test_.../final.pth \\
        --n_samples 5

    # Simulate missing modality (T2 missing)
    python3 -m eval.infer ... --modality_mask 1101 --n_samples 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.brats_dataset import (
    BraTSDataset, apply_modality_mask, sample_modality_mask,
)
from monai.networks.schedulers import DDIMScheduler

from models.multiencoder.encoders import ImageVAE, ImageEncoder, MaskVAE
from models.diffusion.unet3d import UNet3D
from models.diffusion.schedule import DiffusionConfig

MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
REGION_NAMES   = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--data_root",  required=True)
    p.add_argument("--splits_dir", default=None,
                   help="Folder with test.txt. Defaults to --data_root.")
    p.add_argument("--test_split_file", default="test.txt")
    p.add_argument("--num_cases",  type=int, default=4,
                   help="How many test cases to evaluate.")
    p.add_argument("--crop_size",  type=int, default=96)

    # Checkpoints — architecture is read from these, never from CLI flags
    p.add_argument("--seg_checkpoint", required=True,
                   help="train_pixel_test.py final.pth "
                        "(contains encoder_state_dict + unet_state_dict).")
    p.add_argument("--image_vae_checkpoint", default=None,
                   help="Optional separate pretrained VAE checkpoint. "
                        "If omitted, uses the encoder inside --seg_checkpoint.")
    p.add_argument("--mask_vae_checkpoint", default=None,
                   help="Optional MaskVAE checkpoint. Overrides the path stored in "
                        "--seg_checkpoint's config (useful for swapping VAEs).")

    # Inference
    p.add_argument("--n_samples", type=int, default=5,
                   help="Noise samples per case. Paper shows Dice stabilises at 5.")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM denoising steps per sample (50 ≈ 1000-step DDPM quality, much faster).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--modality_mask", default="all",
                   help="'all' | 'random' | 4-char binary e.g. '1101' (T2 missing).")

    # Output
    p.add_argument("--output_dir", default="eval_output")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_arch_config(ckpt: dict, ckpt_path: str) -> dict:
    """
    Architecture hyperparameters (encoder channels, UNet base channels, etc.)
    come only from the training run that produced this checkpoint — never
    from CLI flags, since a human re-typing them is exactly how they end up
    silently wrong and crash on load_state_dict. Preference order:
      1. config embedded directly in the checkpoint
      2. the sibling training run's logs/runs/<run_id>/config.yaml (older checkpoints)
    Raises if neither is found rather than falling back to a guess.
    """
    config = ckpt.get("config")
    if config is not None:
        return config

    run_id = Path(ckpt_path).resolve().parent.name
    log_config_path = Path(__file__).resolve().parents[1] / "logs" / "runs" / run_id / "config.yaml"
    if log_config_path.exists():
        import yaml
        print(f"Architecture config ← {log_config_path}")
        return yaml.safe_load(log_config_path.read_text())

    raise RuntimeError(
        f"Can't determine model architecture for {ckpt_path}: no embedded "
        f"config and no training log at {log_config_path}."
    )


def load_models(args: argparse.Namespace, device: torch.device):
    seg_ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=True)
    seg_config = load_arch_config(seg_ckpt, args.seg_checkpoint)
    num_classes = seg_config["num_classes"]

    if args.image_vae_checkpoint is not None:
        vae_ckpt = torch.load(args.image_vae_checkpoint, map_location=device, weights_only=True)
        vae_config = load_arch_config(vae_ckpt, args.image_vae_checkpoint)
        encoder_channels = tuple(int(c) for c in vae_config["encoder_channels"].split(","))
        embed_dim = encoder_channels[-1]
        encoder = ImageVAE(in_channels=4, channels=encoder_channels,
                           num_res_units=vae_config["encoder_num_res_units"]).to(device)
        encoder.load_state_dict(vae_ckpt["vae_state_dict"])
        print(f"Encoder  ← VAE checkpoint: {args.image_vae_checkpoint}")
    else:
        encoder_channels = tuple(int(c) for c in seg_config["encoder_channels"].split(","))
        embed_dim = encoder_channels[-1]

        # train_pixel_test.py always saves under "encoder_state_dict" regardless
        # of --use_vae, so detect the architecture from the actual key prefixes.
        encoder_state = seg_ckpt["encoder_state_dict"]
        is_vae = any(k.startswith("_backbone.") or k.startswith("mu_head.") or k.startswith("logvar_head.")
                     for k in encoder_state)
        if is_vae:
            encoder = ImageVAE(in_channels=4, channels=encoder_channels,
                               num_res_units=seg_config["encoder_num_res_units"]).to(device)
            print("Encoder  ← seg checkpoint (ImageVAE)")
        else:
            encoder = ImageEncoder(in_channels=4, embed_dim=embed_dim,
                                   num_res_units=seg_config["encoder_num_res_units"]).to(device)
            print("Encoder  ← seg checkpoint (ImageEncoder)")
        encoder.load_state_dict(encoder_state)

    encoder.eval()

    # --- MaskVAE (optional) ---
    # CLI arg takes precedence; fall back to path stored in the seg checkpoint config.
    mask_vae_path = getattr(args, "mask_vae_checkpoint", None) or seg_config.get("mask_vae_checkpoint")
    mask_vae = None
    latent_channels = num_classes  # default: logit-transform path used 3 channels
    if mask_vae_path is not None:
        mvae_ckpt = torch.load(mask_vae_path, map_location=device, weights_only=True)
        mvae_config = mvae_ckpt["config"]
        channels = tuple(int(c) for c in mvae_config["mask_vae_channels"].split(","))
        mask_vae = MaskVAE(
            num_classes=mvae_config.get("num_classes", 3),
            latent_channels=mvae_config["latent_channels"],
            channels=channels,
            num_res_units=mvae_config["num_res_units"],
        ).to(device)
        mask_vae.load_state_dict(mvae_ckpt["vae_state_dict"])
        mask_vae.eval()
        for p in mask_vae.parameters():
            p.requires_grad_(False)
        latent_channels = mask_vae.latent_channels
        print(f"MaskVAE  ← {mask_vae_path}  (latent_channels={latent_channels})")

    config = DiffusionConfig(
        latent_channels=latent_channels,
        condition_channels=embed_dim,
        cond_proj_channels=seg_config["cond_proj_channels"],
        base_channels=seg_config["unet_base_channels"],
        num_timesteps=seg_config["num_timesteps"],
    )
    unet = UNet3D(config).to(device)
    unet.load_state_dict(seg_ckpt["unet_state_dict"])
    unet.eval()
    print(f"UNet     ← seg checkpoint: {args.seg_checkpoint}")

    use_diffusion = seg_config.get("diffusion", False)
    args = argparse.Namespace(**vars(args),
                              num_classes=num_classes,
                              num_timesteps=seg_config["num_timesteps"],
                              use_diffusion=use_diffusion)
    if use_diffusion:
        print("Inference mode  : DDIM denoising loop (checkpoint trained with --diffusion)")
    else:
        print("Inference mode  : single forward pass with random noise (default)")
    return encoder, unet, mask_vae, args


# ---------------------------------------------------------------------------
# Modality mask
# ---------------------------------------------------------------------------

def parse_modality_mask(spec: str, device: torch.device) -> torch.Tensor | None:
    if spec == "all":
        return None
    if spec == "random":
        return sample_modality_mask(1).to(device)
    if len(spec) == 4 and all(c in "01" for c in spec):
        return torch.tensor([[bool(int(c)) for c in spec]], dtype=torch.bool).to(device)
    raise ValueError(f"Invalid --modality_mask '{spec}'.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    tp = (pred & gt).sum()
    return float(tp / ((pred | gt).sum() + eps))


def seg_dist(s1: np.ndarray, s2: np.ndarray) -> float:
    """1 - Dice as segmentation distance."""
    return 1.0 - dice(s1, s2)


def ged(samples: np.ndarray, gt: np.ndarray) -> float:
    """
    Generalized Energy Distance.
      GED = 2 * E[d(pred, gt)] - E[d(pred, pred')]
    samples: (N, H, W, D) bool  |  gt: (H, W, D) bool
    Lower is better (0 = perfect).
    """
    n = len(samples)
    cross = float(np.mean([seg_dist(samples[i], gt) for i in range(n)]))
    if n > 1:
        pairwise = float(np.mean([
            seg_dist(samples[i], samples[j])
            for i in range(n) for j in range(n) if i != j
        ]))
    else:
        pairwise = 0.0
    return 2 * cross - pairwise


def d_max(samples: np.ndarray) -> float:
    """Max pairwise distance between samples — measures sample spread."""
    n = len(samples)
    if n <= 1:
        return 0.0
    return float(max(
        seg_dist(samples[i], samples[j])
        for i in range(n) for j in range(i + 1, n)
    ))


def compute_all_metrics(
    stack_np: np.ndarray,   # (N, 3, H, W, D) float sigmoid
    gt_np: np.ndarray,      # (3, H, W, D) float binary
    unc_np: np.ndarray,     # (3, H, W, D) float variance
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all metrics across all three regions, return flat dict."""
    results: dict[str, float] = {}
    for r, rname in enumerate(REGION_NAMES):
        gt_bin      = gt_np[r] > threshold               # (H, W, D) bool
        ens_bin     = stack_np[:, r].mean(axis=0) > threshold  # ensemble
        samples_bin = stack_np[:, r] > threshold          # (N, H, W, D)

        results[f"{rname}_dice"]        = dice(ens_bin, gt_bin)
        results[f"{rname}_iou"]         = iou(ens_bin, gt_bin)
        results[f"{rname}_ged"]         = ged(samples_bin, gt_bin)
        results[f"{rname}_dmax"]        = d_max(samples_bin)
        results[f"{rname}_uncertainty"] = float(unc_np[r].mean())

    # Macro averages across regions
    for metric in ("dice", "iou", "ged", "dmax", "uncertainty"):
        results[f"mean_{metric}"] = float(
            np.mean([results[f"{r}_{metric}"] for r in REGION_NAMES])
        )
    return results


# ---------------------------------------------------------------------------
# N-sample inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_n_samples(
    encoder,
    unet: UNet3D,
    volume: torch.Tensor,
    n_samples: int,
    num_classes: int,
    num_inference_steps: int,
    num_train_timesteps: int,
    use_diffusion: bool,
    device: torch.device,
    mask_vae: MaskVAE | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate N mask predictions for one volume.

    use_diffusion=True : N independent DDIM denoising chains — each x_T seed
                         gives a different mask (genuine diversity).
    use_diffusion=False: N single forward passes with different random noise.
                         Diversity comes from noise sensitivity of the UNet;
                         in practice D_max will be near 0 (model ignores noise).
    """
    full_spatial   = volume.shape[2:]
    enc_out        = encoder.encode(volume)
    c              = enc_out[0] if isinstance(enc_out, tuple) else enc_out
    latent_spatial = c.shape[2:]

    lat_ch = mask_vae.latent_channels if mask_vae is not None else num_classes

    if use_diffusion:
        inf_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        inf_scheduler.set_timesteps(num_inference_steps)

    sigmoids = []
    for _ in range(n_samples):
        if use_diffusion:
            x = torch.randn(1, lat_ch, *latent_spatial, device=device)
            for t in inf_scheduler.timesteps:
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                pred_noise = unet(x, t_batch, c)
                x = inf_scheduler.step(pred_noise, t, x)[0].to(device).float()
        else:
            noise = torch.randn(1, lat_ch, *latent_spatial, device=device)
            t_zero = torch.zeros(1, dtype=torch.long, device=device)
            x = unet(noise, t_zero, c)

        if mask_vae is not None:
            probs = mask_vae.decode(x).sigmoid()    # (1, 3, D, H, W) full-res
        else:
            x_full = F.interpolate(x, size=full_spatial, mode="trilinear", align_corners=False)
            probs = x_full.sigmoid()
        sigmoids.append(probs)

    stack       = torch.stack(sigmoids)   # (N, 1, num_classes, H, W, D)
    ensemble    = stack.mean(dim=0)
    uncertainty = stack.var(dim=0)
    return ensemble, uncertainty, stack


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _best_slice(vol: np.ndarray) -> int:
    return int(vol.sum(axis=(0, 1, 2)).argmax())


def save_vis(
    vol_np: np.ndarray,       # (4, H, W, D)
    gt_np: np.ndarray,        # (3, H, W, D)
    stack_np: np.ndarray,     # (N, 3, H, W, D)
    unc_np: np.ndarray,       # (3, H, W, D)
    metrics: dict[str, float],
    modality_present: list[bool],
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    n      = stack_np.shape[0]
    best_z = _best_slice(vol_np)

    # cols: T2 | GT | sample_1..N | ensemble | uncertainty
    n_cols = 2 + n + 2
    n_rows = 3  # WT / TC / ET

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 2.4),
                             squeeze=False)

    mod_str = "".join(f"{m}={'✓' if p else '✗'}" for m, p in zip(MODALITY_NAMES, modality_present))
    fig.suptitle(
        f"N={n}  |  {mod_str}\n"
        f"Dice WT={metrics['WT_dice']:.3f} TC={metrics['TC_dice']:.3f} ET={metrics['ET_dice']:.3f}  "
        f"GED WT={metrics['WT_ged']:.3f} TC={metrics['TC_ged']:.3f} ET={metrics['ET_ged']:.3f}",
        fontsize=8,
    )

    t2      = vol_np[2, :, :, best_z]
    t2_norm = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-6)

    for row, rname in enumerate(REGION_NAMES):
        col = 0

        # T2 background
        axes[row][col].imshow(t2_norm, cmap="gray", vmin=0, vmax=1)
        axes[row][col].set_ylabel(rname, fontsize=9)
        axes[row][col].set_title("T2" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        # GT
        axes[row][col].imshow(t2_norm, cmap="gray", vmin=0, vmax=1, alpha=0.5)
        axes[row][col].imshow(gt_np[row, :, :, best_z] > threshold,
                              cmap="Blues", vmin=0, vmax=1, alpha=0.6)
        axes[row][col].set_title("GT" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        # N samples
        for i in range(n):
            s = stack_np[i, row, :, :, best_z] > threshold
            axes[row][col].imshow(t2_norm, cmap="gray", vmin=0, vmax=1, alpha=0.4)
            axes[row][col].imshow(s, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
            axes[row][col].set_title(f"s{i+1}" if row == 0 else "", fontsize=7)
            axes[row][col].axis("off")
            col += 1

        # Ensemble
        ens = stack_np[:, row].mean(axis=0)[:, :, best_z] > threshold
        axes[row][col].imshow(t2_norm, cmap="gray", vmin=0, vmax=1, alpha=0.4)
        axes[row][col].imshow(ens, cmap="Greens", vmin=0, vmax=1, alpha=0.6)
        d = metrics[f"{rname}_dice"]
        axes[row][col].set_title(f"ensemble\nDice={d:.3f}" if row == 0
                                 else f"Dice={d:.3f}", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        # Uncertainty
        unc = unc_np[row, :, :, best_z]
        im  = axes[row][col].imshow(unc, cmap="hot", vmin=0, vmax=max(unc.max(), 1e-6))
        g   = metrics[f"{rname}_ged"]
        axes[row][col].set_title(f"uncertainty\nGED={g:.3f}" if row == 0
                                 else f"GED={g:.3f}", fontsize=7)
        axes[row][col].axis("off")
        plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Vis  → {save_path}")


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    device     = torch.device(args.device)
    splits_dir = args.splits_dir or args.data_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, unet, mask_vae, args = load_models(args, device)

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.test_split_file),
        crop_size=args.crop_size,
        region_based=True,    # loads GT as (3, H, W, D) [WT, TC, ET]
        random_crop=False,
    )
    n_cases = min(args.num_cases, len(dataset))

    mod_tensor = parse_modality_mask(args.modality_mask, device)
    mod_present = [True] * 4 if mod_tensor is None else mod_tensor[0].cpu().tolist()
    present_str = "".join("1" if p else "0" for p in mod_present)

    print(f"\nEvaluating {n_cases} cases  |  n_samples={args.n_samples}  "
          f"|  modalities={''.join(m for m,p in zip(MODALITY_NAMES,mod_present) if p)}\n")

    all_rows = []
    agg: dict[str, list] = {}

    for i in range(n_cases):
        vol, gt = dataset[i]
        vol = vol.unsqueeze(0).to(device)   # (1, 4, H, W, D)
        vol_input = apply_modality_mask(vol, mod_tensor) if mod_tensor is not None else vol

        ensemble, uncertainty, stack = infer_n_samples(
            encoder, unet, vol_input,
            n_samples=args.n_samples,
            num_classes=args.num_classes,
            num_inference_steps=args.num_inference_steps,
            num_train_timesteps=args.num_timesteps,
            use_diffusion=args.use_diffusion,
            device=device,
            mask_vae=mask_vae,
        )

        gt_np    = gt.numpy()                   # (3, H, W, D)
        stack_np = stack[:, 0].cpu().numpy()    # (N, 3, H, W, D)
        unc_np   = uncertainty[0].cpu().numpy() # (3, H, W, D)
        vol_np   = vol[0].cpu().numpy()         # (4, H, W, D)

        metrics = compute_all_metrics(stack_np, gt_np, unc_np, args.threshold)

        # Print per-case table
        print(f"Case {i:03d}")
        print(f"  {'Region':<6} {'Dice':>6} {'IoU':>6} {'GED':>6} {'D_max':>6} {'Unc':>8}")
        for r in REGION_NAMES:
            print(f"  {r:<6} "
                  f"{metrics[f'{r}_dice']:>6.3f} "
                  f"{metrics[f'{r}_iou']:>6.3f} "
                  f"{metrics[f'{r}_ged']:>6.3f} "
                  f"{metrics[f'{r}_dmax']:>6.3f} "
                  f"{metrics[f'{r}_uncertainty']:>8.5f}")
        print(f"  {'mean':<6} "
              f"{metrics['mean_dice']:>6.3f} "
              f"{metrics['mean_iou']:>6.3f} "
              f"{metrics['mean_ged']:>6.3f} "
              f"{metrics['mean_dmax']:>6.3f} "
              f"{metrics['mean_uncertainty']:>8.5f}\n")

        for k, v in metrics.items():
            agg.setdefault(k, []).append(v)

        row = {"case": i, "modalities": present_str, "n_samples": args.n_samples}
        row.update(metrics)
        all_rows.append(row)

        vis_path = output_dir / f"case_{i:03d}_n{args.n_samples}_mod{present_str}.png"
        save_vis(vol_np, gt_np, stack_np, unc_np, metrics, mod_present, vis_path, args.threshold)

    # Summary across all cases
    print("=" * 55)
    print(f"SUMMARY  ({n_cases} cases, N={args.n_samples}, mod={present_str})")
    print(f"  {'Region':<6} {'Dice':>6} {'IoU':>6} {'GED':>6} {'D_max':>6} {'Unc':>8}")
    for r in REGION_NAMES:
        print(f"  {r:<6} "
              f"{np.mean(agg[f'{r}_dice']):>6.3f} "
              f"{np.mean(agg[f'{r}_iou']):>6.3f} "
              f"{np.mean(agg[f'{r}_ged']):>6.3f} "
              f"{np.mean(agg[f'{r}_dmax']):>6.3f} "
              f"{np.mean(agg[f'{r}_uncertainty']):>8.5f}")
    print("=" * 55)

    save_csv(all_rows, output_dir / f"metrics_n{args.n_samples}_mod{present_str}.csv")
    print(f"\nDone. Output → {output_dir}/")


if __name__ == "__main__":
    main()
