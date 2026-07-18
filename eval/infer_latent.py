"""
Inference + evaluation for the latent diffusion pipeline.

Loads a trained diffusion UNet checkpoint (from train_latent_diffusion.py),
runs N independent DDIM sampling chains per test case, and reports:

  Per region (WT / TC / ET):
    Dice       — ensemble prediction vs GT          (higher = better)
    IoU        — intersection over union            (higher = better)
    GED        — Generalized Energy Distance        (lower  = better)
    D_max      — max pairwise distance between samples (sample diversity)
    Uncertainty— mean per-voxel variance across N samples

Use --all_combos to sweep all 15 modality combinations and get the full
missing-modality evaluation table (the main result in the paper).

Single-combo quick test:
    python3 -m eval.infer_latent \\
        --diffusion_ckpt checkpoints/latent_diffusion_.../best.pth \\
        --data_root      data/brats_roi128_2023 \\
        --splits_dir     splits/brats_roi128_smoke \\
        --n_samples      5 \\
        --modality_mask  all \\
        --device         mps

Full 15-combo sweep (A100, full test split):
    python3 -m eval.infer_latent \\
        --diffusion_ckpt checkpoints/latent_diffusion_.../best.pth \\
        --data_root      data/brats_roi128_2023 \\
        --splits_dir     splits/brats_roi128_full \\
        --n_samples      5 \\
        --all_combos \\
        --device         cuda
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.brats_dataset import (
    ALL_MODALITY_COMBINATIONS,
    apply_modality_mask,
    seg_to_regions,
    subregions_to_regions,
)
from models.diffusion.config import DiffusionConfig
from models.diffusion.schedule import GaussianDiffusionSchedule
from models.diffusion.unet3d import UNet3D
from models.multiencoder.encoders import ImageVAE, MaskVAE

MODALITY_NAMES = ["FLAIR", "T1ce", "T1", "T2"]
REGION_NAMES   = ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--diffusion_ckpt", required=True,
                   help="best.pth or final.pth from train_latent_diffusion.py")
    p.add_argument("--data_root",  required=True,
                   help="ROI-cropped 128³ data folder with vol/ and seg/")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--test_split_file", default="test.txt")
    p.add_argument("--num_cases",  type=int, default=None,
                   help="Number of test cases. Default: all in test.txt.")

    # Override VAE paths (optional — defaults to what's stored in the diffusion ckpt)
    p.add_argument("--image_vae_ckpt", default=None)
    p.add_argument("--mask_vae_ckpt",  default=None)

    p.add_argument("--n_samples",           type=int,   default=5,
                   help="Independent DDIM chains per case. Paper shows Dice stabilises at 5.")
    p.add_argument("--num_inference_steps", type=int,   default=50,
                   help="DDIM steps per sample. 50 ≈ 1000-step DDPM quality.")
    p.add_argument("--threshold",           type=float, default=0.5)

    p.add_argument("--modality_mask", default="all",
                   help="'all' = all present | '1101' = specific combo | 'random'.")
    p.add_argument("--all_combos", action="store_true",
                   help="Sweep all 15 modality combinations and write summary table.")

    p.add_argument("--output_dir", default="eval_output/latent_diffusion")
    p.add_argument("--device",
                   default="mps"  if torch.backends.mps.is_available()
                   else   "cuda"  if torch.cuda.is_available()
                   else   "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_image_vae(ckpt_path: str, device: torch.device) -> ImageVAE:
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg      = ckpt["config"]
    channels = tuple(int(c) for c in cfg["encoder_channels"].split(","))
    vae = ImageVAE(
        in_channels=4,
        channels=channels,
        num_res_units=cfg.get("encoder_num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_mask_vae(ckpt_path: str, device: torch.device) -> tuple[MaskVAE, bool]:
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg         = ckpt["config"]
    channels    = tuple(int(c) for c in cfg["mask_vae_channels"].split(","))
    subregion   = bool(cfg.get("subregion_mode", False))
    num_classes = 4 if subregion else 3
    vae = MaskVAE(
        num_classes=num_classes,
        latent_channels=cfg.get("latent_channels", 4),
        channels=channels,
        num_res_units=cfg.get("num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, subregion


def load_models(args: argparse.Namespace, device: torch.device):
    diff_ckpt  = torch.load(args.diffusion_ckpt, map_location=device, weights_only=True)
    diff_cfg   = diff_ckpt["config"]

    # VAE paths: CLI overrides > stored in checkpoint > error
    img_ckpt_path  = args.image_vae_ckpt  or diff_ckpt.get("image_vae_ckpt")
    mask_ckpt_path = args.mask_vae_ckpt   or diff_ckpt.get("mask_vae_ckpt")
    if not img_ckpt_path:
        raise ValueError("Provide --image_vae_ckpt (not stored in checkpoint).")
    if not mask_ckpt_path:
        raise ValueError("Provide --mask_vae_ckpt (not stored in checkpoint).")

    print(f"Loading ImageVAE  from {img_ckpt_path}")
    image_vae = _load_image_vae(img_ckpt_path, device)

    print(f"Loading MaskVAE   from {mask_ckpt_path}")
    mask_vae, subregion = _load_mask_vae(mask_ckpt_path, device)
    print(f"  subregion_mode = {subregion}")

    channel_mults = tuple(int(m) for m in diff_cfg["channel_mults"].split(","))
    cfg = DiffusionConfig(
        latent_channels     = diff_cfg["latent_channels"],
        condition_channels  = image_vae.embed_dim,
        cond_proj_channels  = diff_cfg["cond_proj_channels"],
        base_channels       = diff_cfg["base_channels"],
        channel_multipliers = channel_mults,
        num_res_blocks_per_level = diff_cfg.get("num_res_blocks", 2),
        num_timesteps       = diff_cfg["num_timesteps"],
    )
    unet = UNet3D(cfg).to(device)
    unet.load_state_dict(diff_ckpt["unet_state_dict"])
    unet.eval()
    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"UNet3D loaded  ({n_params:.1f}M params, best_dice={diff_ckpt.get('best_dice', '?'):.4f})")

    schedule = GaussianDiffusionSchedule(
        num_timesteps=diff_cfg["num_timesteps"]
    ).to(device)

    return unet, image_vae, mask_vae, schedule, subregion


# ---------------------------------------------------------------------------
# DDIM sampler  (deterministic, eta=0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    unet:      UNet3D,
    schedule:  GaussianDiffusionSchedule,
    condition: torch.Tensor,
    latent_shape: tuple[int, ...],
    *,
    device:    torch.device,
    n_steps:   int = 50,
) -> torch.Tensor:
    T   = schedule.num_timesteps
    gap = max(T // n_steps, 1)
    ts  = list(reversed(range(0, T, gap)))
    ab  = schedule.alpha_bars

    x_t = torch.randn(latent_shape, device=device)
    for i, t_curr in enumerate(ts):
        t_prev   = ts[i + 1] if i + 1 < len(ts) else -1
        t_batch  = torch.full((latent_shape[0],), t_curr, device=device, dtype=torch.long)
        eps_pred = unet(x_t, t_batch, condition)
        ab_t     = ab[t_curr].float()
        x0_pred  = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        if t_prev >= 0:
            ab_prev = ab[t_prev].float()
            x_t = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred
        else:
            x_t = x0_pred
    return x_t


# ---------------------------------------------------------------------------
# N-sample inference for one volume
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_n_samples(
    unet:       UNet3D,
    image_vae:  ImageVAE,
    mask_vae:   MaskVAE,
    schedule:   GaussianDiffusionSchedule,
    vol:        torch.Tensor,          # (1, 4, H, W, D)
    *,
    n_samples:  int,
    n_inf_steps: int,
    device:     torch.device,
    subregion:  bool,
    use_amp:    bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        stack_np    (N, 3, H, W, D)  — N sigmoid mask predictions [WT,TC,ET]
        unc_np      (3, H, W, D)     — per-voxel variance across N samples
    """
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        mu_img, _ = image_vae.encode(vol)   # (1, embed_dim, 32, 32, 32)

    latent_shape = (1, mask_vae.latent_channels, *mu_img.shape[2:])
    samples = []

    for _ in range(n_samples):
        z0 = ddim_sample(unet, schedule, mu_img, latent_shape,
                         device=device, n_steps=n_inf_steps)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = mask_vae.decode(z0)
        pred = logits.sigmoid().squeeze(0).cpu().numpy()  # (C, H, W, D)

        if subregion:
            pred_regions = subregions_to_regions(pred)    # (3, H, W, D)
        else:
            pred_regions = pred                            # already (3, H, W, D)
        samples.append(pred_regions)

    stack_np = np.stack(samples)                          # (N, 3, H, W, D)
    unc_np   = stack_np.var(axis=0)                       # (3, H, W, D)
    return stack_np, unc_np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    pred_b = pred > 0.5
    gt_b   = gt   > 0.5
    tp = (pred_b & gt_b).sum()
    return float(2 * tp / (pred_b.sum() + gt_b.sum() + eps))


def _iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-5) -> float:
    pred_b = pred > 0.5
    gt_b   = gt   > 0.5
    tp = (pred_b & gt_b).sum()
    return float(tp / ((pred_b | gt_b).sum() + eps))


def _seg_dist(s1: np.ndarray, s2: np.ndarray) -> float:
    return 1.0 - _dice(s1, s2)


def _ged(samples: np.ndarray, gt: np.ndarray) -> float:
    """
    Generalized Energy Distance = 2·E[d(pred,gt)] − E[d(pred,pred')].
    Lower = better. 0 = perfect and no diversity (trivial); a good model
    has low GED with nonzero D_max (accurate AND calibrated spread).
    """
    n = len(samples)
    cross = np.mean([_seg_dist(samples[i], gt) for i in range(n)])
    pairwise = (
        np.mean([_seg_dist(samples[i], samples[j])
                 for i in range(n) for j in range(n) if i != j])
        if n > 1 else 0.0
    )
    return float(2 * cross - pairwise)


def _d_max(samples: np.ndarray) -> float:
    """Max pairwise distance — sample diversity, higher = more spread."""
    n = len(samples)
    if n <= 1:
        return 0.0
    return float(max(
        _seg_dist(samples[i], samples[j])
        for i in range(n) for j in range(i + 1, n)
    ))


def compute_metrics(
    stack_np: np.ndarray,   # (N, 3, H, W, D) sigmoid probabilities
    gt_np:    np.ndarray,   # (3, H, W, D) binary float
    unc_np:   np.ndarray,   # (3, H, W, D) variance
) -> dict[str, float]:
    results: dict[str, float] = {}
    for r, name in enumerate(REGION_NAMES):
        gt       = gt_np[r]
        ensemble = stack_np[:, r].mean(axis=0)
        samples  = stack_np[:, r]

        results[f"{name}_dice"]        = _dice(ensemble, gt)
        results[f"{name}_iou"]         = _iou(ensemble, gt)
        results[f"{name}_ged"]         = _ged(samples,  gt)
        results[f"{name}_dmax"]        = _d_max(samples)
        results[f"{name}_uncertainty"] = float(unc_np[r].mean())

    for metric in ("dice", "iou", "ged", "dmax", "uncertainty"):
        results[f"mean_{metric}"] = float(
            np.mean([results[f"{r}_{metric}"] for r in REGION_NAMES])
        )
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

LABEL_COLOURS = {0: (0,0,0), 1: (0,0,200), 2: (0,200,0), 4: (200,0,0)}


def _regions_to_rgb(wt: np.ndarray, tc: np.ndarray, et: np.ndarray) -> np.ndarray:
    seg = np.zeros(wt.shape, dtype=np.uint8)
    seg[wt > 0.5] = 2   # ED (green)
    seg[tc > 0.5] = 1   # NCR (blue)
    seg[et > 0.5] = 4   # ET (red)
    rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for v, c in LABEL_COLOURS.items():
        rgb[seg == v] = c
    return rgb


def save_vis(
    vol_np:   np.ndarray,    # (4, H, W, D)
    gt_np:    np.ndarray,    # (3, H, W, D) [WT, TC, ET]
    stack_np: np.ndarray,    # (N, 3, H, W, D)
    unc_np:   np.ndarray,    # (3, H, W, D)
    metrics:  dict[str, float],
    mod_present: list[bool],
    save_path: Path,
) -> None:
    N = stack_np.shape[0]
    # Pick best tumour slice along last (D) axis
    best_z = int(gt_np[0].sum(axis=(0, 1)).argmax())

    # FLAIR background (channel 0)
    flair = vol_np[0, :, :, best_z]
    flair_n = (flair - flair.min()) / (flair.max() - flair.min() + 1e-6)

    # Columns: FLAIR | GT | sample_1..N | ensemble | uncertainty-map
    n_cols = 2 + N + 2
    n_rows = 3   # one row per region

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 2.4),
                             squeeze=False)

    mod_str = "".join(m[0] for m, p in zip(MODALITY_NAMES, mod_present) if p)
    d_str   = "  ".join(
        f"{r}:{metrics[f'{r}_dice']:.3f}" for r in REGION_NAMES
    )
    fig.suptitle(
        f"Mods={mod_str}  N={N}  |  {d_str}\n"
        f"GED  " + "  ".join(f"{r}:{metrics[f'{r}_ged']:.3f}" for r in REGION_NAMES),
        fontsize=8,
    )

    gt_rgb = _regions_to_rgb(gt_np[0, :, :, best_z],
                              gt_np[1, :, :, best_z],
                              gt_np[2, :, :, best_z])

    for row, rname in enumerate(REGION_NAMES):
        col = 0

        # FLAIR
        axes[row][col].imshow(flair_n, cmap="gray")
        axes[row][col].set_ylabel(rname, fontsize=9)
        axes[row][col].set_title("FLAIR" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        # GT (colour-coded tumour map, same for each row)
        axes[row][col].imshow(gt_rgb)
        axes[row][col].set_title("GT" if row == 0 else "", fontsize=7)
        axes[row][col].axis("off")
        col += 1

        # N individual samples
        for s in range(N):
            pred_rgb = _regions_to_rgb(
                stack_np[s, 0, :, :, best_z],
                stack_np[s, 1, :, :, best_z],
                stack_np[s, 2, :, :, best_z],
            )
            axes[row][col].imshow(flair_n, cmap="gray", alpha=0.3)
            axes[row][col].imshow(pred_rgb, alpha=0.7)
            axes[row][col].set_title(f"s{s+1}" if row == 0 else "", fontsize=7)
            axes[row][col].axis("off")
            col += 1

        # Ensemble
        ens_rgb = _regions_to_rgb(
            stack_np[:, 0, :, :, best_z].mean(0),
            stack_np[:, 1, :, :, best_z].mean(0),
            stack_np[:, 2, :, :, best_z].mean(0),
        )
        d = metrics[f"{rname}_dice"]
        axes[row][col].imshow(flair_n, cmap="gray", alpha=0.3)
        axes[row][col].imshow(ens_rgb, alpha=0.7)
        axes[row][col].set_title(
            f"ensemble\nDice={d:.3f}" if row == 0 else f"Dice={d:.3f}", fontsize=7
        )
        axes[row][col].axis("off")
        col += 1

        # Uncertainty heatmap (per-region)
        unc = unc_np[row, :, :, best_z]
        im  = axes[row][col].imshow(unc, cmap="hot", vmin=0,
                                    vmax=max(float(unc.max()), 1e-8))
        g   = metrics[f"{rname}_ged"]
        axes[row][col].set_title(
            f"uncertainty\nGED={g:.3f}" if row == 0 else f"GED={g:.3f}", fontsize=7
        )
        axes[row][col].axis("off")
        plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(summary: list[dict], output_dir: Path) -> None:
    col_w = 7
    lines = []
    sep   = "-" * (30 + col_w * 8 + 16)
    lines.append(sep)
    lines.append(
        f"{'Combo':<6}  {'Modalities':<22}"
        + "".join(f"  {f'{r}_{m[:3]}':{col_w}}"
                  for r in REGION_NAMES for m in ("dice", "ged"))
        + f"  {'mean_d':>{col_w}}  {'mean_g':>{col_w}}"
    )
    lines.append(sep)

    for row in summary:
        vals = []
        for r in REGION_NAMES:
            vals.append(f"{row[f'{r}_dice']:>{col_w}.3f}")
            vals.append(f"{row[f'{r}_ged']:>{col_w}.3f}")
        vals.append(f"{row['mean_dice']:>{col_w}.3f}")
        vals.append(f"{row['mean_ged']:>{col_w}.3f}")
        lines.append(
            f"{row['combo']:<6}  {row['modalities'][:22]:<22}" + "".join(vals)
        )

    lines.append(sep)
    overall_dice = np.mean([r["mean_dice"] for r in summary])
    overall_ged  = np.mean([r["mean_ged"]  for r in summary])
    lines.append(
        f"{'ALL':<6}  {'(mean over 15 combos)':<22}"
        + " " * (col_w * 6 + 12)
        + f"  {overall_dice:>{col_w}.3f}  {overall_ged:>{col_w}.3f}"
    )
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)
    tpath = output_dir / "summary_table.txt"
    tpath.write_text(table)
    print(f"\n  Table  → {tpath}")


# ---------------------------------------------------------------------------
# Core eval loop (one modality combo)
# ---------------------------------------------------------------------------

def _read_names(path: str | Path) -> list[str]:
    with open(path) as f:
        return sorted(line.strip() for line in f if line.strip())


def eval_combo(
    unet:       UNet3D,
    image_vae:  ImageVAE,
    mask_vae:   MaskVAE,
    schedule:   GaussianDiffusionSchedule,
    names:      list[str],
    vol_dir:    Path,
    seg_dir:    Path,
    mod_tensor: torch.Tensor | None,    # (1, 4) bool, None = all present
    *,
    n_samples:     int,
    n_inf_steps:   int,
    device:        torch.device,
    subregion:     bool,
    use_amp:       bool,
    output_dir:    Path,
    combo_str:     str,
    save_vis_flag: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    mod_present = [True] * 4 if mod_tensor is None else mod_tensor[0].cpu().tolist()

    for name in names:
        vol = np.load(vol_dir / f"{name}_vol.npy")   # (H,W,D,4)
        seg = np.load(seg_dir / f"{name}_seg.npy")   # (H,W,D)

        vol_t = torch.from_numpy(
            vol.transpose(3, 0, 1, 2).astype(np.float32)
        ).unsqueeze(0).to(device)                     # (1,4,128,128,128)

        if mod_tensor is not None:
            vol_t = apply_modality_mask(vol_t, mod_tensor)

        stack_np, unc_np = infer_n_samples(
            unet, image_vae, mask_vae, schedule, vol_t,
            n_samples=n_samples, n_inf_steps=n_inf_steps,
            device=device, subregion=subregion, use_amp=use_amp,
        )
        gt_np = seg_to_regions(seg)   # (3,H,W,D) [WT,TC,ET]

        metrics = compute_metrics(stack_np, gt_np, unc_np)

        if save_vis_flag:
            vis_path = (output_dir / "vis"
                        / f"{combo_str}_{name[:16]}_n{n_samples}.png")
            save_vis(vol.transpose(3, 0, 1, 2), gt_np,
                     stack_np, unc_np, metrics, mod_present, vis_path)

        row = {"case": name, "combo": combo_str,
               "n_samples": n_samples, **metrics}
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    device  = torch.device(args.device)
    use_amp = device.type == "cuda"

    unet, image_vae, mask_vae, schedule, subregion = load_models(args, device)

    vol_dir    = Path(args.data_root) / "vol"
    seg_dir    = Path(args.data_root) / "seg"
    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_names = _read_names(splits_dir / args.test_split_file)
    if args.num_cases:
        all_names = all_names[:args.num_cases]

    print(f"\nTest cases : {len(all_names)}")
    print(f"N samples  : {args.n_samples}")
    print(f"DDIM steps : {args.num_inference_steps}")
    print(f"Output     : {output_dir}\n")

    if args.all_combos:
        # ── Sweep all 15 modality combinations ──────────────────────────────
        from tqdm import tqdm
        all_rows: list[dict] = []
        summary:  list[dict] = []

        for combo in tqdm(ALL_MODALITY_COMBINATIONS, desc="Combos"):
            combo_str = "".join("1" if c else "0" for c in combo)
            mod_label = "+".join(m for m, c in zip(MODALITY_NAMES, combo) if c)
            mod_t     = torch.tensor([list(combo)], dtype=torch.bool).to(device)

            rows = eval_combo(
                unet, image_vae, mask_vae, schedule,
                all_names, vol_dir, seg_dir, mod_t,
                n_samples=args.n_samples, n_inf_steps=args.num_inference_steps,
                device=device, subregion=subregion, use_amp=use_amp,
                output_dir=output_dir, combo_str=combo_str,
                save_vis_flag=True,
            )
            all_rows.extend(rows)

            agg = {}
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, float):
                        agg.setdefault(k, []).append(v)
            summary.append({
                "combo":       combo_str,
                "modalities":  mod_label,
                "n_mods":      sum(combo),
                **{k: float(np.mean(vs)) for k, vs in agg.items()},
            })

            tqdm.write(
                f"[{combo_str}] {mod_label:<22}  "
                f"Dice WT={summary[-1]['WT_dice']:.3f} "
                f"TC={summary[-1]['TC_dice']:.3f} "
                f"ET={summary[-1]['ET_dice']:.3f}  "
                f"mean={summary[-1]['mean_dice']:.3f}"
            )

        # Save CSVs
        full_csv = output_dir / "metrics_full.csv"
        with full_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        summary_csv = output_dir / "summary.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

        print_summary_table(summary, output_dir)
        print(f"\nFull CSV   → {full_csv}")
        print(f"Summary    → {summary_csv}")

    else:
        # ── Single combo ─────────────────────────────────────────────────────
        if args.modality_mask == "all":
            mod_t     = None
            combo_str = "1111"
        elif args.modality_mask == "random":
            from data.brats_dataset import sample_modality_mask
            mod_t     = sample_modality_mask(1).to(device)
            combo_str = "".join("1" if c else "0" for c in mod_t[0].tolist())
        else:
            spec = args.modality_mask
            if len(spec) != 4 or not all(c in "01" for c in spec):
                raise ValueError(f"--modality_mask must be 'all', 'random', or 4 binary chars, got '{spec}'")
            mod_t     = torch.tensor([[bool(int(c)) for c in spec]],
                                     dtype=torch.bool).to(device)
            combo_str = spec

        rows = eval_combo(
            unet, image_vae, mask_vae, schedule,
            all_names, vol_dir, seg_dir, mod_t,
            n_samples=args.n_samples, n_inf_steps=args.num_inference_steps,
            device=device, subregion=subregion, use_amp=use_amp,
            output_dir=output_dir, combo_str=combo_str,
            save_vis_flag=True,
        )

        # Print per-case table
        for row in rows:
            print(f"\nCase {row['case']}")
            print(f"  {'Region':<5}  {'Dice':>6}  {'IoU':>6}  "
                  f"{'GED':>6}  {'D_max':>6}  {'Unc':>8}")
            for r in REGION_NAMES:
                print(f"  {r:<5}  "
                      f"{row[f'{r}_dice']:>6.3f}  "
                      f"{row[f'{r}_iou']:>6.3f}  "
                      f"{row[f'{r}_ged']:>6.3f}  "
                      f"{row[f'{r}_dmax']:>6.3f}  "
                      f"{row[f'{r}_uncertainty']:>8.5f}")

        # Summary across cases
        print(f"\n{'='*55}")
        print(f"SUMMARY  ({len(rows)} cases, N={args.n_samples}, combo={combo_str})")
        print(f"  {'Region':<5}  {'Dice':>6}  {'IoU':>6}  "
              f"{'GED':>6}  {'D_max':>6}  {'Unc':>8}")
        for r in REGION_NAMES:
            print(f"  {r:<5}  "
                  f"{np.mean([row[f'{r}_dice'] for row in rows]):>6.3f}  "
                  f"{np.mean([row[f'{r}_iou']  for row in rows]):>6.3f}  "
                  f"{np.mean([row[f'{r}_ged']  for row in rows]):>6.3f}  "
                  f"{np.mean([row[f'{r}_dmax'] for row in rows]):>6.3f}  "
                  f"{np.mean([row[f'{r}_uncertainty'] for row in rows]):>8.5f}")
        print(f"{'='*55}")

        csv_path = output_dir / f"metrics_{combo_str}_n{args.n_samples}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV  → {csv_path}")
        print(f"  Vis  → {output_dir}/vis/")


if __name__ == "__main__":
    main()
