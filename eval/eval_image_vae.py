"""
Evaluate a trained ImageVAE on the test split.

Shows original vs reconstructed MRI slices for each modality,
with optional random modality dropout to test missing-modality handling.

Usage:
    python3 -m eval.eval_image_vae \\
        --checkpoint checkpoints/image_vae_xxx/best.pth \\
        --output_dir eval_out/image_vae

Optional flags:
    --n_cases   8        # cases to include in the grid (default 8)
    --split     test     # which split to evaluate on (default: test)
    --drop_mods          # also show a missing-modality reconstruction per case
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask
from models.multiencoder.encoders import ImageVAE

MODALITY_NAMES = ["FLAIR", "T1ce", "T1", "T2"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim_approx(pred: torch.Tensor, target: torch.Tensor, win: int = 7) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(1, 1, win, win, win, device=pred.device) / (win ** 3)
    def conv(x): return F.conv3d(x, k, padding=win // 2)
    mx, my = conv(pred), conv(target)
    sx = conv(pred * pred) - mx * mx
    sy = conv(target * target) - my * my
    sxy = conv(pred * target) - mx * my
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx**2 + my**2 + C1) * (sx + sy + C2)
    return float((num / den).mean().item())


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_vae(ckpt_path: str, device: torch.device) -> tuple[ImageVAE, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    channels = tuple(int(c) for c in cfg["encoder_channels"].split(","))
    vae = ImageVAE(
        in_channels=4,
        channels=channels,
        num_res_units=cfg.get("encoder_num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    return vae, cfg


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def best_signal_slice(vol: np.ndarray) -> int:
    """vol: (C, D, H, W) — W-index with most signal across all channels."""
    signal = vol.sum(axis=(0, 1, 2))  # sum over C,D,H → (W,)
    return int(signal.argmax()) if signal.max() > 0 else vol.shape[-1] // 2


def norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    vae: ImageVAE,
    dataset: BraTSDataset,
    n_cases: int,
    device: torch.device,
    output_dir: Path,
    drop_mods: bool,
) -> None:
    n_cases  = min(n_cases, len(dataset))
    # cols per case: for each modality → [orig | recon | (dropped_recon)]
    n_mod_cols = 3 if drop_mods else 2   # orig, recon, (+dropped)
    n_cols = len(MODALITY_NAMES) * n_mod_cols

    fig, axes = plt.subplots(
        n_cases, n_cols,
        figsize=(n_cols * 2.2, n_cases * 2.5),
        squeeze=False,
    )

    col_headers = []
    for m in MODALITY_NAMES:
        col_headers.append(f"{m}\norig")
        col_headers.append(f"{m}\nrecon")
        if drop_mods:
            col_headers.append(f"{m}\ndropped")

    for col, hdr in enumerate(col_headers):
        axes[0][col].set_title(hdr, fontsize=7)

    all_ssim, all_psnr = [], []

    for row in range(n_cases):
        vol, _ = dataset[row]                     # (4, D, H, W)
        x      = vol.unsqueeze(0).to(device)      # (1, 4, D, H, W)

        # --- Full reconstruction (all modalities present) ---
        mu, _   = vae.encode(x)
        recon   = vae.decode(mu)                  # (1, 4, D, H, W)

        vol_np   = vol.numpy()                    # (4, D, H, W)
        recon_np = recon[0].cpu().float().numpy() # (4, D, H, W)

        z = best_signal_slice(vol_np)

        # --- Dropped-modality reconstruction ---
        dropped_np = None
        if drop_mods:
            mod_mask    = sample_modality_mask(1).to(device)  # random dropout
            vol_masked  = apply_modality_mask(x, mod_mask)
            mu_drop, _  = vae.encode(vol_masked)
            recon_drop  = vae.decode(mu_drop)
            dropped_np  = recon_drop[0].cpu().float().numpy()
            present     = mod_mask[0].cpu().numpy()            # (4,) bool

        # --- Metrics (present channels only) ---
        case_ssim, case_psnr = [], []
        for c in range(4):
            orig  = vol_np[c]
            pred  = recon_np[c]
            vmin, vmax = orig.min(), orig.max()
            if (vmax - vmin) < 1e-6:
                continue
            o_n = torch.from_numpy(norm01(orig)).unsqueeze(0).unsqueeze(0)
            p_n = torch.from_numpy(norm01(pred)).unsqueeze(0).unsqueeze(0)
            case_ssim.append(ssim_approx(p_n, o_n))
            case_psnr.append(psnr(p_n, o_n))
        all_ssim.extend(case_ssim)
        all_psnr.extend(case_psnr)
        mean_ssim = np.mean(case_ssim) if case_ssim else 0.0
        mean_psnr = np.mean(case_psnr) if case_psnr else 0.0

        axes[row][0].set_ylabel(
            f"case {row}\nSSIM={mean_ssim:.3f}\nPSNR={mean_psnr:.1f}dB",
            fontsize=7,
        )

        # --- Draw columns ---
        for c, name in enumerate(MODALITY_NAMES):
            base = c * n_mod_cols

            orig_slice  = vol_np[c, :, :, z]
            recon_slice = recon_np[c, :, :, z]

            axes[row][base].imshow(norm01(orig_slice),  cmap="gray", vmin=0, vmax=1)
            axes[row][base].axis("off")

            axes[row][base + 1].imshow(norm01(recon_slice), cmap="gray", vmin=0, vmax=1)
            axes[row][base + 1].axis("off")

            if drop_mods:
                ax_drop = axes[row][base + 2]
                if present[c]:
                    ax_drop.imshow(norm01(dropped_np[c, :, :, z]), cmap="gray", vmin=0, vmax=1)
                else:
                    ax_drop.imshow(np.ones_like(orig_slice) * 0.3, cmap="gray", vmin=0, vmax=1)
                    ax_drop.text(0.5, 0.5, "MISSING", transform=ax_drop.transAxes,
                                 ha="center", va="center", fontsize=7, color="red", fontweight="bold")
                ax_drop.axis("off")

    plt.tight_layout()
    out_path = output_dir / "test_eval_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid → {out_path}")

    print(f"\n{'='*45}")
    print(f"Reconstruction quality  (n={n_cases} test cases, all modalities present)")
    print(f"{'='*45}")
    print(f"  Mean SSIM : {np.mean(all_ssim):.4f}")
    print(f"  Mean PSNR : {np.mean(all_psnr):.2f} dB")


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="eval_out/image_vae")
    p.add_argument("--split",      default="test")
    p.add_argument("--n_cases",    type=int, default=8)
    p.add_argument("--drop_mods",  action="store_true",
                   help="Show a third column per modality: recon with random modality dropout")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    vae, cfg = load_vae(args.checkpoint, device)

    splits_dir = cfg.get("splits_dir") or cfg["data_root"]
    split_file = Path(splits_dir) / f"{args.split}.txt"

    dataset = BraTSDataset(
        root=cfg["data_root"],
        split_file=split_file,
        crop_size=cfg["crop_size"],
        volume_only=True,
        random_crop=False,
    )
    print(f"Test cases : {len(dataset)}")
    print(f"Checkpoint : {args.checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate(vae, dataset, args.n_cases, device, output_dir, args.drop_mods)


if __name__ == "__main__":
    main()
