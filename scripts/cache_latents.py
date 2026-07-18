"""
Pre-compute and cache VAE latents for all cases in a dataset.

Runs the frozen ImageVAE encoder and MaskVAE encoder once over every case,
saves mu (the deterministic latent) to disk. The diffusion UNet training then
just loads these files — no VAE forward pass needed during diffusion training.

This only works because ROI crops are fixed per case (output of preprocess_roi.py).
Random crops would produce a different latent every time and couldn't be cached.

Input layout (output of preprocess_roi.py):
    data_root/
      vol/{name}_vol.npy    (128, 128, 128, 4) float32  channel-last
      seg/{name}_seg.npy    (128, 128, 128)    uint8
      train.txt / val.txt / test.txt

Output layout:
    output_dir/
      z_image/{name}_z_img.npy    (embed_dim, 16, 16, 16) float32
      z_mask/{name}_z_mask.npy    (latent_channels, 16, 16, 16) float32
      train.txt / val.txt / test.txt   (copied)

Both checkpoints are optional — omit one if only caching the other.

Usage:
    python3 scripts/cache_latents.py \\
        --data_root       data/brats_roi128_2023 \\
        --output_dir      data/brats_roi128_2023_latents \\
        --image_vae_ckpt  checkpoints/image_vae_xxx/final.pth \\
        --mask_vae_ckpt   checkpoints/mask_vae_xxx/final.pth \\
        --batch_size      8
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.multiencoder.encoders import ImageVAE, MaskVAE
from data.brats_dataset import seg_to_regions, seg_to_subregions


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_image_vae(ckpt_path: str, device: torch.device) -> ImageVAE:
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
    return vae


def _load_mask_vae(ckpt_path: str, device: torch.device) -> tuple[MaskVAE, bool]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    channels     = tuple(int(c) for c in cfg["mask_vae_channels"].split(","))
    subregion    = bool(cfg.get("subregion_mode", False))
    num_classes  = 4 if subregion else 3
    vae = MaskVAE(
        num_classes=num_classes,
        latent_channels=cfg.get("latent_channels", 4),
        channels=channels,
        num_res_units=cfg.get("num_res_units", 2),
    ).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()
    return vae, subregion


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_root",      required=True,
                   help="ROI-cropped data folder (output of preprocess_roi.py)")
    p.add_argument("--output_dir",     required=True,
                   help="Where to write cached latents")
    p.add_argument("--splits_dir",     default=None,
                   help="Folder with split .txt files. Defaults to --data_root.")
    p.add_argument("--image_vae_ckpt", default=None,
                   help="Path to ImageVAE checkpoint (final.pth). "
                        "Omit to skip image latent caching.")
    p.add_argument("--mask_vae_ckpt",  default=None,
                   help="Path to MaskVAE checkpoint (final.pth). "
                        "Omit to skip mask latent caching.")
    p.add_argument("--batch_size",     type=int, default=4,
                   help="Cases per GPU batch. 4-8 is fine at 128³ on H100.")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    src        = Path(args.data_root)
    dst        = Path(args.output_dir)
    splits_dir = Path(args.splits_dir) if args.splits_dir else src
    device     = torch.device(args.device)

    if args.image_vae_ckpt is None and args.mask_vae_ckpt is None:
        raise ValueError("Provide at least one of --image_vae_ckpt / --mask_vae_ckpt")

    # --- Prepare output directories ---
    z_img_dir  = dst / "z_image"
    z_mask_dir = dst / "z_mask"
    if args.image_vae_ckpt:
        z_img_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_vae_ckpt:
        z_mask_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    image_vae:    ImageVAE | None = None
    mask_vae:     MaskVAE  | None = None
    mask_subregion: bool = False
    if args.image_vae_ckpt:
        print(f"Loading ImageVAE  from {args.image_vae_ckpt}")
        image_vae = _load_image_vae(args.image_vae_ckpt, device)
        print(f"  embed_dim = {image_vae.embed_dim}")
    if args.mask_vae_ckpt:
        print(f"Loading MaskVAE   from {args.mask_vae_ckpt}")
        mask_vae, mask_subregion = _load_mask_vae(args.mask_vae_ckpt, device)
        mode_str = "subregion [BG,NCR,ED,ET]" if mask_subregion else "region [WT,TC,ET]"
        print(f"  latent_channels = {mask_vae.latent_channels}  mode = {mode_str}")

    # --- Discover all cases across all splits ---
    all_cases: list[str] = sorted(
        p.name.replace("_vol.npy", "")
        for p in (src / "vol").glob("*_vol.npy")
    )
    if not all_cases:
        raise FileNotFoundError(f"No *_vol.npy files found in {src / 'vol'}")
    print(f"\nFound {len(all_cases)} cases in {src / 'vol'}")

    # --- Check which cases still need caching (skip already done) ---
    def _needs_cache(name: str) -> bool:
        if image_vae and not (z_img_dir / f"{name}_z_img.npy").exists():
            return True
        if mask_vae and not (z_mask_dir / f"{name}_z_mask.npy").exists():
            return True
        return False

    pending = [n for n in all_cases if _needs_cache(n)]
    skipped = len(all_cases) - len(pending)
    if skipped:
        print(f"  {skipped} already cached — skipping")
    print(f"  {len(pending)} to process\n")

    # --- Process in batches ---
    use_amp = (device.type == "cuda")
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="caching latents", unit="batch"):
            batch_names = pending[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

            # Load volumes: (H,W,D,4) channel-last → (4,H,W,D) channel-first
            vols = []
            masks = []
            for name in batch_names:
                vol = np.load(src / "vol" / f"{name}_vol.npy")   # (H,W,D,4)
                vol = vol.transpose(3, 0, 1, 2).astype(np.float32)  # (4,H,W,D)
                vols.append(vol)

                if mask_vae:
                    seg = np.load(src / "seg" / f"{name}_seg.npy")   # (H,W,D)
                    seg_fn = seg_to_subregions if mask_subregion else seg_to_regions
                    masks.append(seg_fn(seg))                         # (C,H,W,D) float32

            vol_batch = torch.from_numpy(np.stack(vols)).to(device)   # (B,4,H,W,D)

            # --- Image latents ---
            if image_vae:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    mu_img, _ = image_vae.encode(vol_batch)           # (B, embed_dim, 16,16,16)
                mu_img = mu_img.float().cpu().numpy()
                for i, name in enumerate(batch_names):
                    np.save(z_img_dir / f"{name}_z_img.npy", mu_img[i])

            # --- Mask latents ---
            if mask_vae:
                mask_batch = torch.from_numpy(np.stack(masks)).to(device)  # (B,3,H,W,D)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    mu_mask, _ = mask_vae.encode(mask_batch)          # (B, latent_ch, 16,16,16)
                mu_mask = mu_mask.float().cpu().numpy()
                for i, name in enumerate(batch_names):
                    np.save(z_mask_dir / f"{name}_z_mask.npy", mu_mask[i])

    # --- Copy split files ---
    for split in ("train.txt", "val.txt", "test.txt"):
        src_split = splits_dir / split
        if src_split.exists():
            shutil.copy(src_split, dst / split)

    # --- Summary ---
    print(f"\n{'='*55}")
    if image_vae:
        sample = np.load(z_img_dir / f"{all_cases[0]}_z_img.npy")
        print(f"Image latents  : {sample.shape}  → {z_img_dir}/")
    if mask_vae:
        sample = np.load(z_mask_dir / f"{all_cases[0]}_z_mask.npy")
        print(f"Mask latents   : {sample.shape}  → {z_mask_dir}/")
    print(f"Cases cached   : {len(all_cases)}")
    print(f"Output         : {dst}/")


if __name__ == "__main__":
    main()
