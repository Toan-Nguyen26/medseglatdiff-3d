"""
Conditioned latent diffusion for BraTS segmentation.

Follows MedSegLatDiff (Huynh et al., arXiv:2512.01292) training procedure:

  Training (DDPM forward process on mask space):
    1. Encode image → condition c  (image VAE or encoder)
    2. Sample t ~ Uniform(0, T),  ε ~ N(0, I)
    3. Add noise to GT mask:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    4. Predict noise:  ε_pred = UNet(x_t, t, c)
    5. Loss = MSE(ε_pred, ε)

  Inference (DDIM reverse process, N samples):
    for each of N samples:
      x_T ~ N(0, I)
      for t = T..0:  x_{t-1} = DDIM_step(UNet(x_t, t, c), t, x_t)
      mask = sigmoid(x_0) > 0.5   ← x_0 in logit space, sigmoid inverts it
    final = mean(masks) > 0.5   (ensemble)

  N different starting noises → N different masks → genuine sample diversity.

Run from repo root:
    python3 -m training.train_pixel_test --data_root data/brats2023_processed
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

from data.brats_dataset import BraTSDataset, apply_modality_mask, sample_modality_mask, regions_to_seg
from models.diffusion.config import DiffusionConfig
from models.diffusion.unet3d import UNet3D
from models.multiencoder.encoders import ImageEncoder, ImageVAE, MaskVAE, vae_loss, reconstruction_loss
from utils.run_logger import RunLogger, new_run_id, set_seed

STAGE = "pixel_test"

# Reconstructed label map -> RGB colour (labels from regions_to_seg: 0,1,2,4)
LABEL_COLOURS = {0: (0, 0, 0), 1: (0, 0, 200), 2: (0, 200, 0), 4: (200, 0, 0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data_root", type=str, required=True,
                        help="Processed data folder containing vol/ and seg/. Never modified.")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Folder with train.txt/val.txt/test.txt (from resplit_data.py). "
                             "Defaults to --data_root if not set.")
    parser.add_argument("--split_file", type=str, default="train.txt")
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=3,
                        help="3 sigmoid outputs: WT, TC, ET (region-based)")

    parser.add_argument("--encoder_channels", type=str, default="64,128,256,256",
                        help="Comma-separated channel progression for ImageVAE, e.g. '64,128,256,256'. "
                             "Must match the checkpoint if --use_vae is set.")
    parser.add_argument("--encoder_num_res_units", type=int, default=2)
    parser.add_argument("--use_vae", action="store_true",
                        help="Use ImageVAE (variational) instead of ImageEncoder (deterministic). "
                             "Requires --image_vae_checkpoint pointing to a train_image_vae.py output.")
    parser.add_argument("--image_vae_checkpoint", type=str, default=None,
                        help="Path to a pretrained ImageVAE checkpoint from train_image_vae.py. "
                             "Required when --use_vae is set.")
    parser.add_argument("--freeze_vae", action="store_true",
                        help="Freeze the VAE encoder during segmentation training. "
                             "Recommended when the VAE is already well-pretrained.")
    parser.add_argument("--vae_beta", type=float, default=1e-4,
                        help="Weight on KL term when VAE is unfrozen and reconstruction loss is active.")
    parser.add_argument("--recon_weight", type=float, default=0.0,
                        help="Weight for auxiliary reconstruction loss. "
                             "Set to 0 (default) when using a pretrained VAE — "
                             "reconstruction is already handled in train_image_vae.py.")

    # Spatial patch masking (MAE-style, applied to present modalities only)
    parser.add_argument("--patch_mask_ratio", type=float, default=0.0,
                        help="Fraction of 8^3 spatial patches to mask within present modalities. "
                             "0.0 disables patch masking entirely. Try 0.5 to match UniME.")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size in voxels (must divide crop_size evenly). "
                             "Default 8 → 12^3 patch grid matching the encoder bottleneck.")

    parser.add_argument("--cond_proj_channels", type=int, default=64)
    parser.add_argument("--unet_base_channels", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="DDIM steps at validation/vis time (fast approximation of 1000-step DDPM).")
    parser.add_argument("--diffusion", action="store_true",
                        help="Enable proper DDPM training: noise the GT mask at timestep t and "
                             "train UNet to predict the added noise (MSE). Inference uses DDIM "
                             "denoising loop → genuine sample diversity. "
                             "Without this flag (default): UNet(random_noise, t=0, c) → mask "
                             "directly with Dice+BCE loss. Use this default until a mask VAE "
                             "is available to provide a proper latent space to denoise in.")
    parser.add_argument("--mask_vae_checkpoint", type=str, default=None,
                        help="Path to a pretrained MaskVAE checkpoint from train_mask_vae.py. "
                             "When provided, the diffusion U-Net denoises mask latents (4ch, 12³) "
                             "instead of logit-transformed masks (3ch, 12³). Requires --diffusion.")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)

    # Visualisation
    parser.add_argument("--val_split_file", type=str, default="val.txt")
    parser.add_argument("--vis_every", type=int, default=500,
                        help="Run prediction visualisation every N steps")
    parser.add_argument("--num_vis_cases", type=int, default=3,
                        help="Number of fixed val cases to visualise")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seg_loss(pred: torch.Tensor, gt_regions: torch.Tensor) -> torch.Tensor:
    """
    Dice + BCE for direct mask prediction (default / no --diffusion).
    pred:       (B, 3, *spatial) — raw logits
    gt_regions: (B, 3, *spatial) — binary float masks [WT, TC, ET]
    """
    bce = F.binary_cross_entropy_with_logits(pred, gt_regions)
    pred_sig = pred.sigmoid()
    eps = 1e-5
    tp = (pred_sig * gt_regions).sum(dim=list(range(2, pred.ndim)))
    fp = (pred_sig * (1 - gt_regions)).sum(dim=list(range(2, pred.ndim)))
    fn = ((1 - pred_sig) * gt_regions).sum(dim=list(range(2, pred.ndim)))
    dice = (1 - (2 * tp + eps) / (2 * tp + fp + fn + eps)).mean()
    return bce + dice


def diffusion_loss(pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """
    MSE between predicted and actual noise — standard DDPM objective (--diffusion).
    Matches MedSegLatDiff eq: L = E[||ε - ε_θ(x_t, t)||²]
    """
    return F.mse_loss(pred_noise, target_noise)


def dice_score(pred_x0: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5,
               use_diffusion: bool = False, eps: float = 1e-6) -> dict[str, float]:
    """
    Compute Dice for each of the 3 regions (WT, TC, ET).

    pred_x0: (1, 3, *spatial) — denoised x_0 (diffusion) or raw logits (non-diffusion)
    gt:      (1, 3, *spatial) — binary float masks [WT, TC, ET]
    """
    pred = (pred_x0.sigmoid() > threshold).float()  # both paths: logit space → sigmoid
    names = ["WT", "TC", "ET"]
    scores = {}
    for i, name in enumerate(names):
        p = pred[0, i].flatten()
        g = gt[0, i].flatten()
        intersection = (p * g).sum()
        scores[name] = float((2 * intersection + eps) / (p.sum() + g.sum() + eps))
    scores["mean"] = float(sum(scores[v] for v in names) / 3)
    return scores


def apply_patch_mask(
    volume: torch.Tensor,
    patch_size: int = 8,
    mask_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly zero out spatial patches within present (non-zero) modality channels.
    Missing modality channels (all zeros) are left untouched.

    Args:
        volume:     (B, 4, D, H, W) — already modality-masked (absent channels = 0)
        patch_size: voxels per patch side. Default 8 → 12^3 patch grid at 96^3 input.
        mask_ratio: fraction of patches to zero out within each present channel.

    Returns:
        volume_out:  (B, 4, D, H, W) — volume with patches zeroed
        patch_mask:  (B, 1, D, H, W) — 1 = kept, 0 = masked (for loss computation)
    """
    B, C, D, H, W = volume.shape
    gD, gH, gW = D // patch_size, H // patch_size, W // patch_size

    # random keep mask at patch grid resolution — same mask across all channels
    keep = (torch.rand(B, 1, gD, gH, gW, device=volume.device) > mask_ratio).float()

    # upsample to voxel resolution — each patch becomes a patch_size^3 block
    patch_mask = F.interpolate(keep, size=(D, H, W), mode="nearest")  # (B, 1, D, H, W)

    # only mask patches that belong to a present modality
    # present_mask: (B, 1, D, H, W) — 1 wherever at least one channel has signal
    present_mask = (volume.abs().sum(dim=1, keepdim=True) > 0).float()
    patch_mask = patch_mask * present_mask + (1 - present_mask)  # always keep absent-channel regions

    return volume * patch_mask, patch_mask


def build_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[ImageEncoder | ImageVAE, UNet3D, MaskVAE | None]:
    encoder_channels = tuple(int(c) for c in args.encoder_channels.split(","))
    embed_dim = encoder_channels[-1]

    if args.use_vae:
        if args.image_vae_checkpoint is None:
            raise ValueError("--image_vae_checkpoint is required when --use_vae is set.")
        image_encoder = ImageVAE(
            in_channels=4,
            channels=encoder_channels,
            num_res_units=args.encoder_num_res_units,
        ).to(device)
        ckpt = torch.load(args.image_vae_checkpoint, map_location=device, weights_only=True)
        image_encoder.load_state_dict(ckpt["vae_state_dict"])
        print(f"Loaded ImageVAE from {args.image_vae_checkpoint}")
        if args.freeze_vae:
            for p in image_encoder.parameters():
                p.requires_grad = False
            image_encoder.eval()
            print("ImageVAE frozen — only UNet will be trained.")
    else:
        image_encoder = ImageEncoder(
            in_channels=4,
            embed_dim=embed_dim,
            num_res_units=args.encoder_num_res_units,
        ).to(device)

    # --- MaskVAE (optional): provides learned latent space for diffusion ---
    mask_vae = None
    latent_channels = args.num_classes  # default: 3 channels (logit-transform path)
    if args.mask_vae_checkpoint is not None:
        vae_ckpt = torch.load(args.mask_vae_checkpoint, map_location=device, weights_only=True)
        vae_config = vae_ckpt["config"]
        channels = tuple(int(c) for c in vae_config["mask_vae_channels"].split(","))
        mask_vae = MaskVAE(
            num_classes=vae_config.get("num_classes", 3),
            latent_channels=vae_config["latent_channels"],
            channels=channels,
            num_res_units=vae_config["num_res_units"],
        ).to(device)
        mask_vae.load_state_dict(vae_ckpt["vae_state_dict"])
        mask_vae.eval()
        for p in mask_vae.parameters():
            p.requires_grad_(False)
        latent_channels = mask_vae.latent_channels
        print(f"Loaded MaskVAE from {args.mask_vae_checkpoint}  (latent_channels={latent_channels})")

    diffusion_config = DiffusionConfig(
        latent_channels=latent_channels,
        condition_channels=embed_dim,
        cond_proj_channels=args.cond_proj_channels,
        base_channels=args.unet_base_channels,
        num_timesteps=args.num_timesteps,
    )
    unet = UNet3D(diffusion_config).to(device)
    return image_encoder, unet, mask_vae


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """(H, W) int labels → (H, W, 3) uint8 RGB."""
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label, colour in LABEL_COLOURS.items():
        rgb[label_map == label] = colour
    return rgb


def best_tumour_slices(gt: np.ndarray) -> tuple[int, int, int]:
    """
    Return (z, y, x) indices of the slice with the most tumour voxels on each axis.
    Falls back to the geometric mid-slice if no tumour is present.
    """
    tumour = (gt != 0)
    def best(axis: int) -> int:
        counts = tumour.sum(axis=tuple(a for a in range(gt.ndim) if a != axis))
        return int(counts.argmax()) if counts.max() > 0 else gt.shape[axis] // 2
    return best(0), best(1), best(2)


def save_prediction_visualisation(
    cases: list[tuple[np.ndarray, np.ndarray]],
    step: int,
    out_dir: Path,
) -> None:
    """
    Save a PNG with one (Predicted, GT) row-pair per case, 3 anatomical views each.
    Slices are chosen at the position with the most tumour voxels in the GT
    (not the geometric centre), so the tumour is always visible.

    cases: list of (pred_labels, gt_labels), each (D, H, W) int label array.
    """
    rows = 2 * len(cases)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, (pred, gt) in enumerate(cases):
        z, y, x = best_tumour_slices(gt)
        view_names = [f"Axial z={z}", f"Coronal y={y}", f"Sagittal x={x}"]

        for row_offset, (label_arr, tag) in enumerate([(pred, "Pred"), (gt, "GT")]):
            row = 2 * i + row_offset
            views = [
                label_arr[z],
                label_arr[:, y, :],
                label_arr[:, :, x],
            ]
            for col, (title, sl) in enumerate(zip(view_names, views)):
                axes[row, col].imshow(labels_to_rgb(sl), interpolation="nearest")
                axes[row, col].set_title(f"Case {i + 1} {tag} — {title}", fontsize=8)
                axes[row, col].axis("off")

    plt.suptitle(f"Step {step} — colours: black=BG, blue=NCR, green=ED, red=ET", fontsize=9)
    plt.tight_layout()
    path = out_dir / f"vis_step_{step:06d}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    tqdm.write(f"  [vis] Saved visualisation → {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    image_encoder, unet, mask_vae = build_models(args, device)

    trainable = [p for p in image_encoder.parameters() if p.requires_grad]
    trainable += list(unet.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    splits_dir = args.splits_dir if args.splits_dir is not None else args.data_root

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        image_encoder.load_state_dict(ckpt["encoder_state_dict"])
        unet.load_state_dict(ckpt["unet_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]

    dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.split_file),
        crop_size=args.crop_size,
        region_based=True,   # returns (3, H, W, D) binary masks: [WT, TC, ET]
    )
    print(f"Training cases : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    latent_spatial = image_encoder.output_spatial_shape(args.crop_size)  # (12, 12, 12)

    # DDPM scheduler — used in training forward process
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    run_id = new_run_id(STAGE)
    logger = RunLogger(run_id=run_id, stage=STAGE, config=vars(args))
    checkpoint_dir = Path(args.checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = checkpoint_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)

    # ---- Fixed val cases for periodic prediction visualisation ----
    vis_cases: list[tuple[torch.Tensor, np.ndarray]] = []
    val_split_path = os.path.join(splits_dir, args.val_split_file)
    if os.path.exists(val_split_path):
        val_dataset = BraTSDataset(
            root=args.data_root,
            split_file=val_split_path,
            crop_size=args.crop_size,
            region_based=True,
            random_crop=False,  # centre crop for consistent visualisation
        )
        for i in range(min(args.num_vis_cases, len(val_dataset))):
            vol, gt_regions = val_dataset[i]   # gt_regions: (3, H, W, D) binary [WT, TC, ET]
            gt_labels = regions_to_seg(
                gt_regions[0].numpy(), gt_regions[1].numpy(), gt_regions[2].numpy()
            )                                  # → (H, W, D) with values {0,1,2,4}
            vis_cases.append((vol.unsqueeze(0), gt_labels))
    else:
        print(f"  [warn] val split file not found at {val_split_path}, skipping visualisation")

    image_encoder.train()
    unet.train()
    step = start_step

    steps_per_epoch = len(loader)
    start_epoch = start_step // max(1, steps_per_epoch)
    total_steps = args.num_epochs * steps_per_epoch

    pbar = tqdm(total=total_steps, initial=start_step, desc=f"[{STAGE}]", unit="step")
    for epoch in range(start_epoch, args.num_epochs):
        tqdm.write(f"\n[{STAGE}] Epoch {epoch + 1}/{args.num_epochs}  (steps per epoch: {steps_per_epoch})")
        for volume, mask_onehot in loader:

            volume = volume.to(device)             # (B, 4, 96, 96, 96)
            gt_regions = mask_onehot.to(device)   # (B, 3, 96, 96, 96) binary [WT, TC, ET] — GT, not used as input

            # Simulate missing modalities
            modality_mask = sample_modality_mask(volume.shape[0]).to(device)
            volume_masked = apply_modality_mask(volume, modality_mask)

            # Spatial patch masking within present modalities (MAE-style)
            if args.patch_mask_ratio > 0.0:
                volume_encoder_input, patch_mask = apply_patch_mask(
                    volume_masked, args.patch_size, args.patch_mask_ratio
                )
            else:
                volume_encoder_input = volume_masked
                patch_mask = None

            # --- CONDITION: encode image → c ---
            if args.use_vae:
                mu, logvar = image_encoder.encode(volume_encoder_input)
                c = image_encoder.reparameterize(mu, logvar)   # (B, 256, 12, 12, 12)
                recon = image_encoder.decode(c)
                if patch_mask is not None:
                    masked_region = (1 - patch_mask)
                    recon = recon * masked_region
                    target = volume_masked * masked_region
                else:
                    target = volume_masked
                _, recon_loss, kl = vae_loss(recon, target, mu, logvar, beta=args.vae_beta)
                recon_loss = recon_loss + kl   # already weighted by beta inside vae_loss
            else:
                c = image_encoder.encode(volume_encoder_input) # (B, 256, 12, 12, 12)
                recon = image_encoder.decode(c)
                if patch_mask is not None:
                    masked_region = (1 - patch_mask)
                    recon_loss = reconstruction_loss(recon * masked_region, volume_masked * masked_region)
                else:
                    recon_loss = reconstruction_loss(recon, volume_masked)

            if args.diffusion:
                # --- DDPM FORWARD PROCESS (MedSegLatDiff §3) ---
                if mask_vae is not None:
                    # MaskVAE path: encode GT mask → latent z ∼ N(0,1) — no logit needed.
                    # mu is a stable deterministic x_0 (mask_vae is frozen + eval).
                    with torch.no_grad():
                        x_0, _ = mask_vae.encode(gt_regions)   # (B, lat_ch, 12, 12, 12)
                else:
                    # Logit-transform path: gt_down ≈ {0,1} → x_0 ≈ {-8,+8}.
                    gt_down = F.interpolate(
                        gt_regions, size=latent_spatial, mode="trilinear", align_corners=False
                    )
                    _c = 1e-4
                    x_0 = torch.log(
                        gt_down.clamp(_c, 1 - _c) / (1 - gt_down.clamp(_c, 1 - _c))
                    )
                epsilon = torch.randn_like(x_0)
                t = torch.randint(0, args.num_timesteps, (volume.shape[0],), device=device)
                x_t = noise_scheduler.add_noise(original_samples=x_0, noise=epsilon, timesteps=t)
                pred_noise = unet(x_t, t, c)
                mask_loss = diffusion_loss(pred_noise, epsilon)
            else:
                # --- DEFAULT: direct mask prediction from pure noise ---
                gt_down = F.interpolate(
                    gt_regions, size=latent_spatial, mode="trilinear", align_corners=False
                )
                noise = torch.randn_like(gt_down)
                t = torch.zeros(volume.shape[0], dtype=torch.long, device=device)
                pred = unet(noise, t, c)
                mask_loss = seg_loss(pred, gt_down)

            loss = mask_loss + args.recon_weight * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_label = "noise" if args.diffusion else "seg"
            pbar.set_postfix(
                epoch=f"{epoch + 1}/{args.num_epochs}",
                loss=f"{loss.item():.4f}",
                **{loss_label: f"{mask_loss.item():.4f}"},
                recon=f"{recon_loss.item():.4f}",
            )

            if step % args.log_every == 0:
                logger.log_metrics(
                    step=step,
                    csv="train",
                    epoch=epoch + 1,
                    train_loss=loss.item(),
                    mask_loss=mask_loss.item(),
                    recon_loss=recon_loss.item(),
                )
                tqdm.write(
                    f"[{STAGE}] epoch={epoch + 1}  step={step}  "
                    f"loss={loss.item():.4f}  "
                    f"{'noise' if args.diffusion else 'seg'}={mask_loss.item():.4f}  "
                    f"recon={recon_loss.item():.4f}"
                )

            if step % args.vis_every == 0 and step > start_step and vis_cases:
                tqdm.write(f"  [vis] running validation on {len(vis_cases)} case(s)...")
                image_encoder.eval()
                unet.eval()
                rendered = []
                all_dice: dict[str, list[float]] = {"WT": [], "TC": [], "ET": [], "mean": []}

                with torch.no_grad():
                    for vol, gt_labels in vis_cases:
                        vol_dev = vol.to(device)                              # (1, 4, H, W, D)
                        if args.use_vae:
                            mu, _ = image_encoder.encode(vol_dev)
                            c_val = mu                                        # deterministic at val
                        else:
                            c_val = image_encoder.encode(vol_dev)
                        latent_sp = c_val.shape[2:]

                        # Predict mask — DDIM if --diffusion, else single forward pass
                        lat_ch = mask_vae.latent_channels if mask_vae is not None else args.num_classes
                        if args.diffusion:
                            x_0 = ddim_denoise(unet, c_val, lat_ch,
                                               latent_sp, args.num_inference_steps,
                                               args.num_timesteps, device)
                        else:
                            noise = torch.randn(1, lat_ch, *latent_sp, device=device)
                            t_zero = torch.zeros(1, dtype=torch.long, device=device)
                            x_0 = unet(noise, t_zero, c_val)

                        gt_dev = torch.from_numpy(
                            np.stack([
                                (gt_labels > 0).astype(np.float32),          # WT
                                ((gt_labels == 1) | (gt_labels == 4)).astype(np.float32),  # TC
                                (gt_labels == 4).astype(np.float32),         # ET
                            ])
                        ).unsqueeze(0).to(device)                             # (1, 3, H, W, D)

                        # Dice: MaskVAE → full-res decode; logit path → latent-res
                        if mask_vae is not None:
                            pred_logits = mask_vae.decode(x_0)   # (1, 3, H, W, D)
                            gt_for_dice = gt_dev
                        else:
                            pred_logits = x_0                    # (1, 3, *latent_sp)
                            gt_for_dice = F.interpolate(gt_dev, size=latent_sp,
                                                        mode="trilinear", align_corners=False)
                        d = dice_score(pred_logits, gt_for_dice)
                        for k, v in d.items():
                            all_dice[k].append(v)

                        # Visualisation — full-res inference
                        pred_labels = run_inference(
                            image_encoder, unet, vol_dev, args.num_classes,
                            args.num_inference_steps, args.num_timesteps,
                            device, use_diffusion=args.diffusion,
                            mask_vae=mask_vae,
                        )
                        rendered.append((pred_labels[0].cpu().numpy().astype(np.int32), gt_labels))

                mean_dice = {k: float(np.mean(v)) for k, v in all_dice.items()}
                tqdm.write(
                    f"  [val]  Dice — WT={mean_dice['WT']:.3f}  "
                    f"TC={mean_dice['TC']:.3f}  ET={mean_dice['ET']:.3f}  "
                    f"mean={mean_dice['mean']:.3f}"
                )
                logger.log_metrics(step=step, csv="val", **{f"dice_{k}": v for k, v in mean_dice.items()})
                save_prediction_visualisation(rendered, step, vis_dir)
                image_encoder.train()
                unet.train()

            if step % args.ckpt_every == 0 and step > start_step:
                ckpt_path = checkpoint_dir / f"step_{step}.pth"
                torch.save({
                    "encoder_state_dict": image_encoder.state_dict(),
                    "unet_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": vars(args),
                }, ckpt_path)

            step += 1
            pbar.update(1)

    pbar.close()
    final_ckpt_path = checkpoint_dir / "final.pth"
    torch.save({
        "encoder_state_dict": image_encoder.state_dict(),
        "unet_state_dict": unet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": args.num_epochs,
        "config": vars(args),
    }, final_ckpt_path)
    logger.append_to_experiments_index(
        f"pixel-space test, {args.num_epochs} epochs ({step} steps), "
        f"channels={args.encoder_channels}, final_loss={loss.item():.4f}"
    )


@torch.no_grad()
def ddim_denoise(
    unet: UNet3D,
    c: torch.Tensor,
    num_classes: int,
    latent_spatial: tuple,
    num_inference_steps: int,
    num_train_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    DDIM reverse process: x_T ~ N(0,I) → x_0 via num_inference_steps denoising steps.
    Each call with a different random seed produces a different mask (genuine diversity).

    Returns x_0: (1, num_classes, *latent_spatial)
    """
    inf_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    inf_scheduler.set_timesteps(num_inference_steps)

    x = torch.randn(1, num_classes, *latent_spatial, device=device)  # x_T
    for t in inf_scheduler.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        pred_noise = unet(x, t_batch, c)
        x = inf_scheduler.step(pred_noise, t, x)[0].to(device).float()  # x_{t-1}
    return x                                                       # x_0


@torch.no_grad()
def run_inference(
    image_encoder: ImageEncoder,
    unet: UNet3D,
    volume: torch.Tensor,
    num_classes: int,
    num_inference_steps: int,
    num_train_timesteps: int,
    device: torch.device,
    use_diffusion: bool = False,
    mask_vae: MaskVAE | None = None,
) -> torch.Tensor:
    """
    Produce one segmentation label map from a volume.

    use_diffusion=True : DDIM denoising loop (--diffusion training mode)
    use_diffusion=False: single forward pass with random noise (default)

    Returns: (1, D, H, W) label map with values {0,1,2,4}
    """
    image_encoder.eval()
    unet.eval()

    full_spatial = volume.shape[2:]
    enc_out = image_encoder.encode(volume)
    c = enc_out[0] if isinstance(enc_out, tuple) else enc_out
    latent_spatial = c.shape[2:]

    lat_ch = mask_vae.latent_channels if mask_vae is not None else num_classes
    if use_diffusion:
        x_0 = ddim_denoise(unet, c, lat_ch, latent_spatial,
                            num_inference_steps, num_train_timesteps, device)
    else:
        noise = torch.randn(1, lat_ch, *latent_spatial, device=device)
        t_zero = torch.zeros(1, dtype=torch.long, device=device)
        x_0 = unet(noise, t_zero, c)

    if mask_vae is not None:
        pred_probs = mask_vae.decode(x_0).sigmoid()  # (1, 3, D, H, W) full-res
    else:
        pred_full = F.interpolate(x_0, size=full_spatial, mode="trilinear", align_corners=False)
        pred_probs = pred_full.sigmoid()

    seg = regions_to_seg(
        pred_probs[0, 0].cpu().numpy(),
        pred_probs[0, 1].cpu().numpy(),
        pred_probs[0, 2].cpu().numpy(),
    )
    return torch.from_numpy(seg).unsqueeze(0)                    # (1, D, H, W)


if __name__ == "__main__":
    main()
