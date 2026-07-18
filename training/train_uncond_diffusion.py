"""
Step 1: Unconditional 3D diffusion model on BraTS segmentation masks.

No image conditioning — purely learns the distribution of BraTS mask shapes.
This is a sanity check: if generated masks look vaguely tumor-shaped after
training, the diffusion backbone works and we move to Step 2 (image conditioning).

Expected: Dice will be poor (no conditioning). That is correct and expected.
Training loss should decrease and generated masks should stop looking like
random noise.

Run from repo root:
    python3 -m training.train_uncond_diffusion \\
        --data_root data/brats2023_processed \\
        --output_dir runs/uncond_diffusion
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

from data.brats_dataset import BraTSDataset

# BraTS label → (name, RGB colour for visualisation)
LABEL_COLOURS = {0: (0, 0, 0), 1: (0, 0, 200), 2: (0, 200, 0), 3: (200, 0, 0)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True,
                   help="Processed data folder containing vol/ and seg/. Never modified.")
    p.add_argument("--splits_dir", type=str, default=None,
                   help="Folder with train.txt/val.txt/test.txt (from resplit_data.py). "
                        "Defaults to --data_root if not set.")
    p.add_argument("--output_dir", default="runs/uncond_diffusion")

    # Data
    p.add_argument("--split_file", type=str, default="train.txt")
    p.add_argument("--val_split_file", type=str, default="val.txt")
    p.add_argument("--crop_size", type=int, default=64,
                   help="3D crop size in voxels (64 is faster than 96 for sanity check)")

    # Model
    p.add_argument("--base_channels", type=int, default=32,
                   help="U-Net base channel count (keep small for speed)")

    # Training
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Keep at 1-2 on CPU; 4-8 on GPU. 3D volumes are large.")
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=0,
                   help="0 = no multiprocessing (avoids semaphore leaks on macOS CPU runs)")

    # Evaluation / logging
    p.add_argument("--eval_every", type=int, default=5,
                   help="Run evaluation every N epochs")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM steps for fast inference at eval time")
    p.add_argument("--num_vis_cases", type=int, default=3,
                   help="Number of GT cases to visualise alongside generated masks")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(base_channels: int) -> DiffusionModelUNet:
    c = base_channels
    # norm_num_groups must divide every channel width evenly
    norm_groups = min(32, c)
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=4,          # one-hot mask (4 classes)
        out_channels=4,         # predict noise in same space
        channels=(c, c * 2, c * 2),
        attention_levels=(False, True, True),
        num_head_channels=c * 2,
        num_res_blocks=1,
        norm_num_groups=norm_groups,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / (denom + 1e-8))


def brats_dice(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str, float]:
    """
    pred_labels, gt_labels: (D, H, W) integer arrays, values in {0,1,2,3}.

    BraTS tumour regions:
      WT (Whole Tumour)    = labels 1+2+3
      TC (Tumour Core)     = labels 1+3
      ET (Enhancing Tumour)= label  3
    """
    wt = dice_coefficient(pred_labels > 0,               gt_labels > 0)
    tc = dice_coefficient(np.isin(pred_labels, [1, 3]),  np.isin(gt_labels, [1, 3]))
    et = dice_coefficient(pred_labels == 3,               gt_labels == 3)
    return {"WT": wt, "TC": tc, "ET": et}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    """(H, W) int labels → (H, W, 3) uint8 RGB."""
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for label, colour in LABEL_COLOURS.items():
        rgb[label_map == label] = colour
    return rgb


def save_visualisation(
    generated: np.ndarray,
    gt_cases: list[np.ndarray],
    epoch: int,
    out_dir: Path,
) -> None:
    """
    Save a PNG showing the middle axial slice of:
      - The generated mask (top row)
      - N GT cases from validation set (subsequent rows)

    generated: (D, H, W) int label array
    gt_cases:  list of (D, H, W) int label arrays
    """
    rows = 1 + len(gt_cases)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    views = [
        ("Axial (mid-Z)",    generated[generated.shape[0] // 2]),
        ("Coronal (mid-Y)",  generated[:, generated.shape[1] // 2, :]),
        ("Sagittal (mid-X)", generated[:, :, generated.shape[2] // 2]),
    ]

    for col, (title, sl) in enumerate(views):
        axes[0, col].imshow(labels_to_rgb(sl), interpolation="nearest")
        axes[0, col].set_title(f"Generated — {title}", fontsize=8)
        axes[0, col].axis("off")

    for row, gt in enumerate(gt_cases, start=1):
        gt_views = [
            gt[gt.shape[0] // 2],
            gt[:, gt.shape[1] // 2, :],
            gt[:, :, gt.shape[2] // 2],
        ]
        for col, sl in enumerate(gt_views):
            axes[row, col].imshow(labels_to_rgb(sl), interpolation="nearest")
            axes[row, col].set_title(f"GT case {row} — {['Axial','Coronal','Sagittal'][col]}", fontsize=8)
            axes[row, col].axis("off")

    plt.suptitle(f"Epoch {epoch} — colours: black=BG, blue=NCR, green=ED, red=ET", fontsize=9)
    plt.tight_layout()
    path = out_dir / f"vis_epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualisation → {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DiffusionModelUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False, unit="batch")
    for _volume, mask_onehot in pbar:
        mask_onehot = mask_onehot.to(device)          # (B, 4, D, H, W)

        t = torch.randint(0, scheduler.num_train_timesteps, (mask_onehot.shape[0],), device=device)
        noise = torch.randn_like(mask_onehot)
        noisy = scheduler.add_noise(original_samples=mask_onehot, noise=noise, timesteps=t)

        predicted_noise = model(noisy, timesteps=t)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def generate_one_mask(
    model: DiffusionModelUNet,
    inf_scheduler: DDIMScheduler,
    crop_size: int,
    device: torch.device,
) -> np.ndarray:
    """Full reverse diffusion → hard label map (D, H, W) int."""
    model.eval()
    z = torch.randn(1, 4, crop_size, crop_size, crop_size, device=device)

    for t in inf_scheduler.timesteps:
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        predicted_noise = model(z, timesteps=t_batch)
        z = inf_scheduler.step(predicted_noise, t, z)[0]

    # z: (1, 4, D, H, W) continuous → hard labels via argmax
    return z[0].argmax(dim=0).cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualisations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    splits_dir = args.splits_dir if args.splits_dir is not None else args.data_root

    # ---- Dataset (seg only; ignore volume for unconditional training) ----
    import os
    train_dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.split_file),
        crop_size=args.crop_size,
    )

    val_dataset = BraTSDataset(
        root=args.data_root,
        split_file=os.path.join(splits_dir, args.val_split_file),
        crop_size=args.crop_size,
        random_crop=False,  # centre crop for consistent visualisation
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print(f"Training cases  : {len(train_dataset)}")
    print(f"Validation cases: {len(val_dataset)}")
    print(f"Crop size       : {args.crop_size}³")
    print(f"Device          : {device}")

    # ---- Model & schedulers ----
    model = build_model(args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net params    : {n_params:.1f}M")

    train_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    inf_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
    inf_scheduler.set_timesteps(args.num_inference_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- CSV log ----
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "dice_WT", "dice_TC", "dice_ET"])

    # ---- Grab a few fixed GT cases for visualisation ----
    gt_cases = []
    for i in range(min(args.num_vis_cases, len(val_dataset))):
        _vol, mask_onehot = val_dataset[i]
        gt_labels = mask_onehot.argmax(dim=0).numpy().astype(np.int32)
        gt_cases.append(gt_labels)

    # ---- Training loop ----
    best_loss = float("inf")
    epoch_bar = tqdm(range(1, args.num_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        loss = train_one_epoch(model, loader, optimizer, train_scheduler, device, epoch, args.num_epochs)

        if loss < best_loss:
            best_loss = loss
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "loss": loss},
                       out_dir / "best.pth")

        epoch_bar.set_postfix(loss=f"{loss:.4f}", best=f"{best_loss:.4f}")
        tqdm.write(f"Epoch {epoch:3d}/{args.num_epochs}  loss={loss:.4f}  (best={best_loss:.4f})")

        # ---- Evaluation ----
        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            tqdm.write(f"  [eval] generating 1 mask with DDIM ({args.num_inference_steps} steps)...")
            gen_labels = generate_one_mask(model, inf_scheduler, args.crop_size, device)

            # Dice against a random val GT (pairing is arbitrary — scores will be low)
            gt_ref = gt_cases[0] if gt_cases else np.zeros_like(gen_labels)
            metrics = brats_dice(gen_labels, gt_ref)
            tqdm.write(f"  [eval] Dice  WT={metrics['WT']:.3f}  TC={metrics['TC']:.3f}  ET={metrics['ET']:.3f}"
                       f"  (low is expected — no conditioning)")

            save_visualisation(gen_labels, gt_cases, epoch, vis_dir)

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{loss:.4f}",
                                 f"{metrics['WT']:.4f}",
                                 f"{metrics['TC']:.4f}",
                                 f"{metrics['ET']:.4f}"])
        else:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{loss:.4f}", "", "", ""])

    # ---- Final checkpoint ----
    torch.save({"model_state_dict": model.state_dict(),
                "epoch": args.num_epochs, "loss": loss},
               out_dir / "final.pth")
    print(f"\nDone. Outputs in {out_dir}")
    print(f"  Log      : {log_path}")
    print(f"  Best ckpt: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
