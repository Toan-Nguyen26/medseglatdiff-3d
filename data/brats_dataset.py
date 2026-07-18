"""
BraTS dataset loader, matching the exact on-disk format produced by UniME's
own preprocessing script (scripts/brats23_process.py in Hooorace-S/UniME):

    root/
      vol/{name}_vol.npy   # (H, W, D, 4) float32, channel-last, z-score normalized
      seg/{name}_seg.npy   # (H, W, D) uint8, labels in {0,1,2,3} or raw {0,1,2,4}
      train.txt / val.txt / test.txt   # one case name per line

Two output modes (controlled by region_based flag):

  region_based=False (default):
    mask → one-hot (4, H, W, D), expects labels {0,1,2,3} (UniME convention)

  region_based=True:
    mask → binary regions (3, H, W, D): [WT, TC, ET]
    Works with both raw BraTS labels {0,1,2,4} and remapped {0,1,2,3}.
    This is the correct mode for sigmoid-based training evaluated on BraTS
    overlapping regions (whole tumor, tumor core, enhancing tumor).

Augmentation here is a simple random crop only — UniME's own pipeline has a
fuller augmentation suite (rotation/flip/intensity) in
source/dataset/augmentation.py that we haven't ported. Worth adding later;
not fabricated now since we haven't verified it's needed yet.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# All 15 non-empty modality-availability combinations (2^4 - 1), order
# FLAIR/T1ce/T1/T2 — matches the convention used throughout this project and
# in UniME's own evaluation protocol.
ALL_MODALITY_COMBINATIONS: list[tuple[bool, bool, bool, bool]] = [
    tuple(bool(int(b)) for b in format(combo, "04b"))  # type: ignore[misc]
    for combo in range(1, 16)
]


def sample_modality_mask(batch_size: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random per-sample boolean mask over the 15 non-empty modality combinations."""
    idx = torch.randint(0, len(ALL_MODALITY_COMBINATIONS), (batch_size,), generator=generator)
    combos = torch.tensor(ALL_MODALITY_COMBINATIONS, dtype=torch.bool)
    return combos[idx]


def apply_modality_mask(volume: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
    """
    Zero out missing-modality channels (channel count stays 4 — this isn't a
    subset selection, it's simulating "this modality wasn't acquired").

    Args:
        volume: (B, 4, D, H, W)
        modality_mask: (B, 4) bool, True = modality present
    """
    return volume * modality_mask[:, :, None, None, None].to(dtype=volume.dtype)


def seg_to_regions(label: np.ndarray) -> np.ndarray:
    """
    Convert a BraTS label map to 3 binary region masks [WT, TC, ET].

    Handles both raw BraTS labels {0,1,2,4} and UniME-remapped {0,1,2,3}
    (where 3 or 4 both mean enhancing tumor).

    Returns float32 array of shape (3, H, W, D).
    """
    et_label = 4 if np.any(label == 4) else 3
    WT = (label > 0).astype(np.float32)
    TC = ((label == 1) | (label == et_label)).astype(np.float32)
    ET = (label == et_label).astype(np.float32)
    return np.stack([WT, TC, ET], axis=0)


def seg_to_subregions(label: np.ndarray) -> np.ndarray:
    """
    Convert a BraTS label map to 4 binary subregion masks [BG, NCR, ED, ET].

    These are mutually exclusive in the ground truth — each voxel belongs to
    exactly one class. Using sigmoid on all 4 channels allows soft/uncertain
    predictions at boundaries while giving NCR its own explicit channel.

    Returns float32 array of shape (4, H, W, D).
    """
    et_label = 4 if np.any(label == 4) else 3
    BG  = (label == 0).astype(np.float32)
    NCR = (label == 1).astype(np.float32)
    ED  = (label == 2).astype(np.float32)
    ET  = (label == et_label).astype(np.float32)
    return np.stack([BG, NCR, ED, ET], axis=0)


def subregions_to_regions(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert 4-channel subregion predictions [BG, NCR, ED, ET] back to
    overlapping region masks [WT, TC, ET] for BraTS metric computation.
    """
    NCR = pred[1] > threshold
    ED  = pred[2] > threshold
    ET  = pred[3] > threshold
    WT  = (NCR | ED | ET).astype(np.float32)
    TC  = (NCR | ET).astype(np.float32)
    ET_r = ET.astype(np.float32)
    return np.stack([WT, TC, ET_r], axis=0)


def regions_to_seg(wt: np.ndarray, tc: np.ndarray, et: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert 3 binary region predictions back to a 4-class label map {0,1,2,4}
    for visualization or metric computation.

    Outer regions are written first; inner regions overwrite (ET wins over TC
    wins over WT), which correctly decodes the nested BraTS structure.
    """
    seg = np.zeros_like(wt, dtype=np.uint8)
    seg[wt > threshold] = 2   # edema
    seg[tc > threshold] = 1   # necrosis
    seg[et > threshold] = 4   # enhancing tumor
    return seg


class BraTSDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        *,
        crop_size: int = 96,
        num_classes: int = 4,
        random_crop: bool = True,
        region_based: bool = False,
        subregion_based: bool = False,
        volume_only: bool = False,
        roi_crop_ratio: float = 0.0,
        roi_max_offset: int = 20,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.random_crop = random_crop
        self.region_based = region_based
        self.subregion_based = subregion_based
        self.volume_only = volume_only
        self.roi_crop_ratio = roi_crop_ratio
        self.roi_max_offset = roi_max_offset

        self.names = self._read_names(split_file)
        self.vol_paths = [self.root / "vol" / f"{name}_vol.npy" for name in self.names]
        self.seg_paths = (
            None if volume_only
            else [self.root / "seg" / f"{name}_seg.npy" for name in self.names]
        )

    @staticmethod
    def _read_names(split_file: str | Path) -> list[str]:
        with open(split_file, "r", encoding="utf-8") as f:
            return sorted(line.strip() for line in f if line.strip())

    def __len__(self) -> int:
        return len(self.names)

    def _crop_indices(self, size: int, crop: int) -> tuple[int, int]:
        if size <= crop:
            return 0, size
        if self.random_crop:
            start = int(np.random.randint(0, size - crop + 1))
        else:
            start = (size - crop) // 2
        return start, start + crop

    def _roi_crop_indices(
        self,
        label: np.ndarray,   # (H, W, D)
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Crop 96³ centered on the tumor bounding box centroid + random offset.
        Falls back to random crop when no tumor is present.
        """
        coords = np.where(label > 0)
        if len(coords[0]) == 0:
            h, w, d = label.shape
            return (
                self._crop_indices(h, self.crop_size),
                self._crop_indices(w, self.crop_size),
                self._crop_indices(d, self.crop_size),
            )

        crop = self.crop_size
        h, w, d = label.shape

        # Centroid of tumor bounding box
        ch = int((coords[0].min() + coords[0].max()) // 2)
        cw = int((coords[1].min() + coords[1].max()) // 2)
        cd = int((coords[2].min() + coords[2].max()) // 2)

        # Random offset so tumor isn't always perfectly centred
        if self.random_crop and self.roi_max_offset > 0:
            off = self.roi_max_offset
            ch += int(np.random.randint(-off, off + 1))
            cw += int(np.random.randint(-off, off + 1))
            cd += int(np.random.randint(-off, off + 1))

        def _clamp(center: int, size: int) -> tuple[int, int]:
            start = center - crop // 2
            start = max(0, min(start, size - crop))
            return start, start + crop

        return _clamp(ch, h), _clamp(cw, w), _clamp(cd, d)

    def _pad_to_crop(self, array: np.ndarray, spatial_axes: tuple[int, int, int]) -> np.ndarray:
        pad_widths = [(0, 0)] * array.ndim
        for axis in spatial_axes:
            deficit = self.crop_size - array.shape[axis]
            if deficit > 0:
                before = deficit // 2
                pad_widths[axis] = (before, deficit - before)
        if any(p != (0, 0) for p in pad_widths):
            array = np.pad(array, pad_widths, mode="constant", constant_values=0)
        return array

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        volume = np.load(self.vol_paths[index], mmap_mode="r")  # (H, W, D, 4)
        volume = self._pad_to_crop(np.asarray(volume), spatial_axes=(0, 1, 2))

        if self.volume_only:
            h, w, d = volume.shape[:3]
            h0, h1 = self._crop_indices(h, self.crop_size)
            w0, w1 = self._crop_indices(w, self.crop_size)
            d0, d1 = self._crop_indices(d, self.crop_size)
            volume = volume[h0:h1, w0:w1, d0:d1, :]
            volume = np.ascontiguousarray(volume.transpose(3, 0, 1, 2)).astype(np.float32)
            return torch.from_numpy(volume), torch.empty(0)

        label = np.load(self.seg_paths[index], mmap_mode="r")  # (H, W, D)
        label = self._pad_to_crop(np.asarray(label), spatial_axes=(0, 1, 2))

        h, w, d = label.shape
        use_roi = (
            self.random_crop
            and self.roi_crop_ratio > 0.0
            and np.random.random() < self.roi_crop_ratio
        )
        if use_roi:
            (h0, h1), (w0, w1), (d0, d1) = self._roi_crop_indices(label)
        else:
            h0, h1 = self._crop_indices(h, self.crop_size)
            w0, w1 = self._crop_indices(w, self.crop_size)
            d0, d1 = self._crop_indices(d, self.crop_size)

        volume = volume[h0:h1, w0:w1, d0:d1, :]
        label = label[h0:h1, w0:w1, d0:d1]

        volume = np.ascontiguousarray(volume.transpose(3, 0, 1, 2)).astype(np.float32)  # (4, H, W, D)
        volume_t = torch.from_numpy(volume)

        if self.subregion_based:
            # Returns (4, H, W, D) float32: [BG, NCR, ED, ET] binary masks.
            # Mutually exclusive subregions — NCR has its own channel and gradient.
            mask = torch.from_numpy(seg_to_subregions(label))
        elif self.region_based:
            # Returns (3, H, W, D) float32: [WT, TC, ET] binary masks.
            # Works with raw BraTS labels {0,1,2,4} or remapped {0,1,2,3}.
            mask = torch.from_numpy(seg_to_regions(label))
        else:
            label = label.astype(np.int64, copy=False)
            if np.any((label < 0) | (label >= self.num_classes)):
                bad = np.unique(label[(label < 0) | (label >= self.num_classes)]).tolist()
                raise ValueError(f"Invalid label values {bad} in {self.seg_paths[index]}, expected [0, {self.num_classes - 1}]")
            label_t = torch.from_numpy(label)
            mask = torch.nn.functional.one_hot(label_t, num_classes=self.num_classes).permute(3, 0, 1, 2).float()

        return volume_t, mask
